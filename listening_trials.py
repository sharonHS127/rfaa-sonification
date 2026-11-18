#!/usr/bin/env python3
"""
listening_trials.py -- simulated RF source -> sonification -> binaural (headphone) WAV files + answer key.

Pipeline per trial:  direction -> simulated 5-antenna RF (rfaa_sim) -> sonify (Python reference or the C
program) -> 4-channel B-Format audio (W,X,Y,Z) -> first-order binaural decode with HRIRs -> trial_XX.wav
(2 channels, 16 bit).  WEAR HEADPHONES.

Two ways to choose the source direction (listener convention: azimuth 0 = straight ahead, 90 = right,
180 = behind, 270 = left; elevation +90 = straight up, 0 = ear level, negative = below):

  1) random trials, answers hidden in answers.txt/answers.csv until you check:
       python3 listening_trials.py --random 10 --seed 1
  2) you set the direction(s) yourself (comma separated lists pair up one to one):
       python3 listening_trials.py --az 30 --el 20
       python3 listening_trials.py --az 0,90,200 --el 0,20,-10

After listening, fill responses.csv (trial,az,el -- a template is written) and compare:
       python3 listening_trials.py --check responses.csv

Control conditions (no RF, no sonification; the stimulus is rendered at the TRUE direction, same --random N and same --seed = same directions):
       python3 listening_trials.py --random 5 --seed 7 --reference direct --out ref_direct   # noise + nearest HRIR (best case)
       python3 listening_trials.py --random 5 --seed 7 --reference foa    --out ref_foa      # noise through the ideal first-order
                                                                                              # Ambisonic decoder (decoder limit only)
  --stimulus noise|tone|whitebw (default noise: pink noise 100 Hz-16 kHz; whitebw: flat noise inside --band, default 500,9000)
  --tone-hz 2000 (for --stimulus tone)   --lowpass HZ (low-pass the control stimulus, or the sonified audio, at HZ)

RF source type (default cw = Todd's single tone; the others test how much RF bandwidth the audio needs):
  --signal cw|noise|am   --bw HZ (noise bandwidth, e.g. 8000 / 40000 / 100000)   --am-hz HZ  --am-depth 0..1
  --phase zero|random    zero = Todd's zero-phase audio (broadband input becomes a 10 Hz pulse train);
                         random = same spectra, one fixed random phase per frequency (identical for all 4 channels, so the
                         direction information is unchanged), overlap-added -> smooth hiss.  A presentation layer, not in Todd.
Virtual head rotation (resolves front/back: the sound field is rotated by the head yaw before the binaural decode):
  --head-turn DEG        extra file per trial, trial_XX_turn.wav: the head sweeps 0 -> +DEG -> -DEG -> 0 (DEG to the right)
  --head-yaw A,B,...     extra static files trial_XX_yawA.wav, ...: head fixed turned A degrees to the right (negative = left)

Other options:  --out DIR (default trials)  --snr dB (default 40)  --T seconds (default 3)  --engine python|c
  --hrtf kemar_hrir.npz | file.sofa | synthetic     --el-min/--el-max (random mode, default -40/80)
  --tone-only [HZ]   keep only the tone (+-HZ, default 100) and drop the broadband residual of the sonification
  --tuner Hz (audio pitch shift, default -4000)  --distort (array imperfection, default 1)  --array-seed N

Needs the HRTF file made once by prepare_hrtf.py (or a SOFA file with `pip install sofar`).  Without any HRTF
the 'synthetic' fallback (spherical-head ITD/ILD, azimuth cues only, NO elevation cues) is used.
"""
import argparse
import csv
import os
import subprocess
import tempfile
import wave
from fractions import Fraction

import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

import sonify as S
import test_sonify as T


# --------------------------------------------------------------------------------------- conventions
# Internal (sonify.py / rfaa_sim.py): azimuth counter-clockwise from the front (+Y = left), elevation up.
# Listener convention used on the command line and in the answer files: azimuth clockwise from the front.
def az_cw_to_internal(az):
    return (-np.asarray(az, float)) % 360.0


def az_internal_to_cw(az):
    return (-np.asarray(az, float)) % 360.0


def unit(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def mean_dir(az_deg, el_deg):
    v = unit(np.asarray(az_deg), np.asarray(el_deg)).mean(axis=0)
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360), float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))


def gc_deg(a1, e1, a2, e2):
    """Great-circle angle between two directions (degrees)."""
    u, v = unit(a1, e1), unit(a2, e2)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))))


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------------------- HRTF sets
class HRIRSet:
    """Nearest-direction HRIR lookup.  az/el internal convention (counter-clockwise, SOFA/slab)."""

    def __init__(self, az, el, hrir, fs, name):
        self.v, self.hrir, self.fs, self.name = unit(az, el), hrir, float(fs), name

    def get(self, az, el):
        return self.hrir[int(np.argmax(self.v @ unit(az, el)))]            # (2, taps) left, right


class SyntheticHRIR:
    """Spherical-head model: Woodworth ITD + one-pole head-shadow low-pass on the far ear.  Azimuth cues only."""
    fs, name = 44100.0, "synthetic spherical head (azimuth only, no pinna / elevation cues)"

    def get(self, az, el, n=256, a=0.0875, c=343.0):
        lat = np.arcsin(np.clip(unit(az, el)[1], -1, 1))                    # lateral angle, + = source on the left
        k = np.arange(n)
        out = np.zeros((2, n))
        for ear, side in enumerate((+1.0, -1.0)):                           # ear 0 = left, 1 = right
            s = side * lat                                                  # > 0: this ear is on the source side
            if s >= 0:
                h = np.sinc(k - 40.0) * (1.0 + 0.3 * np.sin(s))
            else:                                                           # far ear: later, duller, quieter
                d = 40.0 + (a / c) * (-s + np.sin(-s)) * self.fs
                alpha = np.exp(-2 * np.pi * (2500.0 / (1 + 2.0 * -s)) / self.fs)
                lp = (1 - alpha) * alpha ** k
                h = fftconvolve(np.sinc(k - d), lp)[:n] * (0.6 + 0.4 * np.cos(s))
            out[ear] = h
        return out


def load_hrtf(spec):
    if spec == "synthetic":
        return SyntheticHRIR()
    if spec.lower().endswith(".npz"):
        d = np.load(spec)
        return HRIRSet(d["az"], d["el"], d["hrir"].astype(float), float(d["fs"]), "KEMAR (MIT, Gardner & Martin 1994)")
    if spec.lower().endswith(".sofa"):
        import sofar
        s = sofar.read_sofa(spec, verbose=False)
        pos = np.asarray(s.SourcePosition)                                  # az (ccw), el, r  (SOFA spherical)
        return HRIRSet(pos[:, 0], pos[:, 1], np.asarray(s.Data_IR), float(np.asarray(s.Data_SamplingRate).ravel()[0]),
                       os.path.basename(spec))
    raise ValueError("--hrtf must be kemar_hrir.npz, a .sofa file or 'synthetic'")


def virtual_speakers(n=50):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z ** 2)
    xyz = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    return np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])), np.degrees(np.arcsin(z)), xyz


def decoder_filters(hrtf, n_spk=50):
    """First-order Ambisonic -> binaural: max-rE decode to n_spk virtual loudspeakers, each rendered with the HRIR
    of its nearest measured direction, folded into 4 x 2 FIR filters (W,X,Y,Z -> left,right).
    A plane wave from d is encoded as [1, d_x, d_y, d_z] (= SN3D), which is what the sonified audio carries."""
    az, el, xyz = virtual_speakers(n_spk)
    h = np.stack([hrtf.get(a, e) for a, e in zip(az, el)])                  # (N, 2, taps)
    a1 = np.cos(np.radians(137.9 / 2.51))                                   # max-rE weight, order 1 (3-D)
    D = np.empty((4, n_spk))
    D[0] = 1.0 / n_spk
    D[1:] = (3.0 * a1 / n_spk) * xyz.T
    return np.einsum("bi,iet->bet", D, h)                                   # (4, 2, taps)


def rotate_yaw(x4, yaw_cw_deg):
    """Virtual head rotation of a first-order sound field (W,X,Y,Z).  yaw_cw_deg: head turned to the right (clockwise seen
    from above) by this angle, scalar or one value per sample.  A source at listener azimuth a then appears at a - yaw,
    i.e. at internal azimuth a_int + yaw, so (X,Y) are rotated counter-clockwise by +yaw.  W and Z do not change."""
    psi = np.radians(np.broadcast_to(np.asarray(yaw_cw_deg, float), (x4.shape[0],)))
    c, s = np.cos(psi), np.sin(psi)
    y = x4.copy()
    y[:, 1] = c * x4[:, 1] - s * x4[:, 2]
    y[:, 2] = s * x4[:, 1] + c * x4[:, 2]
    return y


def sweep_yaw(n, fs, deg):
    """Head-turn trajectory: 0 -> +deg -> -deg -> 0 (one sine period over the file)."""
    return deg * np.sin(2 * np.pi * np.arange(n) / n)


def _decode(x, filt):
    out = np.zeros((x.shape[0], 2))
    for b in range(4):
        for ear in range(2):
            out[:, ear] += fftconvolve(x[:, b], filt[b, ear])[:x.shape[0]]
    return out


def _finish(out, fs_hrtf, gain, fade=0.02):
    nf = int(fade * fs_hrtf)
    ramp = np.ones(len(out)); ramp[:nf] = np.linspace(0, 1, nf); ramp[-nf:] = np.linspace(1, 0, nf)
    return np.clip(out * ramp[:, None] * gain, -1, 1)


def render_binaural(audio4, fs_audio, filt, fs_hrtf, rms=0.1, fade=0.02, yaws=(None,)):
    """audio4 (n,4) at fs_audio -> list of stereo (m,2) at fs_hrtf, one per entry of `yaws` (None = head still;
    a number = fixed head yaw in degrees to the right; a function n,fs -> yaw per sample).  All versions of a trial use the
    gain of the head-still version (fixed RMS rms), so the left/right level difference and the loudness are comparable."""
    r = Fraction(int(round(fs_hrtf)), int(round(fs_audio))).limit_denominator(1000)
    x = resample_poly(audio4, r.numerator, r.denominator, axis=0)
    ref = _decode(x, filt)
    nf = int(fade * fs_hrtf)
    ramp = np.ones(len(ref)); ramp[:nf] = np.linspace(0, 1, nf); ramp[-nf:] = np.linspace(1, 0, nf)
    gain = rms / np.sqrt(np.mean((ref * ramp[:, None]) ** 2))
    outs = []
    for y in yaws:
        if y is None:
            o = ref
        else:
            yy = y(len(x), fs_hrtf) if callable(y) else y
            o = _decode(rotate_yaw(x, yy), filt)
        outs.append(_finish(o, fs_hrtf, gain, fade))
    return outs


def random_phase_audio(audio4, L, seed=777):
    """Presentation layer.  The sonified audio is a sequence of chunks of L samples whose spectra are real (zero phase),
    so a broadband spectrum becomes one short pulse per chunk.  Here every chunk gets the SAME random phase per frequency
    bin (identical for the 4 channels, so channel ratios = direction information are untouched), each chunk is read as one
    period of a periodic signal, tiled twice, Hann-windowed and overlap-added (hop L): no boundary clicks, tones stay
    continuous, the amplitude modulation from chunk to chunk is kept."""
    n = audio4.shape[0]
    nc = n // L
    rng = np.random.default_rng(seed)
    ph = np.zeros(L)
    half = rng.uniform(0, 2 * np.pi, L // 2 - 1)
    ph[1:L // 2] = half
    ph[L // 2 + 1:] = -half[::-1]                                            # conjugate symmetry -> real output
    P = np.exp(1j * ph)[:, None]
    win = np.hanning(2 * L + 1)[:2 * L] if False else 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(2 * L) / (2 * L))   # periodic Hann
    out = np.zeros(((nc + 1) * L, 4))
    for c in range(nc):
        seg = np.fft.ifft(np.fft.fft(audio4[c * L:(c + 1) * L], axis=0) * P, axis=0).real
        out[c * L:c * L + 2 * L] += np.tile(seg, (2, 1)) * win[:, None]
    return out[L // 2:L // 2 + nc * L]


def lowpass(x, fs, fc, order=8):
    """Zero-phase Butterworth low-pass (steep: 8th order, applied forward and backward)."""
    return sosfiltfilt(butter(order, fc, btype="low", fs=fs, output="sos"), x, axis=0)


def make_stimulus(kind, fs, T, tone_hz, rng, band=(500.0, 9000.0)):
    n = int(round(T * fs))
    if kind == "tone":
        return np.sin(2 * np.pi * tone_hz * np.arange(n) / fs)
    if kind == "whitebw":                                                    # flat spectrum inside `band`, nothing outside
        w = rng.standard_normal(n)
        return sosfiltfilt(butter(8, band, btype="band", fs=fs, output="sos"), w)
    X = np.fft.rfft(rng.standard_normal(n))                                  # pink noise, 100 Hz - 16 kHz
    f = np.fft.rfftfreq(n, 1.0 / fs)
    g = np.zeros_like(f)
    band = (f >= 100.0) & (f <= min(16000.0, 0.49 * fs))
    g[band] = 1.0 / np.sqrt(f[band])
    return np.fft.irfft(X * g, n)


def reference_trial(hrtf, filt, mode, az_int, el, stim, rms=0.1, fade=0.02):
    """Control condition at the true direction.  direct: nearest measured HRIR.  foa: the same first-order Ambisonic
    decoder as the sonified trials, fed with an ideal plane wave [1, x, y, z]."""
    if mode == "direct":
        h = hrtf.get(az_int, el)                                             # (2, taps)
    else:
        h = np.einsum("b,bet->et", np.concatenate([[1.0], unit(az_int, el)]), filt)
    out = np.stack([fftconvolve(stim, h[e])[:len(stim)] for e in range(2)], axis=1)
    nf = int(fade * hrtf.fs)
    ramp = np.ones(len(out)); ramp[:nf] = np.linspace(0, 1, nf); ramp[-nf:] = np.linspace(1, 0, nf)
    out *= ramp[:, None]
    return np.clip(out * (rms / np.sqrt(np.mean(out ** 2))), -1, 1)


def keep_tone(audio, fs, half_width_hz):
    """Keep only the spectrum within +-half_width_hz of the strongest line (all channels share one mask), so the
    broadband residual of the sonification (whose channel ratios do not describe the source) is removed."""
    n = len(audio)
    X = np.fft.rfft(audio, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    p = (np.abs(X) ** 2).sum(axis=1)
    p[f < 100.0] = 0.0
    f0 = f[int(np.argmax(p))]
    d = np.abs(f - f0)
    mask = np.where(d <= half_width_hz, 0.5 * (1 + np.cos(np.pi * d / half_width_hz)), 0.0)    # raised-cosine window
    return np.fft.irfft(X * mask[:, None], n=n, axis=0)


def write_wav16(path, x, fs):
    pcm = np.round(np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1]); w.setsampwidth(2); w.setframerate(int(round(fs)))
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------------------- RF -> sonified audio
def audio_direction(audio4):
    """Direction carried by the sonified audio: intensity vector sum(W * [X,Y,Z]) over the whole file (internal az, el).
    This is what the binaural decoder renders.  For broadband input it is far more reliable than the per-channel spectral
    peaks sonify.py reports (those can jump to different bins for noise)."""
    m = (audio4[:, [0]] * audio4[:, 1:]).sum(axis=0)
    return float(np.degrees(np.arctan2(m[1], m[0])) % 360), float(np.degrees(np.arctan2(m[2], np.hypot(m[0], m[1]))))


def signal_kwargs(a):
    """Options for rfaa_sim.synth_timeseries; empty for cw so the old random draws / results stay unchanged."""
    if a.signal == "cw":
        return {}
    if a.signal == "noise":
        return dict(signal="noise", bw=a.bw)
    return dict(signal="am", am_hz=a.am_hz, am_depth=a.am_depth)


def sonify_trial(arr, coeff, az_int, el, seed, a):
    """Returns audio (n,4) at 30 kHz, and the RF-side and sonified-side direction estimates (internal az, el)."""
    x = T.recording(arr, az_int, seed, T=a.T, f0=a.f0, snr_db=a.snr, el=el, **signal_kwargs(a))
    p = S.SonifyParams(tuner=a.tuner, sign_mode="aligned", svd_method="gram").validate()
    if a.engine == "python":
        audio, rf, son, _ = S.sonify(x, coeff, p)
        est = {k: mean_dir(np.degrees(S.aoa_rad(v)), np.degrees(S.elevation_rad(v))) for k, v in (("rf", rf), ("son", son))}
        if a.signal != "cw":
            est["son"] = audio_direction(audio)
        return audio, est["rf"], est["son"]
    # --- C engine: write the chunk-layout file, run ./sonify_cli, read raw float32 audio and the CSV log
    n = x.shape[1] // p.chunk_len
    with tempfile.TemporaryDirectory() as d:
        np.ascontiguousarray(x[:, :n * p.chunk_len].reshape(5, n, p.chunk_len).transpose(1, 0, 2), dtype=np.complex64).tofile(f"{d}/rf.f32")
        np.ascontiguousarray(coeff, dtype=np.float32).tofile(f"{d}/coeff.f32")
        r = subprocess.run([a.cli, "--coeff", f"{d}/coeff.f32", "-i", f"{d}/rf.f32", "-o", "-", "--out-format", "f32",
                            "--gain", "1", "--tuner", str(a.tuner), "--quiet", "--log", f"{d}/log.csv"],
                           capture_output=True, check=True)
        audio = np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, 4).astype(float)
        lg = np.genfromtxt(f"{d}/log.csv", delimiter=",", names=True)
    son = audio_direction(audio) if a.signal != "cw" else mean_dir(lg["az_son_deg"], lg["el_son_deg"])
    return audio, mean_dir(lg["az_rf_deg"], lg["el_rf_deg"]), son


# --------------------------------------------------------------------------------------- answer files / check
FIELDS = ["trial", "file", "true_az", "true_el", "rf_est_az", "rf_est_el", "son_est_az", "son_est_el"]


def write_answers(out, rows):
    with open(f"{out}/answers.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(FIELDS)
        for r in rows:
            w.writerow([r["trial"], r["file"]] + [f"{r[k]:.1f}" for k in FIELDS[2:]])
    with open(f"{out}/answers.txt", "w") as f:
        f.write("ANSWER KEY (az: 0 = front, 90 = right, 180 = behind, 270 = left; el: + up).  Do not open before listening.\n")
        f.write("true = direction used in the simulation | rf_est = estimated from the RF array before sonification |\n")
        f.write("son_est = estimated after sonification (what the audio carries)\n\n")
        f.write(f"{'trial':>5} {'file':<22} {'true az/el':>13} {'rf_est az/el':>14} {'son_est az/el':>15}\n")
        for r in rows:
            f.write(f"{r['trial']:>5} {r['file']:<22} {r['true_az']:6.1f}/{r['true_el']:5.1f} {r['rf_est_az']:7.1f}/{r['rf_est_el']:5.1f} "
                    f"{r['son_est_az']:8.1f}/{r['son_est_el']:5.1f}\n")
    with open(f"{out}/responses.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["trial", "az", "el"])
        for r in rows:
            w.writerow([r["trial"], "", ""])


def check(resp_path, out):
    ans = {int(r["trial"]): r for r in csv.DictReader(open(f"{out}/answers.csv"))}
    print("trial |  you (az/el)  | true (az/el)  | err vs true: az  el  angle | angle vs RF est | angle vs sonified | front-back flip?")
    errs, errs_rf, errs_son = [], [], []
    for r in csv.DictReader(open(resp_path)):
        if r["az"].strip() == "" or r["el"].strip() == "":
            continue
        t = int(r["trial"]); a = ans[t]
        ga, ge = float(r["az"]), float(r["el"])
        ta, te = float(a["true_az"]), float(a["true_el"])
        ra, re_ = float(a["rf_est_az"]), float(a["rf_est_el"])
        sa, se = float(a["son_est_az"]), float(a["son_est_el"])
        e_son = gc_deg(az_cw_to_internal(ga), ge, az_cw_to_internal(sa), se)
        e_gc, e_rf = gc_deg(az_cw_to_internal(ga), ge, az_cw_to_internal(ta), te), gc_deg(az_cw_to_internal(ga), ge, az_cw_to_internal(ra), re_)
        # front/back confusion: the mirror image about the ear axis (az -> 180 - az, same elevation) is closer than the truth
        ga_m = (180.0 - ga) % 360.0
        near_ear_axis = min(abs(wrap180(ta - 90.0)), abs(wrap180(ta + 90.0))) < 30.0     # mirror ~ itself there: not meaningful
        flip = (not near_ear_axis) and gc_deg(az_cw_to_internal(ga_m), ge, az_cw_to_internal(ta), te) < e_gc
        errs.append(e_gc); errs_rf.append(e_rf); errs_son.append(e_son)
        print(f"{t:5d} | {ga:6.1f}/{ge:5.1f} | {ta:6.1f}/{te:5.1f} | {wrap180(ga - ta):7.1f} {ge - te:5.1f} {e_gc:6.1f} | {e_rf:9.1f} | {e_son:11.1f} | {'n/a' if near_ear_axis else ('yes' if flip else 'no')}")
    if errs:
        print(f"\n{len(errs)} trials: median angle error vs truth {np.median(errs):.1f} deg (mean {np.mean(errs):.1f}), "
              f"vs RF estimate {np.median(errs_rf):.1f} deg, vs sonified estimate {np.median(errs_son):.1f} deg")


# --------------------------------------------------------------------------------------- main
def parse_list(s):
    return [float(v) for v in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", type=int, metavar="N", help="N random directions (answers hidden)")
    ap.add_argument("--seed", type=int, default=0, help="random-mode seed (same seed = same directions)")
    ap.add_argument("--az", help="azimuth(s) in listener convention, e.g. 30 or 0,90,200")
    ap.add_argument("--el", help="elevation(s), same count as --az")
    ap.add_argument("--el-min", type=float, default=-40.0)
    ap.add_argument("--el-max", type=float, default=80.0)
    ap.add_argument("--out", default="trials")
    ap.add_argument("--check", metavar="RESPONSES.csv", help="compare your responses with the answer key in --out")
    ap.add_argument("--snr", type=float, default=40.0)
    ap.add_argument("--T", type=float, default=3.0, help="seconds of RF per trial")
    ap.add_argument("--f0", type=float, default=None)
    ap.add_argument("--tuner", type=float, default=-4000.0)
    ap.add_argument("--distort", type=float, default=1.0)
    ap.add_argument("--array-seed", type=int, default=0)
    ap.add_argument("--engine", choices=["python", "c"], default="python")
    ap.add_argument("--cli", default="./sonify_cli", help="path of the compiled C program (--engine c)")
    ap.add_argument("--hrtf", default="kemar_hrir.npz")
    ap.add_argument("--reference", choices=["off", "direct", "foa"], default="off",
                    help="control condition: render a stimulus at the true direction instead of the sonified audio")
    ap.add_argument("--stimulus", choices=["noise", "tone", "whitebw"], default="noise",
                    help="control stimulus: pink noise 100 Hz-16 kHz | tone | whitebw = flat-spectrum noise inside --band")
    ap.add_argument("--band", default="500,9000", help="LOW,HIGH Hz of --stimulus whitebw (default 500,9000 = the band of the sonified audio)")
    ap.add_argument("--lowpass", type=float, default=None, metavar="HZ", help="low-pass the stimulus (controls) or the sonified audio at HZ")
    ap.add_argument("--tone-hz", type=float, default=2000.0)
    ap.add_argument("--tone-only", type=float, nargs="?", const=100.0, default=None, metavar="HALFWIDTH_HZ",
                    help="keep only +-HALFWIDTH_HZ (default 100) around the tone; removes the sonification's broadband residual")
    ap.add_argument("--signal", choices=["cw", "noise", "am"], default="cw", help="RF source type (default cw)")
    ap.add_argument("--bw", type=float, default=40000.0, help="noise bandwidth in Hz (--signal noise)")
    ap.add_argument("--am-hz", type=float, default=10.0, help="modulation frequency in Hz (--signal am)")
    ap.add_argument("--am-depth", type=float, default=0.5, help="modulation depth 0..1 (--signal am)")
    ap.add_argument("--phase", choices=["zero", "random"], default="zero", help="audio phase: zero (Todd) or random (smooth hiss)")
    ap.add_argument("--head-turn", type=float, default=None, metavar="DEG", help="extra file with a head sweep 0 -> +DEG -> -DEG -> 0")
    ap.add_argument("--head-yaw", default=None, metavar="A,B", help="extra static files with the head turned A, B ... degrees right")
    a = ap.parse_args()

    if a.check:
        check(a.check, a.out)
        raise SystemExit

    if a.random:
        rng = np.random.default_rng(a.seed)
        az_cw = rng.uniform(0, 360, a.random)
        el = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(a.el_min)), np.sin(np.radians(a.el_max)), a.random)))
        hidden = True
    elif a.az is not None and a.el is not None:
        az_cw, el = np.array(parse_list(a.az)), np.array(parse_list(a.el))
        if len(az_cw) != len(el):
            raise SystemExit("--az and --el need the same number of values")
        hidden = False
    else:
        raise SystemExit("choose --random N  or  --az ... --el ...   (see --help)")

    if not os.path.exists(a.hrtf) and a.hrtf != "synthetic":
        print(f"HRTF file '{a.hrtf}' not found -> using the synthetic fallback (azimuth cues only, no elevation).\n"
              f"Run prepare_hrtf.py once to create it.")
        a.hrtf = "synthetic"
    hrtf = load_hrtf(a.hrtf)
    filt = decoder_filters(hrtf)
    os.makedirs(a.out, exist_ok=True)
    if a.reference == "off":
        print(f"HRTF: {hrtf.name}; building the simulated array (about 15 s) ...")
        arr, coeff = T.make_array3d(seed=a.array_seed, distort=a.distort)
    else:
        print(f"HRTF: {hrtf.name}; CONTROL CONDITION '{a.reference}': {a.stimulus} at the true direction, no RF / sonification.")

    if a.signal != "cw" and a.tone_only:
        print("note: --tone-only only makes sense for --signal cw; ignored.")
        a.tone_only = None
    if a.reference == "direct" and (a.head_turn or a.head_yaw):
        print("note: head rotation needs the Ambisonic path; ignored for --reference direct.")
        a.head_turn = a.head_yaw = None
    yaw_list = parse_list(a.head_yaw) if a.head_yaw else []
    variants = [("", None)] + [(f"_yaw{y:+g}", y) for y in yaw_list]
    if a.head_turn:
        variants.append(("_turn", lambda n, fs, d=a.head_turn: sweep_yaw(n, fs, d)))
    band = tuple(float(v) for v in a.band.split(","))
    fsA = S.SonifyParams().fsA
    Lc = S.SonifyParams().out_len

    rows = []
    for i, (azc, e) in enumerate(zip(az_cw, el), start=1):
        if a.reference == "direct":
            rng_s = np.random.default_rng(10_000 + 100 * a.seed + i)
            stim = make_stimulus(a.stimulus, hrtf.fs, a.T, a.tone_hz, rng_s, band)
            if a.lowpass:
                stim = lowpass(stim, hrtf.fs, a.lowpass)
            sts = [reference_trial(hrtf, filt, "direct", float(az_cw_to_internal(azc)), float(e), stim)]
            rf = son = (float(az_cw_to_internal(azc)), float(e))              # no estimation: the "estimates" are the truth
            variants = variants[:1]
        elif a.reference == "foa":
            rng_s = np.random.default_rng(10_000 + 100 * a.seed + i)
            stim = make_stimulus(a.stimulus, hrtf.fs, a.T, a.tone_hz, rng_s, band)
            if a.lowpass:
                stim = lowpass(stim, hrtf.fs, a.lowpass)
            enc = np.concatenate([[1.0], unit(float(az_cw_to_internal(azc)), float(e))])
            sts = render_binaural(stim[:, None] * enc[None, :], hrtf.fs, filt, hrtf.fs, yaws=[v for _, v in variants])
            rf = son = (float(az_cw_to_internal(azc)), float(e))
        else:
            audio, rf, son = sonify_trial(arr, coeff, float(az_cw_to_internal(azc)), float(e), 100 * a.seed + i, a)
            if a.phase == "random":
                audio = random_phase_audio(audio, Lc)
            if a.lowpass:
                audio = lowpass(audio, fsA, a.lowpass)
            if a.tone_only:
                audio = keep_tone(audio, fsA, a.tone_only)
            sts = render_binaural(audio, fsA, filt, hrtf.fs, yaws=[v for _, v in variants])
        base = f"trial_{i:02d}" if hidden else f"trial_{i:02d}_az{azc:05.1f}_el{e:+05.1f}"
        for (suffix, _), st in zip(variants, sts):
            write_wav16(f"{a.out}/{base}{suffix}.wav", st, hrtf.fs)
        name = f"{base}.wav"
        rows.append(dict(trial=i, file=name, true_az=azc % 360, true_el=float(e),
                         rf_est_az=float(az_internal_to_cw(rf[0])), rf_est_el=rf[1],
                         son_est_az=float(az_internal_to_cw(son[0])), son_est_el=son[1]))
        print(f"  trial {i:02d} written" + (f" (+{len(sts) - 1} head-rotation file(s))" if len(sts) > 1 else "") +
              ("" if hidden else f": true az {azc:.1f} el {e:.1f} -> RF estimate {rows[-1]['rf_est_az']:.1f}/{rf[1]:.1f}, sonified {rows[-1]['son_est_az']:.1f}/{son[1]:.1f}"))
    write_answers(a.out, rows)
    with open(f"{a.out}/config.txt", "w") as f:
        f.write(" ".join(__import__("sys").argv) + "\n")
        f.write(f"signal={a.signal} bw={a.bw} am_hz={a.am_hz} am_depth={a.am_depth} phase={a.phase} reference={a.reference} "
                f"head_turn={a.head_turn} head_yaw={a.head_yaw} stimulus={a.stimulus} band={a.band} lowpass={a.lowpass}\n")
    if len(variants) > 1:
        with open(f"{a.out}/responses_head.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["trial", "az", "el"])
            for r in rows:
                w.writerow([r["trial"], "", ""])
        print(f"head-rotation files written. Answer ONCE with the still file (responses.csv), then again after also listening to the "
              f"_yaw/_turn files (responses_head.csv); compare the two with --check.")
    print(f"\nwrote {len(rows)} wav file(s) to {a.out}/  (headphones!)")
    print(f"answer key: {a.out}/answers.txt" + ("   <- do not open before you have listened" if hidden else ""))
    print(f"fill {a.out}/responses.csv with what you heard, then:  python3 listening_trials.py --check {a.out}/responses.csv --out {a.out}")

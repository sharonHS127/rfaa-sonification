#!/usr/bin/env python3
"""
export_wav.py -- simulated RF -> sonification (sonify.py) -> WAV files you can listen to.

  python3 export_wav.py                          # source az 60, el 35, 3 s, SNR 40 dB -> wav/
  python3 export_wav.py --az 200 --el 0 --snr 60 --T 3 --f0 1000

Writes (30 kHz, 16-bit, one common gain for all channels so their level ratios are kept):
  wav/<name>_WXYZ.wav   4-channel B-Format (W,X,Y,Z)      -> for a later binaural/ambisonic decoder
  wav/<name>_W.wav      mono W channel only                -> plays in any player
  wav/<name>_stereo.wav quick stereo preview, L = W+Y, R = W-Y (crude, NOT an HRTF render)
"""
import argparse
import os
import wave

import numpy as np

import sonify as S
import test_sonify as T


def write_wav(path, x, fs):
    """x: (n,) or (n,ch) float, already scaled to |x| <= 1."""
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1]); w.setsampwidth(2); w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", type=float, default=60.0)
    ap.add_argument("--el", type=float, default=35.0)
    ap.add_argument("--snr", type=float, default=40.0)
    ap.add_argument("--T", type=float, default=3.0, help="seconds of RF to simulate")
    ap.add_argument("--f0", type=float, default=None, help="CW offset in the RF baseband [Hz] (default: simulator default)")
    ap.add_argument("--tuner", type=float, default=-4000.0)
    ap.add_argument("--out", default="wav")
    a = ap.parse_args()

    arr, coeff = T.make_array3d()
    p = S.SonifyParams(tuner=a.tuner, sign_mode="aligned", svd_method="gram").validate()
    x = T.recording(arr, a.az, 11, T=a.T, f0=a.f0, snr_db=a.snr, el=a.el)
    audio, rf, son, kept = S.sonify(x, coeff, p)             # (n,4)
    g = 0.9 / np.abs(audio).max()
    os.makedirs(a.out, exist_ok=True)
    name = f"az{a.az:g}_el{a.el:g}_snr{a.snr:g}"
    write_wav(f"{a.out}/{name}_WXYZ.wav", audio * g, p.fsA)
    write_wav(f"{a.out}/{name}_W.wav", audio[:, 0] * g, p.fsA)
    st = np.column_stack([audio[:, 0] + audio[:, 2], audio[:, 0] - audio[:, 2]])
    write_wav(f"{a.out}/{name}_stereo.wav", st * (0.9 / np.abs(st).max()), p.fsA)
    print(f"{audio.shape[0] / p.fsA:.1f} s of audio at {p.fsA:.0f} Hz, gain {g:.3g}; wrote {a.out}/{name}_*.wav")
    print("RF AoA %.1f deg, sonified AoA %.1f deg (circular mean over chunks)"
          % (S.circ_mean_deg(S.aoa_rad(rf)), S.circ_mean_deg(S.aoa_rad(son))))

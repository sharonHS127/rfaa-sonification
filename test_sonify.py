#!/usr/bin/env python3
"""
test_sonify.py -- sanity checks for the Python reference sonify.py (plain asserts, no pytest needed).

  python3 test_sonify.py            # run everything (~1 min)

1. shapes / Z channel / finiteness
2. tuner: where does a CW at RF offset +4 kHz land in the audio spectrum? (all three tuner branches)
3. svd vs gram decomposition agree in 'aligned' mode
4. RF AoA vs sonified AoA over many angles ('aligned' mode)  -- the check Todd did in Fig 5.9
5. informational: what the literal 'thesis' sign recipe does under numpy/LAPACK
"""
import numpy as np

import rfaa_sim as R
import sonify as S

CFG = dict(fs=12e3, T=10, Trec=1, snr_db=40.0)      # only used to fit the 5x4 coefficient matrix
P = S.SonifyParams()


def make_array(seed=0, distort=1.0):
    arr = R.SimArray(seed, az_err_deg=8.0 * distort, gain_sigma=0.3 * distort,
                     ripple_amp=0.30 * distort, spike_amp=0.5 * distort)
    rng = np.random.default_rng(1000 + seed)
    coeff, _, _ = R.optimize(arr, rng, CFG)
    return arr, coeff


def make_array3d(seed=0, distort=1.0):
    """Same kind of array, but with elevation-dependent distortion and a coefficient matrix fitted on the
    whole sphere (W,X,Y,Z all non-zero).  The calibration sim uses a low fs: only +-100 Hz matters."""
    arr = R.SimArray(seed, az_err_deg=8.0 * distort, gain_sigma=0.3 * distort, ripple_amp=0.30 * distort,
                     spike_amp=0.5 * distort, el_ripple_amp=0.15 * distort)
    rng = np.random.default_rng(1000 + seed)
    coeff = R.optimize_sphere(arr, rng, dict(fs=2e3, T=10, Trec=1, snr_db=40.0))[0]
    return arr, coeff


def recording(arr, angle, seed, T=3.0, f0=None, snr_db=40.0, el=0.0, **signal):
    """signal: signal='cw'|'noise'|'am' and bw / am_hz / am_depth, passed to rfaa_sim.synth_timeseries."""
    rng = np.random.default_rng(seed)
    pw = arr.true_power(np.array([angle]), rng, el_deg=el)[0]
    return R.synth_timeseries(pw, rng, P.fsU, T, 1.0, snr_db, f0=f0, **signal)


def test_shapes(arr, coeff):
    x = recording(arr, 60, 1)
    audio, rf, son, kept = S.sonify(x, coeff, P)
    n = x.shape[1] // P.chunk_len
    assert audio.shape == (n * P.out_len, 4) == (30 * 3000, 4)
    assert np.all(audio[:, 3] == 0) and np.all(np.isfinite(audio))
    assert rf.shape == son.shape == kept.shape == (n, 4)
    print(f"[1] shapes ok: {n} chunks -> {audio.shape[0]} audio samples x 4 channels, Z == 0")


def test_tuner(arr, coeff):
    """CW at +4000 Hz = RF bin j0 = N/2 + 50 = 800 (0-based).  Expected audio bin per branch."""
    x = recording(arr, 60, 2, T=0.1, f0=4000.0, snr_db=80.0)
    j0 = P.N // 2 + int(round(4000 / (P.fsU / P.N)))
    exp = {0.0: j0, -4000.0: j0 + 2 - 400, 4000.0: j0 + 400}
    got = {}
    for tuner in exp:
        p = S.SonifyParams(tuner=tuner)
        r = S.sonify_chunk(x, coeff, p)
        spec = np.abs(np.fft.fft(r.audio[:, 0]) / p.fsA)
        got[tuner] = int(np.argmax(spec[:P.N]))
        assert got[tuner] == exp[tuner], (tuner, got[tuner], exp[tuner])
        acs = S.place_spectrum(np.arange(1.0, P.N + 1), p)                  # conj-symmetry check
        k = np.arange(1, P.N)
        assert np.allclose(acs[k], acs[2 * P.N - k]) and acs[0] == 0 and acs[P.N] == 0
    shift_down = (got[0.0] - got[-4000.0]) * 10
    shift_up = (got[4000.0] - got[0.0]) * 10
    print(f"[2] tuner ok: RF bin {j0} -> audio bins {got}  (10 Hz/bin)")
    print(f"    tuner=-4000 Hz really shifts by {shift_down} Hz, tuner=+4000 Hz by +{shift_up} Hz "
          f"(Todd's negative branch uses Noffset-2 bins: asymmetric by 20 Hz)")


def test_gram_vs_svd(arr, coeff):
    x = recording(arr, 200, 3, T=0.5)
    pa = S.SonifyParams(sign_mode="aligned", svd_method="svd")
    pb = S.SonifyParams(sign_mode="aligned", svd_method="gram")
    a, ra, sa, ka = S.sonify(x, coeff, pa)
    b, rb, sb, kb = S.sonify(x, coeff, pb)
    err = np.max(np.abs(a - b)) / np.max(np.abs(a))
    assert (ka == kb).all() and err < 1e-9, err
    print(f"[3] svd vs gram (aligned): identical k, max relative diff {err:.1e}")


def test_aoa(arr, coeff, snr_db, limit_deg=None):
    """Compare per-chunk RF AoA with sonified AoA.  snr_db is the sim's definition (peak/noise-floor
    bin ratio for a power-1 antenna and a 1 s record); the 0.0125 s records of C.3 see 19 dB less."""
    angles = np.arange(0, 360, 30)
    rows = []
    for ang in angles:
        x = recording(arr, ang, 100 + int(ang), snr_db=snr_db)
        _, rf, son, kept = S.sonify(x, coeff, P)
        a_rf, a_son = S.aoa_rad(rf), S.aoa_rad(son)
        d = S.ang_diff_deg(a_son, a_rf)
        rows.append((ang, S.circ_mean_deg(a_rf), S.circ_mean_deg(a_son),
                     np.median(np.abs(d)), np.mean(np.abs(d) < 10), kept.mean(axis=0)))
    print(f"[4] RF AoA vs sonified AoA ('aligned' + 'gram'), sim SNR {snr_db} dB, 30 chunks per angle")
    print("    angle  RF-circmean  son-circmean  median|diff|  frac<10deg   mean k (W,X,Y,Z)")
    for ang, mr, ms, med, fr, kc in rows:
        print(f"    {ang:4d}   {mr:8.1f}    {ms:8.1f}      {med:6.2f}       {fr:5.2f}       {kc.round(1)}")
    worst = max(r[3] for r in rows)
    print(f"    worst median difference over angles: {worst:.2f} deg")
    if limit_deg is not None:
        assert worst < limit_deg, f"median |RF - sonified| = {worst:.2f} deg exceeds {limit_deg} deg"
        print(f"    assert < {limit_deg} deg: ok")


def test_thesis_sign_info(arr, coeff):
    print("[5] informational: literal Todd sign recipe ('thesis' + svd) under numpy/LAPACK, sim SNR 60 dB")
    pt = S.SonifyParams(sign_mode="thesis", svd_method="svd")
    for ang in (60, 150, 240, 330):                       # one angle per quadrant
        x = recording(arr, ang, 500 + ang, T=0.5, snr_db=60.0)
        _, rf, son, _ = S.sonify(x, coeff, pt)
        print(f"    transmit {ang:3d} deg: RF AoA {S.circ_mean_deg(S.aoa_rad(rf)):6.1f}  "
              f"thesis-recipe sonified AoA {S.circ_mean_deg(S.aoa_rad(son)):6.1f}  "
              f"sign of sonified (X,Y) peaks {np.sign(son[:, 1].mean()):+.0f},{np.sign(son[:, 2].mean()):+.0f}")


def test_npca_compat(arr, coeff):
    """n_pca=3 (Todd, horizontal) must equal the first three channels of n_pca=4, with Z exactly 0."""
    x = recording(arr, 100, 11, T=0.3, snr_db=50.0)
    a3 = S.sonify(x, coeff, S.SonifyParams(n_pca=3))[0]
    a4 = S.sonify(x, coeff, S.SonifyParams(n_pca=4))[0]
    assert np.array_equal(a3[:, :3], a4[:, :3]) and np.all(a3[:, 3] == 0)
    print("[6] n_pca=3 == first three channels of n_pca=4, Z == 0 for n_pca=3")


def test_3d(arr3, coeff3, snr_db, limit_deg=None):
    """Sources on the whole sphere.  Reports (a) array-side error: RF direction vs truth, (b) sonification
    error: sonified direction vs RF direction (great-circle degrees), (c) elevation sign agreement."""
    dirs = [(az, el) for el in (-60, -30, 0, 30, 60) for az in (0, 90, 180, 270)] + [(0, 90), (0, -90)]
    rows = []
    for az, el in dirs:
        x = recording(arr3, az, 700 + 7 * az + el, T=1.5, snr_db=snr_db, el=el)
        _, rf, son, kept = S.sonify(x, coeff3, P)                       # P: n_pca = 4
        v_rf, v_son = S.doa_vec(rf), S.doa_vec(son)
        m_rf, R_rf = S.mean_direction(v_rf)
        m_son, R_son = S.mean_direction(v_son)
        truth = S.unit_vec(az, el)
        rows.append((az, el, S.az_el_deg(m_rf), S.az_el_deg(m_son), float(S.great_circle_deg(m_rf, truth)),
                     float(np.median(S.great_circle_deg(v_son, v_rf))), R_son, kept.mean(axis=0)))
    print(f"[7] 3-D sources, sim SNR {snr_db} dB, 15 chunks per direction  (az/el in degrees)")
    print("    truth az,el   RF mean az,el    son mean az,el   array err  son-vs-RF  R_son  mean k (W,X,Y,Z)")
    for az, el, (raz, rel), (saz, sel), e_arr, e_son, r_son, kc in rows:
        print(f"    {az:4d},{el:4d}    {float(raz):6.1f},{float(rel):6.1f}   {float(saz):6.1f},{float(sel):6.1f}"
              f"    {e_arr:6.1f}    {e_son:6.2f}    {r_son:5.3f}   {kc.round(1)}")
    errs_arr = np.array([r[4] for r in rows]); errs_son = np.array([r[5] for r in rows])
    print(f"    array-side error vs truth: median {np.median(errs_arr):.1f} deg, max {errs_arr.max():.1f} deg")
    print(f"    sonified vs RF direction : median {np.median(errs_son):.2f} deg, worst {errs_son.max():.2f} deg")
    if limit_deg is not None:
        assert errs_son.max() < limit_deg, f"worst sonified-vs-RF error {errs_son.max():.2f} deg >= {limit_deg}"
        print(f"    assert worst sonified-vs-RF < {limit_deg} deg: ok")
    return errs_arr, errs_son


def test_signals(arr3, coeff3):
    """[8] RF source types: cw / AM / noise of 8, 40, 100 kHz.  Checks (a) the occupied RF bandwidth is the requested one,
    (b) the direction carried by the audio (intensity vector) agrees with the RF direction, (c) the CW draws are unchanged (old results valid)."""
    print("[8] source types (3-D array, SNR 40 dB, el 25 deg)")
    p = S.SonifyParams(sign_mode="aligned", svd_method="gram").validate()
    ref = R.synth_timeseries(np.ones(5), np.random.default_rng(3), P.fsU, 0.5, 1.0, 40.0, f0=5)
    old_rng = np.random.default_rng(3); f0 = 5
    t = np.arange(int(P.fsU * 0.5)) / P.fsU; ph = old_rng.uniform(0, 2 * np.pi, (5, 1))
    old = np.exp(1j * (2 * np.pi * f0 * t + ph)) + np.sqrt(P.fsU / 10 ** 4 / 2) * (
        old_rng.standard_normal((5, len(t))) + 1j * old_rng.standard_normal((5, len(t))))
    assert np.allclose(ref, old), "CW draw order changed"
    print("    CW time series identical to the pre-extension generator: ok")
    for name, kw, bw in [("cw", {}, None), ("am 10 Hz", dict(signal="am", am_hz=10.0), None),
                         ("noise 8k", dict(signal="noise", bw=8000.0), 8e3), ("noise 40k", dict(signal="noise", bw=40000.0), 4e4),
                         ("noise 100k", dict(signal="noise", bw=100000.0), 1e5)]:
        x = recording(arr3, 60, 4, T=2.0, snr_db=40.0, el=25.0, **kw)
        psd = np.abs(np.fft.fft(x[0])) ** 2
        k = 400                                                              # smooth, then threshold against the lowest 5 %
        sm = np.convolve(np.tile(psd, 3), np.ones(k) / k, mode="same")[len(psd):2 * len(psd)]
        occ = (sm > 10 * np.quantile(sm, 0.05)).sum() * (P.fsU / len(psd))
        audio, rf, son, kept = S.sonify(x, coeff3, p)
        m = (audio[:, [0]] * audio[:, 1:]).sum(0)                           # intensity vector = direction in the audio
        a_son = np.degrees(np.arctan2(m[1], m[0])) % 360
        a_rf = np.degrees(S.aoa_rad(rf)); a_rf = np.degrees(np.arctan2(np.sin(np.radians(a_rf)).mean(), np.cos(np.radians(a_rf)).mean())) % 360
        d = (a_son - a_rf + 180) % 360 - 180
        print(f"    {name:10s} occupied RF band ~{occ / 1e3:6.1f} kHz   RF az {a_rf:6.1f}  audio intensity az {a_son:6.1f}  (diff {d:+.1f})  PCA kept {kept.mean():.1f}")
        if bw:
            assert 0.6 * bw < occ < 1.5 * bw, f"{name}: occupied band {occ:.0f} Hz vs requested {bw:.0f}"
        assert abs(d) < 5.0, f"{name}: audio direction differs from RF direction by {d:.1f} deg"
    print("    ok")


if __name__ == "__main__":
    arr, coeff = make_array()
    test_shapes(arr, coeff)
    test_tuner(arr, coeff)
    test_gram_vs_svd(arr, coeff)
    test_aoa(arr, coeff, snr_db=60.0, limit_deg=1.0)   # enough SNR: PCA keeps 1 component, must match RF AoA
    test_aoa(arr, coeff, snr_db=40.0)                  # noisy: informational (PCA keeps 3-6 components)
    test_thesis_sign_info(arr, coeff)
    arr3, coeff3 = make_array3d()                      # 3-D array: coefficient matrix fitted on the sphere
    test_npca_compat(arr, coeff)
    test_3d(arr3, coeff3, snr_db=60.0, limit_deg=1.0)  # enough SNR: sonified direction must match RF direction
    test_3d(arr3, coeff3, snr_db=45.0)                 # noisier: informational
    test_signals(arr3, coeff3)
    print("ALL CHECKS PASSED")

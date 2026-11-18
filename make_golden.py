#!/usr/bin/env python3
"""
make_golden.py -- write golden test vectors for the C port (sonify.h / sonify.c).

  python3 make_golden.py [outdir]        # default ./golden

3-D version: the source is at azimuth 60 deg, elevation +35 deg, n_pca = 4 (W,X,Y,Z all sonified) and the
5x4 coefficient matrix has a non-zero Z column (fitted on the whole sphere).  The horizontal-only
(n_pca = 3) golden set of the first version is kept in v1_horizontal/golden.

Cases (all 'aligned' sign mode; one chunk of 12000 complex samples x 5 antennas each):
  snr60_tunerM4000 : strong signal, PCA keeps 1 component, tuner -4000 Hz (Todd's default)
  snr40_tunerM4000 : noisy, PCA keeps several components (exercises the multi-component sum)
  snr60_tunerP4000 : tuner +4000 Hz branch
  snr60_tuner0     : no tuner branch
For the noisy case a chunk is chosen whose cumulative energy is >= 1 percentage point away from the
80 % threshold in all three channels, so float32 (C) and float64 (Python) cannot disagree on k.

Each case directory holds raw little-endian files, C order:
  iq.f32 (5,chunk_len,2)  coeff.f32 (5,4)  audio.f32 (out_len,4)  info.f32 (8)  kept.i32 (4)  params.json
"""
import json
import os
import sys

import numpy as np

import sonify as S
import test_sonify as T


def energy_margin(x, coeff, p):
    """Smallest distance (percentage points) between the cumulative PCA energy and the threshold."""
    bp = S.to_bformat_power(S.chunk_psd(x, p), coeff)
    m = 1e9
    for b in range(p.n_pca):
        B = bp[:, :, b] - bp[:, :, b].mean(axis=1, keepdims=True)
        w = np.sort(np.maximum(np.linalg.eigvalsh(B @ B.T), 0))[::-1]
        cum = 100 * np.cumsum(w) / w.sum()
        m = min(m, np.abs(cum - p.energy_pct).min())
    return m


def pick_chunk(x, coeff, p, min_margin):
    n = x.shape[1] // p.chunk_len
    for j in range(n):
        c = x[:, j * p.chunk_len:(j + 1) * p.chunk_len]
        if energy_margin(c, coeff, p) >= min_margin:
            return c, j
    raise RuntimeError("no chunk with enough margin; use a longer recording")


def verify_roundtrip(d, p):
    """Re-read the raw files, run the reference on the float32 input, compare with audio.f32."""
    iq = np.fromfile(f"{d}/iq.f32", dtype=np.complex64).reshape(5, p.chunk_len)
    coeff = np.fromfile(f"{d}/coeff.f32", dtype=np.float32).reshape(5, 4).astype(np.float64)
    audio = np.fromfile(f"{d}/audio.f32", dtype=np.float32).reshape(p.out_len, 4)
    info = np.fromfile(f"{d}/info.f32", dtype=np.float32)
    kept = np.fromfile(f"{d}/kept.i32", dtype=np.int32)
    r = S.sonify_chunk(iq.astype(np.complex128), coeff, p)
    e_audio = np.abs(r.audio - audio).max() / np.abs(audio).max()
    e_info = np.abs(np.concatenate([r.rf_max, r.son_max]) - info).max() / np.abs(info).max()
    assert (r.kept == kept).all()
    return e_audio, e_info


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "golden"
    arr, coeff = T.make_array3d()                               # sphere-fitted 5x4 matrix, Z column non-zero
    cases = [("snr60_tunerM4000", 60.0, -4000.0, 0.0), ("snr40_tunerM4000", 40.0, -4000.0, 1.0),
             ("snr60_tunerP4000", 60.0, 4000.0, 0.0), ("snr60_tuner0", 60.0, 0.0, 0.0)]
    for name, snr, tuner, margin in cases:
        p = S.SonifyParams(tuner=tuner, sign_mode="aligned", svd_method="gram").validate()
        x = T.recording(arr, 60, 7, T=1.0, snr_db=snr, el=35.0)  # 10 chunks, source at az 60 deg, el +35 deg
        chunk, j = pick_chunk(x, coeff, p, margin)
        d = os.path.join(out, name)
        r = S.export_golden(d, coeff, chunk, p)
        e_a, e_i = verify_roundtrip(d, p)
        print(f"{name:18s} chunk {j}  kept(W,X,Y,Z)={r.kept}  max|audio|={np.abs(r.audio).max():.3g}  "
              f"float32 round-trip rel.err audio {e_a:.1e} info {e_i:.1e}")

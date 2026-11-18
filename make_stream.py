#!/usr/bin/env python3
"""
make_stream.py -- write a simulated 5-antenna RF recording for sonify_cli (and coeff.f32).

  python3 make_stream.py                              # source az 60, el 35, 5 s -> stream/rf_chunk.f32 + coeff.f32
  python3 make_stream.py --az 0,90,180,270 --el 0 --T 2     # source jumps every 2 s (moving-source demo, 8 s)
  python3 make_stream.py --layout interleaved               # time-major 10 floats per sample (SDR style)
  python3 make_stream.py --s16                              # int16 samples (full scale = 1.0 -> peak scaled to 0.5)

Files (raw little endian):
  coeff.f32        5x4 float32 matrix fitted on the sphere (same as the golden set)
  rf_chunk.f32     chunk layout:       per chunk [ant0 I,Q ... ant4 I,Q] antenna-major     (sonify_cli --layout chunk)
  rf_inter.f32     interleaved layout: per sample [ant0 I,Q, ant1 I,Q, ... ant4 I,Q]       (--layout interleaved)
"""
import argparse
import os

import numpy as np

import sonify as S
import test_sonify as T

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--az", default="60", help="azimuth(s) in deg, comma separated; one segment of --T seconds each")
    ap.add_argument("--el", type=float, default=35.0)
    ap.add_argument("--T", type=float, default=5.0, help="seconds per segment")
    ap.add_argument("--snr", type=float, default=40.0)
    ap.add_argument("--f0", type=float, default=None)
    ap.add_argument("--layout", choices=["chunk", "interleaved"], default="chunk")
    ap.add_argument("--s16", action="store_true")
    ap.add_argument("--out", default="stream")
    a = ap.parse_args()

    arr, coeff = T.make_array3d()
    p = S.SonifyParams()
    segs = []
    for k, az in enumerate(float(s) for s in a.az.split(",")):
        x = T.recording(arr, az, 50 + k, T=a.T, f0=a.f0, snr_db=a.snr, el=a.el)
        n = x.shape[1] // p.chunk_len * p.chunk_len
        segs.append(x[:, :n])
    x = np.concatenate(segs, axis=1).astype(np.complex64)                  # (5, Ntot)
    os.makedirs(a.out, exist_ok=True)
    np.ascontiguousarray(coeff, dtype=np.float32).tofile(f"{a.out}/coeff.f32")

    if a.layout == "chunk":
        n = x.shape[1] // p.chunk_len
        data = np.ascontiguousarray(x.reshape(5, n, p.chunk_len).transpose(1, 0, 2))      # (chunk, ant, sample)
        name = "rf_chunk"
    else:
        data = np.ascontiguousarray(x.T)                                                  # (sample, ant)
        name = "rf_inter"
    if a.s16:
        v = np.stack([data.real, data.imag], axis=-1)
        v = np.round(v / np.abs(v).max() * 16384).astype("<i2")
        v.tofile(f"{a.out}/{name}.s16"); path = f"{a.out}/{name}.s16"
    else:
        data.tofile(f"{a.out}/{name}.f32"); path = f"{a.out}/{name}.f32"
    print(f"wrote {path} and {a.out}/coeff.f32: {x.shape[1] / p.fsU:.1f} s of RF, {x.shape[1] // p.chunk_len} chunks")

#!/usr/bin/env python3
"""
sonify.py  --  Python REFERENCE implementation of Todd's 2025 dissertation, Appendix C.3 "RFAA Full System".

Data flow per 0.1 s chunk (names in brackets are the MATLAB variables):

  iq chunk (5 ant x 12000 complex @ fsU=120 kHz)
    1. chunk_psd         : reshape to M=8 records of N=1500, FFT/fsU, fftshift, |.|^2/(N/fsU)   [SxxMat]
    2. to_bformat_power  : (M,N,5) @ coeff(5x4)  ->  (M,N,4)  W X Y Z power (can be negative)   [bFormatPower]
    3. signed_peak       : per record signed value at max |.| over N bins, mean over records      [RFMaxVal]
    4. pca_channel (W,X,Y,Z): subtract row mean, SVD, keep >80 % energy, sum of S*V' rows, sign fix [newPower]
    5. place_spectrum    : tuner shift + conjugate-symmetric 2N-point spectrum                     [acsPow]
    6. ifft * fsA        : 2N = 3000 audio samples @ fsA = 30 kHz                                  [timeSeriesA]
    Azimuth = atan2(Y, X) is computed twice: from RFMaxVal (before) and from maxVal (after sonification).

3-D extension (this version):  Todd's system is horizontal-plane only (coefficient matrix column Z = 0,
Z channel of the output = 0, only W,X,Y are sonified).  With n_pca = 4 the Z channel is sonified as well and
elevation = atan2(Z, hypot(X, Y)) becomes available.  n_pca = 3 reproduces the original behaviour exactly.
The 5x4 coefficient matrix must then be fitted on the whole sphere (rfaa_sim.optimize_sphere); with a
horizontal-only matrix (Z column 0) the Z channel simply comes out as 0 (k = 0 components).

This file is the golden reference for the C port (see sonify.h).  Two switches matter:

  sign_mode  'thesis'  : Todd's formula taken literally:  newPower = signB * sum(S*V' rows).
                         signB only fixes the sign of the DATA; the sign of S*V' is an arbitrary
                         convention of the SVD routine (u_i and v_i may both be negated).  Measured
                         with numpy/LAPACK (gesdd and gesvd alike): the sonified peak ends up always
                         NEGATIVE, i.e. positive peaks are flipped and negative peaks kept (see
                         test_sonify.py).  How MATLAB's svd behaves cannot be checked here, so this mode
                         is NOT a stable golden reference; it exists to diagnose his algorithm.
             'aligned' : each component's sign is fixed so that the sum of its record weights u_i is
                         >= 0.  Deterministic and implementation independent -> golden reference for C.
  svd_method 'svd'     : np.linalg.svd of the (M x N) matrix (what MATLAB does).
             'gram'    : eigen-decomposition of the small MxM matrix B B^T, Z = U^T B  (= S V^T).
                         This is what the C version will do (no LAPACK needed, 8x8 Jacobi).
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


# ----------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class SonifyParams:
    fsU: float = 120e3          # recorded (RF baseband) sample rate [Hz]
    fsA: float = 30e3           # audio sample rate [Hz]
    Tchunk: float = 0.1         # sonification window [s]
    tuner: float = -4000.0      # frequency shift [Hz]; <0 down, >0 up, 0 none
    energy_pct: float = 80.0    # keep principal components until more than this % energy is reached
    sign_mode: str = "aligned"  # 'thesis' | 'aligned'
    svd_method: str = "gram"    # 'svd' | 'gram'
    n_pca: int = 4              # B-Format channels that are sonified: 3 = W,X,Y (Todd, horizontal), 4 = W,X,Y,Z (3-D)

    # derived sizes (same formulas as C.3)
    @property
    def M(self):                # records per chunk
        return int(round(2 * self.fsU / self.fsA))

    @property
    def N(self):                # samples per record = PSD bins
        return int(round(self.Tchunk * self.fsA / 2))

    @property
    def chunk_len(self):        # RF samples per antenna per chunk (= M*N)
        return int(round(self.Tchunk * self.fsU))

    @property
    def out_len(self):          # audio samples per chunk (= 2N)
        return 2 * self.N

    @property
    def noffset(self):          # tuner expressed in audio-spectrum bins (df = 1/Tchunk)
        return int(round(abs(self.tuner) * self.Tchunk))

    def validate(self):
        if self.M * self.N != self.chunk_len:
            raise ValueError("M*N must equal Tchunk*fsU (check fsU/fsA/Tchunk)")
        if self.N % 2:
            raise ValueError("N must be even (fftshift logic of C.3)")
        if self.tuner < 0 and not (2 <= self.noffset <= self.N):
            raise ValueError("negative tuner needs 2 <= Noffset <= N")
        if self.tuner > 0 and not (0 <= self.noffset <= self.N - 2):
            raise ValueError("positive tuner needs Noffset <= N-2")
        if self.sign_mode not in ("thesis", "aligned") or self.svd_method not in ("svd", "gram"):
            raise ValueError("bad sign_mode / svd_method")
        if self.n_pca not in (3, 4):
            raise ValueError("n_pca must be 3 (W,X,Y) or 4 (W,X,Y,Z)")
        if self.sign_mode == "thesis" and self.svd_method != "svd":
            raise ValueError("'thesis' sign mode is only defined together with svd_method='svd'")
        return self


class ChunkResult(NamedTuple):
    audio: np.ndarray       # (out_len, 4) float64, columns W X Y Z (Z = 0 when n_pca = 3)
    rf_max: np.ndarray      # (4,)  record-averaged signed peak, before sonification   [RFMaxVal]
    son_max: np.ndarray     # (4,)  signed peak of the PCA-reconstructed spectrum (Z = 0 when n_pca = 3) [maxVal]
    kept: np.ndarray        # (4,)  number of principal components kept for W, X, Y, Z (0 for Z when n_pca = 3)


# ----------------------------------------------------------------------------- stages
def chunk_psd(x, p):
    """x: (5, chunk_len) complex -> Sxx (M, N, 5).
    MATLAB: reshape(chunk(:,b),[N,M]).' -> record m = samples m*N:(m+1)*N  (same as C-order reshape)."""
    M, N = p.M, p.N
    rec = np.asarray(x).reshape(x.shape[0], M, N)
    lsp = np.fft.fft(rec, axis=2) / p.fsU
    lsp = np.fft.fftshift(lsp, axes=2)            # = [lsp(:,N/2+1:end), lsp(:,1:N/2)] for even N
    sxx = (lsp * np.conj(lsp)).real / (N / p.fsU)
    return np.transpose(sxx, (1, 2, 0))           # (M, N, 5)


def to_bformat_power(sxx, coeff):
    """(M,N,5) @ (5,4) -> (M,N,4)."""
    return np.einsum("mnk,kb->mnb", sxx, coeff)


def signed_peak(bp):
    """Per record: value at max |.| over the N bins (keeping its sign) -> mean over records -> (4,)."""
    idx = np.argmax(np.abs(bp), axis=1)                             # (M,4)
    vals = np.take_along_axis(bp, idx[:, None, :], axis=1)[:, 0, :]  # (M,4)
    return vals.mean(axis=0)


def _matlab_sign(v):
    return float(np.sign(v))                      # sign(0) = 0, like MATLAB


def _n_keep(sv2, pct):
    """Number of components until the cumulative energy first EXCEEDS pct (= counter80-1 in C.3)."""
    tot = sv2.sum()
    if not np.isfinite(tot) or tot <= 0:
        return 0
    cum = 100.0 * np.cumsum(sv2) / tot
    above = np.nonzero(cum > pct)[0]
    return int(above[0]) + 1 if above.size else len(sv2)


def pca_channel(Bp, p):
    """Bp: (M,N) power of ONE B-Format channel -> (newPower (N,), k kept components)."""
    B = Bp - Bp.mean(axis=1, keepdims=True)                         # 'row mean subtract' (mean over N)
    if p.svd_method == "svd":
        U, S, Vh = np.linalg.svd(B, full_matrices=False)
        Z = S[:, None] * Vh                                         # scaled principal components (M,N)
        sv2 = S ** 2
    else:                                                           # 'gram': what the C code does
        G = B @ B.T                                                 # (M,M) symmetric
        w, U = np.linalg.eigh(G)                                    # ascending
        order = np.argsort(w)[::-1]
        w, U = np.maximum(w[order], 0.0), U[:, order]
        Z = U.T @ B                                                 # = S V^T
        sv2 = w
    k = _n_keep(sv2, p.energy_pct)
    if k == 0:
        return np.zeros(B.shape[1]), 0

    Bmean = B.mean(axis=0)                                          # (N,)
    signB = _matlab_sign(Bmean[np.argmax(np.abs(Bmean))])
    if p.sign_mode == "thesis":
        newp = signB * Z[:k].sum(axis=0)
    else:  # 'aligned'
        c = U[:, :k].sum(axis=0)                                    # sum of record weights per component
        s = np.where(c < 0, -1.0, 1.0)
        newp = (s[:, None] * Z[:k]).sum(axis=0)
    return newp, k


def place_spectrum(newp, p):
    """Build the conjugate-symmetric 2N-point spectrum incl. tuner shift (all three branches of C.3).
    newp is real, so conj() is a no-op.  Returns acsPow (2N,)."""
    N, no = p.N, p.noffset
    z = np.zeros
    if p.tuner == 0:
        acs = np.concatenate([[0.0], newp[1:], [0.0], newp[1:][::-1]])
    elif p.tuner > 0:
        d = newp[1:N - no]                                          # newPower(2:end-Noffset)
        acs = np.concatenate([[0.0], z(no), d, [0.0], d[::-1], z(no)])
    else:
        d = newp[no - 1:]                                           # newPower(Noffset:end)
        acs = np.concatenate([[0.0], d, z(no - 2), [0.0], z(no - 2), d[::-1]])
    assert acs.size == 2 * N
    return acs


def sonify_chunk(x, coeff, p):
    """x: (5, chunk_len) complex baseband, coeff: (5,4).  Returns ChunkResult."""
    sxx = chunk_psd(x, p)
    bp = to_bformat_power(sxx, coeff)
    rf_max = signed_peak(bp)
    audio = np.zeros((p.out_len, 4))
    son_max = np.zeros(4)
    kept = np.zeros(4, dtype=int)
    for b in range(p.n_pca):                                        # W, X, Y (, Z when n_pca = 4)
        newp, k = pca_channel(bp[:, :, b], p)
        kept[b] = k
        if k:
            son_max[b] = newp[np.argmax(np.abs(newp))]
        audio[:, b] = np.fft.ifft(place_spectrum(newp, p)).real * p.fsA
    return ChunkResult(audio, rf_max, son_max, kept)


def sonify(x, coeff, p):
    """Whole recording x (5, Ntot) complex -> audio (Nchunk*out_len, 4), per-chunk arrays."""
    p.validate()
    n = x.shape[1] // p.chunk_len
    audio = np.zeros((n * p.out_len, 4))
    rf = np.zeros((n, 4)); son = np.zeros((n, 4)); kept = np.zeros((n, 4), dtype=int)
    for j in range(n):
        r = sonify_chunk(x[:, j * p.chunk_len:(j + 1) * p.chunk_len], coeff, p)
        audio[j * p.out_len:(j + 1) * p.out_len] = r.audio
        rf[j], son[j], kept[j] = r.rf_max, r.son_max, r.kept
    return audio, rf, son, kept


# ----------------------------------------------------------------------------- angles
def aoa_rad(maxval):
    """AoA = atan2(Y, X) per chunk (radians, in (-pi, pi])."""
    return np.arctan2(maxval[..., 2], maxval[..., 1])


def circ_mean_deg(rad):
    """Circular mean in degrees [0,360).  (C.3 uses a plain mean(AoA), which breaks at +-180.)"""
    return float(np.degrees(np.angle(np.exp(1j * np.asarray(rad)).mean())) % 360)


def ang_diff_deg(a_rad, b_rad):
    """Wrapped difference a-b in degrees, (-180, 180]."""
    return (np.degrees(np.asarray(a_rad) - np.asarray(b_rad)) + 180) % 360 - 180


# ----------------------------------------------------------------------------- 3-D directions
# Azimuth alone is meaningless near the poles (X = Y = 0 -> atan2 is arbitrary) and 'azimuth error +
# elevation error' is not an angle on the sphere.  Work with unit vectors and great-circle angles.
def elevation_rad(maxval):
    """Elevation = atan2(Z, hypot(X, Y)) per chunk, radians in [-pi/2, pi/2] (needs a 3-D coefficient matrix)."""
    return np.arctan2(maxval[..., 3], np.hypot(maxval[..., 1], maxval[..., 2]))


def doa_vec(maxval):
    """Unit direction vector(s) (x, y, z) = (X, Y, Z) / |(X, Y, Z)| from signed B-Format peaks, shape (..., 3)."""
    v = np.asarray(maxval)[..., 1:4]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def unit_vec(az_deg, el_deg):
    """Unit vector for azimuth / elevation in degrees (broadcasts), shape (..., 3)."""
    az, el = np.deg2rad(np.asarray(az_deg, float)), np.deg2rad(np.asarray(el_deg, float))
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def az_el_deg(v):
    """(azimuth in [0, 360), elevation in [-90, 90]) in degrees from unit vector(s)."""
    v = np.asarray(v)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360, np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))


def great_circle_deg(v1, v2):
    """Angle between unit vectors in degrees, 0..180 (atan2 form: accurate for small and large angles)."""
    v1, v2 = np.asarray(v1), np.asarray(v2)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1), np.sum(v1 * v2, axis=-1)))


def mean_direction(v):
    """Mean direction of unit vectors (n,3) -> (unit vector, resultant length R in [0,1]).
    The 3-D analogue of the circular mean; R near 1 = chunks agree, R near 0 = scattered."""
    m = np.asarray(v).mean(axis=0)
    r = float(np.linalg.norm(m))
    return (m / r if r > 0 else m), r


# ----------------------------------------------------------------------------- golden files for C
def export_golden(outdir, coeff, x_chunk, p):
    """Write everything the C unit test needs (all little-endian float32 raw, C order).
       iq.f32       (5, chunk_len, 2)  I/Q interleaved  == numpy complex64 (5, chunk_len)
       coeff.f32    (5, 4)
       audio.f32    (out_len, 4)       expected output frames [W X Y Z]
       info.f32     rf_max[4], son_max[4]            (8 floats)
       kept.i32     kept[4]            (W, X, Y, Z; Z entry is 0 when n_pca = 3)
       params.json  parameters (incl. n_pca)"""
    os.makedirs(outdir, exist_ok=True)
    r = sonify_chunk(x_chunk, coeff, p)
    np.ascontiguousarray(x_chunk, dtype=np.complex64).tofile(os.path.join(outdir, "iq.f32"))
    np.ascontiguousarray(coeff, dtype=np.float32).tofile(os.path.join(outdir, "coeff.f32"))
    np.ascontiguousarray(r.audio, dtype=np.float32).tofile(os.path.join(outdir, "audio.f32"))
    np.concatenate([r.rf_max, r.son_max]).astype(np.float32).tofile(os.path.join(outdir, "info.f32"))
    r.kept.astype(np.int32).tofile(os.path.join(outdir, "kept.i32"))
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(dict(fsU=p.fsU, fsA=p.fsA, Tchunk=p.Tchunk, tuner=p.tuner, energy_pct=p.energy_pct,
                       sign_mode=p.sign_mode, n_pca=p.n_pca, M=p.M, N=p.N, chunk_len=p.chunk_len, out_len=p.out_len), f, indent=1)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the derived sizes for the default C.3 parameters")
    ap.add_argument("--tuner", type=float, default=-4000.0)
    a = ap.parse_args()
    p = SonifyParams(tuner=a.tuner).validate()
    print(f"M={p.M} records, N={p.N} bins, chunk_len={p.chunk_len}, out_len={p.out_len}, Noffset={p.noffset}")

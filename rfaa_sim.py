#!/usr/bin/env python3
"""
rfaa_sim.py
Python reproduction of Todd's 2025 dissertation, Appendix C.1 (optimizer) and C.2 (beamformer),
run on SIMULATED antenna data instead of the real RFAA recordings.

Pipeline (same order as the thesis):
  1. Simulate 5 antenna power patterns on the horizontal midplane (10 deg steps)
  2. Per angle: synthesize complex-baseband CW recordings, 1 s records, averaged PSD,
     max PSD within +-100 Hz  -> Gphi (36x5)               [C.1 "Processing Data"]
  3. OMP fits the W, X, Y targets from the averaged patterns -> coeffMat (5x4, Z = 0)
  4. A NEW measurement run (like the 2025-07-06 beamformer dataset):
     PSD @ coeffMat -> signed value at max |.| per B-Format channel -> AoA = atan2(Y, X)  [C.2]
  5. Linear fit of AoA vs transmit angle: slope / intercept / R^2 / standard error

Usage:
  python rfaa_sim.py                # horizontal plane only (Todd's system, Z = 0): one run + figure
  python rfaa_sim.py --mc 30        # 30 different random "physical arrays", summary statistics
  python rfaa_sim.py --sphere       # 3-D: calibrate W,X,Y,Z on the whole sphere, test on random directions
"""
import argparse
import numpy as np

DEGS = np.arange(0, 360, 10)  # transmit angles used in the thesis
# Table 4.1. The table prints -35.7 deg elevation for antennas 1 and 3, but their zenith
# angle (125.3) implies -35.3 deg (tetrahedron face normals), so -35.3 is used here.
ANT_AZ = np.deg2rad([135, 0, -45, -135, 45])
ANT_EL = np.deg2rad([-35.3, 90, -35.3, 35.3, 35.3])


def unit(az, el):
    """Unit vector(s) for azimuth/elevation in radians (broadcasts)."""
    az, el = np.broadcast_arrays(np.asarray(az, float), np.asarray(el, float))
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def ripple(az_deg, rng, amp, n_spikes, spike_amp, n_harm=6, spike_w=6.0):
    """Random smooth ripple + narrow spikes as a multiplicative power gain (cf. Fig 4.10)."""
    th = np.deg2rad(az_deg)
    g = np.ones_like(th)
    for k in range(1, n_harm + 1):
        g += (amp / k) * rng.normal() * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    for _ in range(n_spikes):
        d = (az_deg - rng.uniform(0, 360) + 180) % 360 - 180
        g += spike_amp * rng.uniform(0.3, 1.0) * np.exp(-0.5 * (d / spike_w) ** 2)
    return np.clip(g, 0.05, None)


class SimArray:
    """One 'physical' RFAA. Its imperfections stay fixed across all measurement runs."""

    def __init__(self, seed, p=2.0, az_err_deg=8.0, gain_sigma=0.3,
                 ripple_amp=0.30, n_spikes=3, spike_amp=0.5, el_ripple_amp=0.0):
        r = np.random.default_rng(seed)
        self.p = p  # cardioid sharpness: ((1+cos)/2)**p ; p>1 is narrower than a cardioid
        self.axes = unit(ANT_AZ + np.deg2rad(r.normal(0, az_err_deg, 5)), ANT_EL)  # mounting error
        self.gain = np.exp(r.normal(0, gain_sigma, 5))                              # channel gain error
        self.grid = np.arange(0.0, 361.0)
        sysg = np.stack([ripple(self.grid[:-1], r, ripple_amp, n_spikes, spike_amp)
                         for _ in range(5)], axis=1)
        self.sysg = np.vstack([sysg, sysg[:1]])  # periodic (361,5): repeatable distortion
        # Elevation-dependent systematic distortion, drawn AFTER everything above so that arrays with
        # el_ripple_amp=0 (the default) are bit-identical to the horizontal-only simulator.
        # Model assumption: a smooth random function of elevation multiplies the azimuth distortion.
        self.el_grid = np.arange(-90.0, 91.0)
        self.sysg_el = np.ones((self.el_grid.size, 5))
        if el_ripple_amp > 0:
            elr = np.deg2rad(self.el_grid)
            for i in range(5):
                g = np.ones_like(elr)
                for k in range(1, 4):
                    g += (el_ripple_amp / k) * r.normal() * np.cos(k * elr + r.uniform(0, 2 * np.pi))
                self.sysg_el[:, i] = np.clip(g, 0.05, None)

    def true_power(self, az_deg, rng, run_amp=0.08, run_spikes=1, run_spike_amp=0.15, el_deg=0.0):
        """Received power of each antenna for a source at az_deg / el_deg (arrays of equal length or
        el_deg scalar) -> (n,5).  Includes a small run-to-run random part (each real recording differed
        slightly); that part is a function of azimuth only and is drawn once per call."""
        az_deg = np.asarray(az_deg, float)
        el_deg = np.broadcast_to(np.asarray(el_deg, float), az_deg.shape)
        u = unit(np.deg2rad(az_deg), np.deg2rad(el_deg))      # (n,3) direction to source
        cosg = u @ self.axes.T                                  # (n,5)
        ideal = ((1 + cosg) / 2) ** self.p
        sysg = np.stack([np.interp(az_deg % 360, self.grid, self.sysg[:, i]) for i in range(5)], axis=1)
        sysel = np.stack([np.interp(el_deg, self.el_grid, self.sysg_el[:, i]) for i in range(5)], axis=1)
        run = np.stack([ripple(az_deg, rng, run_amp, run_spikes, run_spike_amp) for _ in range(5)], axis=1)
        return self.gain * ideal * sysg * sysel * run


def freqs(cfg):
    N = int(cfg["fs"] * cfg["Trec"])
    return np.fft.fftshift(np.fft.fftfreq(N, 1 / cfg["fs"]))


def synth_timeseries(power5, rng, fs, T, Trec, snr_db, f0=None, signal="cw", bw=8000.0, am_hz=10.0, am_depth=0.5,
                     ref_bins=1500):
    """Complex-baseband source + white noise on 5 antennas -> (5, int(fs*T)) complex128.
    power5 : received power per antenna (from SimArray.true_power)
    Trec   : reference record length that DEFINES the SNR (noise PSD stays fixed if fs changes)
    snr_db : peak-bin / noise-floor-bin ratio for an antenna with power 1 and a Trec-long record
    f0     : tone offset in Hz (None -> random integer in [-20, 20] like the thesis' small offset).
    signal : 'cw'    single tone (Todd's chapter-4 experiment; the default, RNG draw order identical to the old code)
             'noise' complex Gaussian noise, ideal band-pass of width `bw` Hz centred on f0 (a wideband transmitter)
             'am'    carrier amplitude-modulated by a cosine of `am_hz` Hz, depth `am_depth` (Todd appendix B used 1 Hz and 10 Hz)
    For 'noise' the in-band signal-to-noise ratio per FFT bin is set equal to the per-bin ratio a CW gets in one
    sonification record of `ref_bins` points, so snr_db means the same thing for all signal types.
    All antennas see the SAME waveform (only the power differs), as for a single point source.
    The RNG draw order (f0, phases, noise I, noise Q) is kept identical to the old psd_record."""
    Ntot = int(fs * T)
    t = np.arange(Ntot) / fs
    if f0 is None:
        f0 = rng.integers(-20, 21)                   # Hz (integer so the tone sits on one bin)
    ph = rng.uniform(0, 2 * np.pi, (5, 1))           # unknown phase per channel (power-only: irrelevant)
    if signal == "cw":
        sig = np.sqrt(power5)[:, None] * np.exp(1j * (2 * np.pi * f0 * t + ph))
    elif signal == "am":
        env = 1.0 + am_depth * np.cos(2 * np.pi * am_hz * t)
        sig = np.sqrt(power5)[:, None] * env * np.exp(1j * (2 * np.pi * f0 * t + ph))
    elif signal == "noise":
        if not 0 < bw <= fs:
            raise ValueError("bw must be in (0, fs]")
        w = (rng.standard_normal(Ntot) + 1j * rng.standard_normal(Ntot)) / np.sqrt(2)     # white, unit variance
        f = np.fft.fftfreq(Ntot, 1 / fs)
        W = np.fft.fft(w) * (np.abs(f - f0) <= bw / 2)                                     # ideal band-pass
        w = np.fft.ifft(W)                                                                  # in-band PSD = white, var 1
        sig = np.sqrt(power5 * ref_bins)[:, None] * np.exp(1j * ph) * w[None, :]
    else:
        raise ValueError("signal must be 'cw', 'noise' or 'am'")
    sigma2 = fs * Trec / 10 ** (snr_db / 10)
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal((5, Ntot)) + 1j * rng.standard_normal((5, Ntot)))
    return sig + noise


def psd_record(power5, rng, fs, T, Trec, snr_db):
    """Synthesize one 30 s recording of a CW (complex baseband) on 5 antennas, then do exactly
    the thesis processing: 1 s records, FFT/fs, |.|^2/Trec, average over records. -> (N,5)
    snr_db = peak-bin / noise-floor-bin ratio for an antenna with power 1."""
    N, Nrec = int(fs * Trec), int(T / Trec)
    x = synth_timeseries(power5, rng, fs, T, Trec, snr_db).reshape(5, Nrec, N)
    S = (np.abs(np.fft.fft(x, axis=2) / fs) ** 2 / Trec).mean(axis=1)   # (5,N)
    return np.fft.fftshift(S, axes=1).T


def measure_gphi(array, rng, cfg):
    """C.1 inner loop: max PSD inside +-100 Hz for every angle and antenna -> (36,5)."""
    P = array.true_power(DEGS, rng)
    band = (freqs(cfg) >= -100) & (freqs(cfg) <= 100)
    G = np.empty((len(DEGS), 5))
    for i in range(len(DEGS)):
        G[i] = psd_record(P[i], rng, **cfg)[band].max(axis=0)
    return G


def omp(Y, D, select, tol=1e-4):
    """Orthogonal matching pursuit as in the thesis' omp(): atoms are NOT normalised,
    stop at `select` atoms or when residual energy < tol * signal energy."""
    nS, nA = Y.shape[1], D.shape[1]
    X = np.zeros((nA, nS))
    for s in range(nS):
        y = Y[:, s]
        r, idx = y.copy(), []
        while len(idx) < select and r @ r > (y @ y) * tol:
            k = int(np.argmax(np.abs(D.T @ r)))
            if k in idx:
                break
            idx.append(k)
            coef, *_ = np.linalg.lstsq(D[:, idx], y, rcond=None)
            r = y - D[:, idx] @ coef
        X[idx, s] = coef
    return X


def optimize(array, rng, cfg, n_datasets=3):
    """C.1: average the patterns of 3 datasets, fit W,X,Y with OMP, Z column = 0."""
    avg = np.mean([measure_gphi(array, rng, cfg) for _ in range(n_datasets)], axis=0)
    th = np.deg2rad(DEGS)
    target = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])   # W, X, Y
    coeff = np.zeros((5, 4))
    coeff[:, :3] = omp(target, avg, select=5)
    return coeff, avg, target


# --------------------------------------------------------------------------- 3-D (full sphere)
def sphere_grid(az_step=10.0, el_step=15.0):
    """Calibration directions over the whole sphere (the thesis only measured the horizontal plane).
    Rings of constant elevation (every az_step) plus one point at each pole.
    Returns pts (n,2) = (az, el) in degrees and area (n,) = solid angle represented by each point [sr]."""
    pts, area = [], []
    for e in np.arange(-90.0 + el_step, 90.0 - 1e-9, el_step):
        for a in np.arange(0.0, 360.0, az_step):
            pts.append((a, e))
            area.append(np.deg2rad(az_step) * np.deg2rad(el_step) * np.cos(np.deg2rad(e)))
    cap = 2 * np.pi * (1 - np.cos(np.deg2rad(el_step / 2)))
    for e in (-90.0, 90.0):
        pts.append((0.0, e))
        area.append(cap)
    return np.array(pts), np.array(area)


def bformat_targets(az_deg, el_deg):
    """Ideal first-order B-Format [W, X, Y, Z] for a plane wave from (az, el): 1, cos el cos az, cos el sin az, sin el."""
    az, el = np.deg2rad(np.asarray(az_deg, float)), np.deg2rad(np.asarray(el_deg, float))
    return np.stack([np.ones_like(az), np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def measure_gphi_sphere(array, rng, cfg, pts):
    """3-D version of C.1's inner loop: max PSD inside +-100 Hz for every direction and antenna -> (n,5).
    One true_power call per elevation ring, so the run-to-run random part differs between rings."""
    band = (freqs(cfg) >= -100) & (freqs(cfg) <= 100)
    G = np.empty((len(pts), 5))
    for e in np.unique(pts[:, 1]):
        idx = np.nonzero(pts[:, 1] == e)[0]
        P = array.true_power(pts[idx, 0], rng, el_deg=e)
        for row, i in enumerate(idx):
            G[i] = psd_record(P[row], rng, **cfg)[band].max(axis=0)
    return G


def optimize_sphere(array, rng, cfg, n_datasets=3, az_step=10.0, el_step=15.0):
    """3-D C.1: fit all four B-Format channels W, X, Y, Z (Z column no longer 0) on the whole sphere.
    Solid-angle weighted least squares (with 5 atoms selected, Todd's OMP is plain least squares, see
    the OMP == lstsq check in rfaa_sim.py).  Returns coeff (5,4), avg patterns (n,5), targets (n,4),
    pts (n,2), area (n,).  Use a small cfg (e.g. fs=2e3, T=10) - the sphere has ~400 directions."""
    pts, area = sphere_grid(az_step, el_step)
    avg = np.mean([measure_gphi_sphere(array, rng, cfg, pts) for _ in range(n_datasets)], axis=0)
    target = bformat_targets(pts[:, 0], pts[:, 1])
    w = np.sqrt(area)[:, None]
    coeff = np.linalg.lstsq(w * avg, w * target, rcond=None)[0]
    return coeff, avg, target, pts, area


def beamform(array, coeff, rng, cfg):
    """C.2: a fresh measurement run -> signed value at max |.| over ALL bins -> atan2(Y, X)."""
    P = array.true_power(DEGS, rng)
    maxval = np.empty((len(DEGS), 4))
    for i in range(len(DEGS)):
        bp = psd_record(P[i], rng, **cfg) @ coeff        # (N,4) can be negative
        idx = np.argmax(np.abs(bp), axis=0)
        maxval[i] = bp[idx, np.arange(4)]
    aoa = np.degrees(np.arctan2(maxval[:, 2], maxval[:, 1])) % 360
    return aoa, maxval


def fit_stats(deg, est):
    """Same quantities the thesis reports: slope, intercept, R^2, standard error (deg)."""
    p = np.polyfit(deg, est, 1)
    res = est - np.polyval(p, deg)
    n = len(deg)
    se = np.sqrt((res ** 2).sum() / (n - 2))                      # = MATLAB S.normr/sqrt(S.df)
    r2 = 1 - (res ** 2).sum() / ((n - 1) * np.var(est, ddof=1))
    return p[0], p[1], r2, se


def unwrap_to_truth(deg, est):
    """Re-express each estimate as truth + wrapped error in (-180, 180]."""
    return deg + ((est - deg + 180) % 360 - 180)


def one_run(seed, cfg, **array_kw):
    arr = SimArray(seed, **array_kw)
    rng = np.random.default_rng(1000 + seed)
    coeff, avg, target = optimize(arr, rng, cfg)
    aoa, _ = beamform(arr, coeff, rng, cfg)
    # stats      : exactly the thesis recipe (polyfit on mod(AoA,360)) - breaks if estimates wrap at 0/360
    # stats_wrap : same fit after unwrapping errors around the truth - the meaningful number
    return dict(stats=fit_stats(DEGS, aoa), stats_wrap=fit_stats(DEGS, unwrap_to_truth(DEGS, aoa)),
                aoa=aoa, coeff=coeff, avg=avg, target=target)


# --------------------------------------------------------------------------- held-out 3-D evaluation
def random_directions(n, rng):
    """n directions uniformly distributed on the sphere -> (n,2) = (az, el) in degrees."""
    return np.column_stack([rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))])


def great_circle_deg(v1, v2):
    """Angle between unit vectors (rows) in degrees."""
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1), np.sum(v1 * v2, axis=-1)))


def evaluate_sphere(array, coeff, rng, cfg, n=300):
    """Array-side 3-D accuracy on directions NOT used for calibration: fresh simulated measurements of n
    random directions, B = max-PSD-per-antenna @ coeff, direction = (X,Y,Z)/|(X,Y,Z)|.
    Returns pts (n,2) and the great-circle error in degrees (n,)."""
    pts = random_directions(n, rng)
    G = measure_gphi_sphere(array, rng, cfg, pts)
    v = (G @ coeff)[:, 1:4]
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return pts, great_circle_deg(v, bformat_targets(pts[:, 0], pts[:, 1])[:, 1:4])


# --------------------------------------------------------------------------- plotting
# Colours: categorical slots 1-4 of the dataviz reference palette for W, X, Y, Z (validated on the light
# surface: adjacent CVD / lightness checks pass; aqua and yellow are below 3:1 contrast there, so every line
# chart carries a legend).  Antennas use other slots so no antenna shares a colour with a B-Format channel.
CH_COLOR = {"W": "#2a78d6", "X": "#eb6834", "Y": "#1baf7a", "Z": "#eda100"}
ANT_COLOR = ["#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9})
    return plt


def _channel_lines(ax, x, fit, ideal):
    """Fitted W,X,Y,Z (solid, channel colours) against the ideal B-Format (dashed grey)."""
    for j, name in enumerate("WXYZ"):
        ax.plot(x, ideal[:, j], "--", color=MUTED, lw=0.9)
        ax.plot(x, fit[:, j], color=CH_COLOR[name], lw=2, label=f"{name} fitted")
    ax.plot([], [], "--", color=MUTED, lw=0.9, label="ideal")
    ax.legend(loc="best")


def plot(res, path):
    """Horizontal-plane figure.  Z is drawn too: it is 0 by construction here (target and coefficient column)."""
    plt = _plt()
    th = np.deg2rad(DEGS)
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_subplot(1, 3, 1, projection="polar")
    for i in range(5):
        ax.plot(np.r_[th, th[0]], np.r_[res["avg"][:, i], res["avg"][0, i]], color=ANT_COLOR[i], lw=1.6,
                label=f"ant {i + 1}")
    ax.set_title("Simulated antenna patterns (max PSD)", loc="left")
    ax.legend(loc="lower left", fontsize=7)
    ax = fig.add_subplot(1, 3, 2)
    opt = res["avg"] @ res["coeff"]                                           # (36,4) incl. Z (column 0)
    ideal = np.column_stack([res["target"], np.zeros(len(DEGS))])             # W, X, Y, Z = 0
    _channel_lines(ax, DEGS, opt, ideal)
    ax.set_xlabel("angle [deg]")
    ax.set_title("OMP fit vs ideal B-Format (dashed)\nhorizontal plane only: Z = 0 by construction")
    ax = fig.add_subplot(1, 3, 3)
    s, b, r2, se = res["stats_wrap"]
    ax.plot(DEGS, res["aoa"], "o", color=INK2, ms=5, label="estimated AoA")
    ax.plot(DEGS, DEGS, "--", color=MUTED, lw=1, label="ideal")
    ax.plot(DEGS, s * DEGS + b, color=INK, lw=1.6, label=f"wrap-aware fit: slope {s:.2f}, int {b:.1f}, SE {se:.1f} deg")
    ax.set_xlabel("transmit angle [deg]"); ax.set_ylabel("estimated [deg]"); ax.legend(fontsize=8)
    ax.set_title("Beamformer dataset")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_sphere(res, path):
    """3-D figure: W,X,Y,Z fitted on the whole sphere.  res: pts, avg, coeff, target, err_pts, err_deg, note."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#3987e5", "#0d366b"])      # magnitude
    div = LinearSegmentedColormap.from_list("div_bluered", ["#1c5cab", "#5598e7", "#f0efec", "#ef8c8b", "#c23332"])
    pts, target = res["pts"], res["target"]
    fit = res["avg"] @ res["coeff"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    ax = axs[0, 0]                                                    # horizontal cut
    m = np.nonzero(pts[:, 1] == 0)[0]; m = m[np.argsort(pts[m, 0])]
    _channel_lines(ax, pts[m, 0], fit[m], target[m])
    ax.set_xlim(0, 350); ax.set_xlabel("azimuth [deg]")
    ax.set_title("Horizontal cut (elevation 0°)\nZ should stay at 0")

    ax = axs[0, 1]                                                    # meridian cut, pole to pole
    m = np.nonzero(pts[:, 0] == 0)[0]; m = m[np.argsort(pts[m, 1])]
    _channel_lines(ax, pts[m, 1], fit[m], target[m])
    ax.set_xlim(-90, 90); ax.set_xlabel("elevation [deg]")
    ax.set_title("Vertical cut (azimuth 0°, pole to pole)\nZ should follow sin(elevation)")

    ax = axs[1, 0]                                                    # Z fit error map
    zerr = fit[:, 3] - target[:, 3]
    lim = float(np.ceil(np.abs(zerr).max() * 10) / 10)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=zerr, cmap=div, vmin=-lim, vmax=lim, s=60,
                    edgecolor=SURFACE, linewidth=0.8)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("fitted Z − ideal Z"); cb.outline.set_edgecolor(AXIS)
    ax.set_xlim(-5, 355); ax.set_ylim(-95, 95); ax.set_xlabel("azimuth [deg]"); ax.set_ylabel("elevation [deg]")
    ax.set_title("Z fit error on the calibration directions")

    ax = axs[1, 1]                                                    # held-out direction error
    e = res["err_deg"]
    sc = ax.scatter(res["err_pts"][:, 0], res["err_pts"][:, 1], c=e, cmap=seq, vmin=0, vmax=np.percentile(e, 98),
                    s=40, edgecolor=SURFACE, linewidth=0.6)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("great-circle error [deg]"); cb.outline.set_edgecolor(AXIS)
    ax.set_xlim(-5, 365); ax.set_ylim(-95, 95); ax.set_xlabel("azimuth [deg]"); ax.set_ylabel("elevation [deg]")
    ax.set_title(f"Direction error, {len(e)} random held-out directions\n"
                 f"median {np.median(e):.1f}°, 90 % below {np.percentile(e, 90):.1f}°, max {e.max():.1f}°")
    fig.text(0.01, 0.006, res["note"], color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mc", type=int, default=0, help="number of random arrays for Monte Carlo")
    ap.add_argument("--fs", type=float, default=12e3, help="sample rate; thesis used 120e3 (only +-100 Hz matters)")
    ap.add_argument("--snr", type=float, default=40.0)
    ap.add_argument("--distort", type=float, default=1.0,
                    help="scales mounting error, gain error, pattern ripple and spikes (1 = default)")
    ap.add_argument("--sphere", action="store_true",
                    help="3-D mode: fit W,X,Y,Z on the whole sphere (Z column != 0) and test on random directions")
    ap.add_argument("--out", default=None, help="figure file (default rfaa_sim_result.png / rfaa_sim_sphere.png)")
    a = ap.parse_args()
    cfg = dict(fs=a.fs, T=30, Trec=1, snr_db=a.snr)
    kw = dict(az_err_deg=8.0 * a.distort, gain_sigma=0.3 * a.distort,
              ripple_amp=0.30 * a.distort, spike_amp=0.5 * a.distort)

    if a.sphere:
        out = a.out or "rfaa_sim_sphere.png"
        arr = SimArray(a.seed, el_ripple_amp=0.15 * a.distort, **kw)       # elevation-dependent distortion on
        cal = dict(fs=2e3, T=10, Trec=1, snr_db=a.snr)                      # only +-100 Hz matters -> low fs is fast
        rng = np.random.default_rng(1000 + a.seed)
        coeff, avg, target, pts, area = optimize_sphere(arr, rng, cal)
        rms = np.sqrt((area[:, None] * (avg @ coeff - target) ** 2).sum(axis=0) / area.sum())
        print(f"calibration: {len(pts)} directions x 3 datasets; weighted RMS fit error  W {rms[0]:.3f}  X {rms[1]:.3f}"
              f"  Y {rms[2]:.3f}  Z {rms[3]:.3f}")
        print("coefficient matrix (rows = antennas, columns = W X Y Z):")
        print(np.array2string(coeff, precision=2, suppress_small=True))
        ept, edeg = evaluate_sphere(arr, coeff, rng, cal, n=300)
        near = np.abs(ept[:, 1]) < 20
        print(f"held-out directions (300 random): median {np.median(edeg):.1f} deg, 90% < {np.percentile(edeg, 90):.1f} deg,"
              f" max {edeg.max():.1f} deg;  within 20 deg of the horizontal plane: median {np.median(edeg[near]):.1f} deg")
        note = (f"Simulation, seed {a.seed}, distort {a.distort}, SNR {a.snr:g} dB. Z fitted on the whole sphere "
                f"(solid-angle weighted least squares); errors depend on the assumed distortion model.")
        plot_sphere(dict(pts=pts, avg=avg, coeff=coeff, target=target, err_pts=ept, err_deg=edeg, note=note), out)
        print("saved", out)
    elif a.mc:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        with ProcessPoolExecutor() as ex:
            runs = list(ex.map(partial(one_run, cfg=cfg, **kw), range(a.mc)))
        print(f"{a.mc} random arrays, distort={a.distort}   thesis: slope 0.90  intercept 23.8  R2 0.98  SE 12.6")
        for key, title in [("stats", "thesis recipe (mod 360, wraps hurt)"), ("stats_wrap", "wrap-aware")]:
            S = np.array([r[key] for r in runs])
            print(f"[{title}]")
            for k, name in enumerate(["slope", "intercept", "R^2", "std.err"]):
                q = np.percentile(S[:, k], [10, 50, 90])
                print(f"  {name:9s} 10%/50%/90% = {q[0]:8.2f} {q[1]:8.2f} {q[2]:8.2f}")
    else:
        res = one_run(a.seed, cfg, **kw)
        for key, title in [("stats", "thesis recipe (mod 360)"), ("stats_wrap", "wrap-aware     ")]:
            s, b, r2, se = res[key]
            print(f"{title}: slope {s:.3f}  intercept {b:.1f} deg  R^2 {r2:.3f}  std.err {se:.1f} deg")
        # sanity: with all 5 atoms selected, OMP must equal plain least squares
        ls = np.linalg.lstsq(res["avg"], res["target"], rcond=None)[0]
        print("OMP == least squares:", np.allclose(res["coeff"][:, :3], ls, atol=1e-6))
        out = a.out or "rfaa_sim_result.png"
        plot(res, out)
        print("saved", out)

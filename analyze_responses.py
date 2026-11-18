#!/usr/bin/env python3
"""
analyze_responses.py -- statistics + figure for the listening experiments made with listening_trials.py.

  python3 analyze_responses.py s_cw s_n8 s_n40 s_n100 s_n100r s_am10 --file responses_head.csv \
        --labels "CW,noise 8 kHz,noise 40 kHz,noise 100 kHz,noise 100 kHz (random phase),AM 10 Hz" --out report

Each folder needs answers.csv (written by listening_trials.py) and the response file (trial,az,el; listener convention:
az 0 = front, 90 = right, 180 = behind; el + up).  Writes report/results.csv and report/figure.png and prints the tables.

What is computed (n is small, one listener: everything is descriptive / exploratory):
  * great-circle error of each answer; permutation test = the answers are re-paired with random trials (20000 times), so the
    test knows the listener's own answer habits and asks only "is the pairing with the truth better than random?"
  * elevation: Spearman rho (true vs answered elevation), permutation p (one-sided, rho > 0), mean bias, mean absolute error
  * lateral (left/right): trials with |true lateral angle| >= 20 deg, correct side, exact binomial test vs 0.5 + Clopper-Pearson CI
  * polar (front/back/up/down inside the cone around the ear axis): quadrant error = |polar error| > 90 deg, trials with |true lateral| <= 60
  * pooled over groups: the SAME permutation of the true directions is applied to all groups (keeps the dependence between groups,
    because the same 10 directions were used in every group)
  * Friedman test across groups on the error (paired by trial)
"""
import argparse, csv, os
import numpy as np
from scipy import stats


def unit(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)         # x front, y right, z up


def gc(az1, el1, az2, el2):
    u, v = unit(az1, el1), unit(az2, el2)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), (u * v).sum(-1)))


def lateral(az, el):
    return np.degrees(np.arcsin(np.clip(unit(az, el)[..., 1], -1, 1)))


def polar(az, el):
    u = unit(az, el)
    return np.degrees(np.arctan2(u[..., 2], u[..., 0]))


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def cp_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def load(folder, fname):
    ans = {int(r["trial"]): r for r in csv.DictReader(open(f"{folder}/answers.csv"))}
    t, ra, re_, ta, te, ea, ee = [], [], [], [], [], [], []
    for r in csv.DictReader(open(f"{folder}/{fname}")):
        if r["az"].strip() == "" or r["el"].strip() == "":
            continue
        k = int(r["trial"]); a = ans[k]
        t.append(k); ra.append(float(r["az"])); re_.append(float(r["el"]))
        ta.append(float(a["true_az"])); te.append(float(a["true_el"]))
        ea.append(float(a["son_est_az"])); ee.append(float(a["son_est_el"]))
    return dict(trial=np.array(t), r_az=np.array(ra), r_el=np.array(re_), t_az=np.array(ta), t_el=np.array(te),
                s_az=np.array(ea), s_el=np.array(ee))


def metrics(d, tr_az, tr_el):
    """metrics of responses d against the truth (tr_az, tr_el) -- the truth is permuted for the null distribution."""
    err = gc(d["r_az"], d["r_el"], tr_az, tr_el)
    out = dict(n=len(err), med=np.median(err), mean=err.mean())
    out["rho_el"] = stats.spearmanr(tr_el, d["r_el"])[0] if len(err) > 2 else np.nan
    out["mae_el"] = np.mean(np.abs(d["r_el"] - tr_el))
    out["bias_el"] = np.mean(d["r_el"] - tr_el)
    tl, rl = lateral(tr_az, tr_el), lateral(d["r_az"], d["r_el"])
    m = np.abs(tl) >= 20.0
    out["lat_n"], out["lat_k"] = int(m.sum()), int((np.sign(tl[m]) == np.sign(rl[m])).sum())
    out["rho_lat"] = stats.spearmanr(tl, rl)[0] if len(err) > 2 else np.nan
    tp, rp = polar(tr_az, tr_el), polar(d["r_az"], d["r_el"])
    mp = np.abs(tl) <= 60.0
    out["pol_n"], out["pol_k"] = int(mp.sum()), int((np.abs(wrap180(rp - tp))[mp] > 90.0).sum())
    return out, err


def run(data, n_perm, rng):
    names = list(data)
    obs = {g: metrics(data[g], data[g]["t_az"], data[g]["t_el"]) for g in names}
    # joint permutation of the trial pairing, same permutation in every group (all groups share the same trials)
    n = len(data[names[0]]["t_az"])
    perms = [rng.permutation(n) for _ in range(n_perm)]
    null = {g: {"med": [], "rho_el": [], "rho_lat": [], "lat_k": [], "pol_k": []} for g in names + ["pooled"]}
    for p in perms:
        acc = {"med": [], "rho_el": [], "rho_lat": [], "lat_k": [], "pol_k": []}
        for g in names:
            d = data[g]
            m, _ = metrics(d, d["t_az"][p], d["t_el"][p])
            for k in acc:
                null[g][k].append(m[k]); acc[k].append(m[k])
        null["pooled"]["med"].append(np.mean(acc["med"]))
        for k in ("rho_el", "rho_lat"):
            null["pooled"][k].append(np.nanmean(acc[k]))
        for k in ("lat_k", "pol_k"):
            null["pooled"][k].append(np.sum(acc[k]))
    for g in null:
        for k in null[g]:
            null[g][k] = np.array(null[g][k], float)
    return obs, null


def pval(null, obs, lower):
    """one-sided permutation p with the +1 correction; lower: small values are 'better'"""
    return (1 + (null <= obs).sum()) / (1 + len(null)) if lower else (1 + (null >= obs).sum()) / (1 + len(null))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--file", default="responses.csv", help="response file name inside each folder")
    ap.add_argument("--labels", help="comma separated display names (same order as folders)")
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="report")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    labels = a.labels.split(",") if a.labels else [os.path.basename(f.rstrip("/")) for f in a.folders]
    assert len(labels) == len(a.folders)
    data = {lab: load(f, a.file) for lab, f in zip(labels, a.folders)}
    n0 = {len(d["t_az"]) for d in data.values()}
    ref = next(iter(data.values()))
    same = all(np.allclose(d["t_az"], ref["t_az"]) and np.allclose(d["t_el"], ref["t_el"]) for d in data.values())
    if len(n0) != 1 or not same:
        raise SystemExit("all groups must use the same trials/true directions (same --random N and --seed) for the joint permutation")
    rng = np.random.default_rng(a.seed)
    obs, null = run(data, a.perm, rng)
    os.makedirs(a.out, exist_ok=True)

    hdr = (f"{'group':<24}{'n':>3} {'median err':>10} {'(chance*)':>9} {'p':>7} | {'elev rho':>8} {'p':>7} {'bias':>6} | "
           f"{'L/R correct':>12} {'95% CI':>13} | {'quadrant err':>12}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for g in list(data) + ["pooled"]:
        if g == "pooled":
            o = dict(n=sum(obs[x][0]["n"] for x in data), med=np.mean([obs[x][0]["med"] for x in data]),
                     rho_el=np.nanmean([obs[x][0]["rho_el"] for x in data]), bias_el=np.mean([obs[x][0]["bias_el"] for x in data]),
                     lat_n=sum(obs[x][0]["lat_n"] for x in data), lat_k=sum(obs[x][0]["lat_k"] for x in data),
                     pol_n=sum(obs[x][0]["pol_n"] for x in data), pol_k=sum(obs[x][0]["pol_k"] for x in data),
                     rho_lat=np.nanmean([obs[x][0]["rho_lat"] for x in data]))
        else:
            o = obs[g][0]
        nl = null[g]
        p_med = pval(nl["med"], o["med"], True); p_rho = pval(nl["rho_el"], o["rho_el"], False)
        lo, hi = cp_ci(o["lat_k"], o["lat_n"]); plr = stats.binomtest(o["lat_k"], o["lat_n"], 0.5).pvalue if o["lat_n"] else np.nan
        plo, phi = cp_ci(o["pol_k"], o["pol_n"])
        lab = g if g != "pooled" else "pooled (mean of groups)"
        print(f"{lab:<24}{o['n']:>3} {o['med']:>9.1f}° {np.mean(nl['med']):>8.1f}° {p_med:>7.3f} | {o['rho_el']:>+8.2f} {p_rho:>7.3f} {o['bias_el']:>+5.0f}° | "
              f"{o['lat_k']:>5}/{o['lat_n']:<3} p={plr:<5.3f} [{lo:.2f},{hi:.2f}] | {o['pol_k']:>3}/{o['pol_n']:<3} ({100 * o['pol_k'] / max(o['pol_n'], 1):.0f}%)")
        rows.append(dict(group=g, n=o["n"], median_err=o["med"], null_median_err=np.mean(nl["med"]), p_err=p_med, rho_el=o["rho_el"], p_rho_el=p_rho,
                         bias_el=o["bias_el"], lat_correct=o["lat_k"], lat_n=o["lat_n"], p_lat_binom=plr, lat_ci_lo=lo, lat_ci_hi=hi,
                         quad_err=o["pol_k"], quad_n=o["pol_n"], quad_ci_lo=plo, quad_ci_hi=phi, rho_lat=o["rho_lat"]))
    print("* chance = mean over permutations of the median error when answers are re-paired with random trials (keeps your answer habits)")
    # pooled counts p-values (sum of correct sides across groups, joint permutation)
    pl = pval(null["pooled"]["lat_k"], sum(obs[x][0]["lat_k"] for x in data), False)
    pq = pval(null["pooled"]["pol_k"], sum(obs[x][0]["pol_k"] for x in data), True)
    print(f"pooled permutation p: L/R correct sides {pl:.3f}, fewer quadrant errors than chance {pq:.3f}")
    errs = np.stack([gc(d["r_az"], d["r_el"], d["t_az"], d["t_el"]) for d in data.values()])
    if len(errs) >= 3:
        fr = stats.friedmanchisquare(*errs)
        print(f"Friedman test, error differs between groups (paired by trial): chi2={fr.statistic:.2f}, p={fr.pvalue:.3f}")
    else:
        print("Friedman test skipped (needs at least 3 groups)")
    # ceiling: what the AUDIO itself carries (son_est in answers.csv)
    print("\nCeiling -- direction carried by the audio (answers.csv, 'son_est') vs truth:")
    for g, d in data.items():
        e = gc(d["s_az"], d["s_el"], d["t_az"], d["t_el"])
        print(f"  {g:<24} median error {np.median(e):5.1f}°   elevation r(true, audio) = {np.corrcoef(d['t_el'], d['s_el'])[0, 1]:+.2f}")
    with open(f"{a.out}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if not a.no_figure:
        figure(data, obs, null, a.out)
    print(f"\nwrote {a.out}/results.csv" + ("" if a.no_figure else f" and {a.out}/figure.png"))


def figure(data, obs, null, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, accent, grid = "#1f2937", "#6b7280", "#1d4ed8", "#d1d5db"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.9), constrained_layout=True)
    names = list(data)
    # (a) error per group with the chance band
    for i, g in enumerate(names):
        e = gc(data[g]["r_az"], data[g]["r_el"], data[g]["t_az"], data[g]["t_el"])
        nl = null[g]["med"]
        ax[0].plot([i, i], np.percentile(nl, [2.5, 97.5]), color=grid, lw=7, solid_capstyle="butt", zorder=1)
        ax[0].scatter(np.full(len(e), i) + np.linspace(-.18, .18, len(e)), e, s=10, color=muted, alpha=.6, zorder=2)
        ax[0].scatter([i], [np.median(e)], s=38, color=accent, zorder=3)
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels([n.replace(" (", "\n(") for n in names], rotation=35, ha="right", fontsize=8)
    ax[0].set_ylabel("angular error (deg)"); ax[0].set_ylim(0, 180)
    ax[0].set_title("(a) error per answer; blue = median\ngray bar = 95% range expected by chance", fontsize=9, loc="left")
    # (b) elevation, all groups
    te = np.concatenate([d["t_el"] for d in data.values()]); re_ = np.concatenate([d["r_el"] for d in data.values()])
    ax[1].plot([-90, 90], [-90, 90], color=grid, lw=1)
    ax[1].scatter(te, re_, s=14, color=accent, alpha=.55)
    ax[1].set_xlabel("true elevation (deg)"); ax[1].set_ylabel("answered elevation (deg)"); ax[1].set_xlim(-60, 90); ax[1].set_ylim(-90, 90)
    ax[1].set_title("(b) elevation, all groups pooled\n(gray line = perfect)", fontsize=9, loc="left")
    # (c) lateral
    tl = np.concatenate([lateral(d["t_az"], d["t_el"]) for d in data.values()]); rl = np.concatenate([lateral(d["r_az"], d["r_el"]) for d in data.values()])
    ax[2].plot([-90, 90], [-90, 90], color=grid, lw=1)
    ax[2].scatter(tl, rl, s=14, color=accent, alpha=.55)
    ax[2].set_xlabel("true lateral angle (deg, + = right)"); ax[2].set_ylabel("answered lateral angle (deg)")
    ax[2].set_title("(c) left/right, all groups pooled", fontsize=9, loc="left")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False); x.grid(color=grid, lw=.4, alpha=.6)
    fig.savefig(f"{out}/figure.png", dpi=200)


if __name__ == "__main__":
    main()

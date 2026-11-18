#!/usr/bin/env python3
"""
make_session.py -- mix several listening_trials.py output folders into ONE blind, shuffled session, and analyse it.

1) Make the condition folders with the SAME --random N --seed (same directions in every folder), e.g.
     python3 listening_trials.py --random 10 --seed 41 --reference foa --stimulus noise                       --out A_full
     python3 listening_trials.py --random 10 --seed 41 --reference foa --stimulus noise  --lowpass 9000       --out B_lp9k
     python3 listening_trials.py --random 10 --seed 41 --reference foa --stimulus whitebw --band 500,9000     --out C_flat

2) Build the blind session (files s01.wav, s02.wav, ... in a random order; no condition names anywhere inside it):
     python3 make_session.py --build A_full B_lp9k C_flat --out session --seed 123
   -> session/s01.wav ...   session/responses.csv (fill it: id, az, el; az 0 = front, 90 = right, el + up)
      session_key_DO_NOT_OPEN.csv   (id -> condition, true direction)   do not open before you have answered everything

3) After ALL answers are filled in:
     python3 make_session.py --check session/responses.csv --key session_key_DO_NOT_OPEN.csv

The shuffle keeps the same direction at least 4 positions apart.  Take a break after every 10 files.
"""
import argparse
import csv
import itertools
import os
import shutil

import numpy as np
from scipy import stats


# ------------------------------------------------------------------------------------- geometry (listener convention)
def unit(az, el):
    az, el = np.radians(az), np.radians(el)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)      # x front, y right, z up


def gc(a1, e1, a2, e2):
    u, v = unit(a1, e1), unit(a2, e2)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), (u * v).sum(-1)))


def lateral(az, el):
    return np.degrees(np.arcsin(np.clip(unit(az, el)[..., 1], -1, 1)))


def polar(az, el):
    u = unit(az, el)
    return np.degrees(np.arctan2(u[..., 2], u[..., 0]))


def w180(x):
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------------------------- build
def build(folders, out, seed):
    conds = {}
    for f in folders:
        rows = list(csv.DictReader(open(f"{f}/answers.csv")))
        conds[os.path.basename(os.path.normpath(f))] = (f, rows)
    names = list(conds)
    n = len(conds[names[0]][1])
    for k in names:
        assert len(conds[k][1]) == n, f"{k}: different number of trials"
        for r0, r in zip(conds[names[0]][1], conds[k][1]):
            assert abs(float(r0["true_az"]) - float(r["true_az"])) < 0.2 and abs(float(r0["true_el"]) - float(r["true_el"])) < 0.2, \
                "the folders do not contain the same directions: use the same --random N and --seed"
    items = [(k, int(r["trial"])) for k in names for r in conds[k][1]]
    rng = np.random.default_rng(seed)
    for _ in range(100000):                                                  # shuffle with min gap 4 for equal directions
        p = rng.permutation(len(items))
        order = [items[i] for i in p]
        last, ok = {}, True
        for pos, (_, t) in enumerate(order):
            if t in last and pos - last[t] < 4:
                ok = False
                break
            last[t] = pos
        if ok:
            break
    else:
        raise SystemExit("could not find a shuffle with the required spacing")
    os.makedirs(out, exist_ok=True)
    key_path = f"{out.rstrip('/')}_key_DO_NOT_OPEN.csv"
    with open(key_path, "w", newline="") as kf, open(f"{out}/responses.csv", "w", newline="") as rf:
        kw, rw = csv.writer(kf), csv.writer(rf)
        kw.writerow(["id", "condition", "trial", "true_az", "true_el", "source_file"])
        rw.writerow(["id", "az", "el"])
        for i, (k, t) in enumerate(order, 1):
            r = next(r for r in conds[k][1] if int(r["trial"]) == t)
            src = f"{conds[k][0]}/trial_{t:02d}.wav"
            shutil.copyfile(src, f"{out}/s{i:02d}.wav")
            kw.writerow([i, k, t, r["true_az"], r["true_el"], src])
            rw.writerow([i, "", ""])
    print(f"{len(order)} files in {out}/ ({len(names)} conditions x {n} trials, shuffled); fill {out}/responses.csv")
    print(f"key: {key_path}  <- do not open before you have answered everything")


# ------------------------------------------------------------------------------------- analysis
def holm(p):
    p = np.asarray(p, float); idx = np.argsort(p); out = np.empty_like(p); run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (len(p) - r) * p[i]); out[i] = min(1.0, run)
    return out


def boot_ci(x, fn, n=5000, seed=1):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    vals = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def flip_flags(ra, re_, ta, te):
    out = []
    for a, e, t_a, t_e in zip(ra, re_, ta, te):
        near = min(abs(w180(t_a - 90)), abs(w180(t_a + 90))) < 30                  # mirror = itself near the ear axis
        out.append(None if near else bool(gc((180 - a) % 360, e, t_a, t_e) < gc(a, e, t_a, t_e)))
    return out


def check(resp_path, key_path, csv_out):
    key = {int(r["id"]): r for r in csv.DictReader(open(key_path))}
    data = {}
    for r in csv.DictReader(open(resp_path)):
        if r["az"].strip() == "" or r["el"].strip() == "":
            continue
        k = key[int(r["id"])]
        data.setdefault(k["condition"], []).append(
            [int(k["trial"]), float(r["az"]), float(r["el"]), float(k["true_az"]), float(k["true_el"])])
    conds = sorted(data)
    D = {c: np.array(sorted(data[c])) for c in conds}
    rng = np.random.default_rng(7)
    print(f"{sum(len(v) for v in D.values())} answers in {len(conds)} conditions\n")
    hdr = f"{'condition':12s} {'n':>2s} {'mean err':>8s} {'[95% CI]':>12s} {'median':>6s} {'p vs chance':>11s} {'L/R ok':>7s} {'elev rho [95% CI]':>22s} {'|d el|':>6s} {'el bias':>7s} {'quad%':>5s} {'F/B flips':>9s}"
    print(hdr); print("-" * len(hdr))
    err, eel = {}, {}
    rows_out = []
    for c in conds:
        a = D[c]; t, ra, re_, ta, te = a.T
        e = gc(ra, re_, ta, te); err[c] = e
        nul = np.array([gc(ra[p], re_[p], ta, te).mean() for p in (rng.permutation(len(ra)) for _ in range(10000))])
        pv = (nul <= e.mean()).mean()
        lr = int(np.sum(np.sign(lateral(ra, re_)) == np.sign(lateral(ta, te))))
        rho, _ = stats.spearmanr(re_, te)
        rci = boot_ci(np.arange(len(a)), lambda i: stats.spearmanr(re_[i], te[i])[0] if len(set(te[i])) > 1 and len(set(re_[i])) > 1 else np.nan)
        rci = np.nanpercentile([stats.spearmanr(re_[i], te[i])[0] for i in (rng.integers(0, len(a), len(a)) for _ in range(3000))
                                if len(set(te[i])) > 1 and len(set(re_[i])) > 1], [2.5, 97.5])
        ci = boot_ci(e, np.mean)
        de = np.abs(re_ - te); eel[c] = de
        quad = np.mean(np.abs(w180(polar(ra, re_) - polar(ta, te))) > 90) * 100
        fl = [f for f in flip_flags(ra, re_, ta, te) if f is not None]
        print(f"{c:12s} {len(a):2d} {e.mean():7.1f}° [{ci[0]:4.0f},{ci[1]:4.0f}] {np.median(e):5.1f}° {pv:11.3f} {lr:4d}/{len(a):<2d} {rho:+6.2f} [{rci[0]:+.2f},{rci[1]:+.2f}] {de.mean():5.1f}° {np.mean(re_ - te):+6.1f}° {quad:4.0f}% {sum(fl):3d}/{len(fl):<3d}")
        for tt, aa, ee_, ta_, te_, e_ in zip(t, ra, re_, ta, te, e):
            rows_out.append([c, int(tt), aa, ee_, ta_, te_, round(float(e_), 1), round(float(abs(ee_ - te_)), 1)])
    print("\n(chance level of the mean angular error is about 87 degrees; |d el| = mean absolute elevation error;")
    print(" el bias = mean answered minus true elevation; quad% = share of polar-angle errors above 90 degrees;")
    print(" F/B flips = answers closer to the front/back mirror image than to the truth, ear-axis directions not counted)\n")
    if len(conds) >= 2:
        tr = [set(D[c][:, 0]) for c in conds]
        if all(s == tr[0] for s in tr):
            print("Paired comparisons (same directions in every condition; Wilcoxon signed-rank, Holm over pairs):")
            for name, store in (("angular error", err), ("|elevation error|", eel)):
                pairs = list(itertools.combinations(conds, 2)); ps = []; diffs = []
                for x, y in pairs:
                    ix = np.argsort(D[x][:, 0]); iy = np.argsort(D[y][:, 0])
                    d = store[x][ix] - store[y][iy]; diffs.append(d.mean())
                    ps.append(stats.wilcoxon(store[x][ix], store[y][iy]).pvalue if np.any(d != 0) else 1.0)
                hp = holm(ps)
                print(f"  {name}:")
                for (x, y), d, p, h in zip(pairs, diffs, ps, hp):
                    print(f"    {x} - {y}: mean difference {d:+6.1f} deg, p = {p:.3f}, Holm {h:.3f}")
            if len(conds) >= 3:
                print(f"  Friedman over all conditions: angular error p = {stats.friedmanchisquare(*[err[c][np.argsort(D[c][:, 0])] for c in conds]).pvalue:.3f}, "
                      f"|elevation error| p = {stats.friedmanchisquare(*[eel[c][np.argsort(D[c][:, 0])] for c in conds]).pvalue:.3f}")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["condition", "trial", "resp_az", "resp_el", "true_az", "true_el", "angle_error", "abs_el_error"])
        w.writerows(rows_out)
    print(f"\nper-trial table written to {csv_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", nargs="+", metavar="FOLDER", help="listening_trials output folders (same directions) to mix")
    ap.add_argument("--out", default="session")
    ap.add_argument("--seed", type=int, default=123, help="shuffle seed")
    ap.add_argument("--check", metavar="RESPONSES.csv")
    ap.add_argument("--key", default="session_key_DO_NOT_OPEN.csv")
    ap.add_argument("--csv", default="session_results.csv")
    a = ap.parse_args()
    if a.build:
        build(a.build, a.out, a.seed)
    elif a.check:
        check(a.check, a.key, a.csv)
    else:
        ap.print_help()

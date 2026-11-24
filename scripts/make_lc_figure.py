"""Figure 4: one B5 catalog-truth-audit case shown on the light curve (TIC 219882262).

(a) early-window light curve (the data the searches and the resolver saw) folded at the catalog period, odd and even
    cycles coloured separately; (b) the frozen v2 resolver's alias probabilities and the six evidence terms of the
    catalog-period alias relative to the chosen alias, both read from the stored B5 run record; (c) later-sector light
    curve (not seen by the resolver) folded at the resolver period.
The pinned files are downloaded from MAST and compacted with the same functions as the benchmark caches
(scripts/build_cache.py: lightkurve default mask, PDCSAP, per-sector median normalisation) into data/cache/fig/
(gitignored). Every period and probability printed on the figure is read from the results files and asserted against
the released audit table (CLAUDE.md rule 5). Output: paper/figures/fig_lc_example.pdf and results/fig_lc_example.json.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua.resolver import model as M  # noqa: E402
from b1_run import read_records  # noqa: E402
from build_cache import compact, download  # noqa: E402
import resolver_models as RM  # noqa: E402

TIC = 219882262
LATER_SHOWN = 4  # first later sectors plotted in panel (c); the audit itself used all later sectors
CACHE = ROOT / "data" / "cache" / "fig"
ORDER = ["1/5", "1/4", "1/3", "1/2", "2/3", "1", "3/2", "2", "3", "4", "5"]
TERMS = [("d_F1_coverage", "F1"), ("d_F2_oddeven", "F2"), ("d_F3_secondary", "F3"), ("d_F4_density", "F4"),
         ("d_F5_coherence", "F5"), ("d_F6_shape", "F6")]


def load_sectors(pins):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for r in pins.itertuples():
        f = CACHE / r.filename
        npz = CACHE / (r.filename + ".npz")
        if not npz.exists():
            assert download(r.url, f), r.filename
            (t, fl, fe), meta = compact(str(f))
            np.savez(npz, t=t, f=fl, good_frac=meta["good_frac"])
            f.unlink(missing_ok=True)
        z = np.load(npz)
        if float(z["good_frac"]) >= 0.5:  # X11, as in the benchmark
            out.append((int(r.sector), z["t"], z["f"]))
    return out


def stitch(secs, T):
    t = np.concatenate([s[1] for s in secs])
    f = np.concatenate([s[2] for s in secs])
    ok = np.isfinite(t) & np.isfinite(f)
    o = np.argsort(t[ok])
    t, y = M.detrend(t[ok][o], f[ok][o], duration=T)
    return t, 1.0 - y


def binned(ph, fl, nb=200):
    e = np.linspace(-0.5, 0.5, nb + 1)
    i = np.digitize(ph, e) - 1
    c = 0.5 * (e[1:] + e[:-1])
    m = np.array([np.median(fl[i == k]) if np.any(i == k) else np.nan for k in range(nb)])
    return c, m


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = pd.read_csv(ROOT / "paper" / "data" / "eb_candidate_period_revisions.csv").query("sample == 'B5'").set_index("tic").loc[TIC]
    tgt = pd.read_csv(ROOT / "data" / "b5_audit_targets.csv").set_index("tic").loc[TIC]
    rec = [r for p in glob.glob(str(ROOT / "results/kaggle/b5r/*/b5/b5_tls_*.jsonl.gz")) for r in read_records(p) if r.get("tic") == TIC]
    assert len(rec) == 1
    rec = rec[0]
    P_cat, P_res = float(tgt["P_true"]), float(tgt["P_v2_tls"])
    assert abs(P_res - float(rel["P_v2"])) / P_res < 1e-6 and abs(P_cat - float(rel["P_catalog"])) / P_cat < 1e-6
    assert rel["status"].startswith("candidate revision")
    T = float(rec["summary"]["T_seed"])
    t0 = float(rec["summary"]["t0_seed"])

    model, thr = RM.load("v2")
    rows = {row["r"]: row for row in rec["rows"]}
    group = [(row["r"], row["P"], row["x"]) for row in rec["rows"]]
    p_v2 = dict(zip([g[0] for g in group], model.predict(group)))
    i_choice, p_choice = RM.decide(model, group)
    r_choice = group[i_choice][0]
    assert abs(rows[r_choice]["P"] - P_res) / P_res < 1e-6, (r_choice, rows[r_choice]["P"], P_res)
    r_cat = min(rows, key=lambda r: abs(rows[r]["P"] / P_cat - 1))
    assert abs(rows[r_cat]["P"] / P_cat - 1) < 1e-3

    pins_e = pd.read_csv(ROOT / "data" / "b5_pinned_products.csv")
    pins_e = pins_e[(pins_e["tic"] == TIC) & (pins_e["role"] == "b5_early")].sort_values("sector")
    pins_l = pd.read_csv(ROOT / "data" / "b5_pins_audit.csv")
    pins_l = pins_l[pins_l["tic"] == TIC].sort_values("sector").head(LATER_SHOWN)
    early, later = load_sectors(pins_e), load_sectors(pins_l)
    te, fe = stitch(early, T)
    tl, fl = stitch(later, T)

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.7})
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.4), gridspec_kw={"width_ratios": [1, 1.05, 1]})

    cyc = np.floor((te - t0) / P_cat + 0.5).astype(int)
    ph = ((te - t0) / P_cat + 0.5) % 1 - 0.5
    for par, col, lab in ((0, "#1f4e79", "even cycles"), (1, "#c55a11", "odd cycles")):
        m = (cyc % 2) == par
        ax[0].plot(ph[m], fe[m], ",", color=col, alpha=0.25, rasterized=True)
        c, b = binned(ph[m], fe[m])
        ax[0].plot(c, b, "-", color=col, lw=1.0, label=lab)
        bmin = min(bmin, np.nanmin(b)) if par else np.nanmin(b)
    ax[0].set_xlim(-0.25, 0.25)
    ax[0].set_xlabel(f"Phase at catalog $P$ = {P_cat:.4f} d")
    ax[0].set_ylabel("Normalized flux")
    ax[0].set_title(f"(a) Early data, sectors {early[0][0]}–{early[-1][0]}", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")

    x = np.arange(len(ORDER))
    pv = [p_v2[r] for r in ORDER]
    ax[1].bar(x, pv, color=["#1f4e79" if r == r_choice else "#9e9e9e" for r in ORDER], width=0.7)
    ax[1].set_xticks(x, [f"{rows[r]['P']:.2f}" for r in ORDER], rotation=90, fontsize=6)
    ax[1].set_xlabel("Alias period (d)")
    ax[1].set_ylabel("v2 alias probability")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("(b) Resolver output (TLS seed)", fontsize=8)
    terms = {lab: rows[r_cat]["x"][k] - rows[r_choice]["x"][k] for k, lab in TERMS}
    txt = "\n".join([f"{rows[r_cat]['P']:.3f} d minus", f"{rows[r_choice]['P']:.3f} d:"] +
                    [f"  $\\Delta${lab} = {v:+.0f}" if abs(v) >= 1 else f"  $\\Delta${lab} = {v:+.2f}" for lab, v in terms.items()])
    ax[1].text(0.97, 0.97, txt, transform=ax[1].transAxes, va="top", ha="right", fontsize=5.8, family="serif")

    ph2 = ((tl - t0) / P_res + 0.25) % 1 - 0.25
    ax[2].plot(ph2, fl, ",", color="#7f7f7f", alpha=0.25, rasterized=True)
    e = np.linspace(-0.25, 0.75, 301)
    i = np.digitize(ph2, e) - 1
    bl = np.array([np.median(fl[i == k]) if np.any(i == k) else np.nan for k in range(300)])
    ax[2].plot(0.5 * (e[1:] + e[:-1]), bl, "-", color="#1f4e79", lw=1.0)
    bmin = min(bmin, np.nanmin(bl))
    ax[2].set_xlim(-0.25, 0.75)
    ax[2].set_xlabel(f"Phase at resolver $P$ = {P_res:.4f} d")
    ax[2].set_title(f"(c) Later data, sectors {later[0][0]}–{later[-1][0]}", fontsize=8)
    for a in (ax[0], ax[2]):
        a.set_ylim(bmin - 0.35 * (1 - bmin), 1 + 0.35 * (1 - bmin))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(ROOT / "paper" / "figures" / "fig_lc_example.pdf", dpi=300)

    out = {"tic": TIC, "P_catalog": P_cat, "P_resolver": P_res, "r_choice": r_choice, "p_v2_choice": p_choice,
           "abstain_threshold_v2": thr["learned"], "p_v2_by_alias": {r: float(p_v2[r]) for r in ORDER},
           "terms_catalog_minus_choice": terms, "delta_chi2_later": float(rel["delta_chi2"]),
           "early_sectors": [s[0] for s in early], "later_sectors_shown": [s[0] for s in later],
           "seed_period_tls": rec["summary"]["P0"]}
    (ROOT / "results" / "fig_lc_example.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

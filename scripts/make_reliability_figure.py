"""Reliability diagrams (post hoc presentation of the pre-registered calibration metric; no new result).
Ten equal-width bins of the top probability, as in the ECE. Each curve's ECE is recomputed here from the per-target values
and asserted equal to the value in the committed report that the manuscript quotes (CLAUDE.md rule 5).
(a) injections: B1(i) wrong-alias seeds with v1; B1(ii) TLS seeds with v1 and with v2. (b) real targets: B2 and B5 with v2.
Output: paper/figures/fig_reliability.pdf and results/fig_reliability.json."""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua.resolver import combine as C  # noqa: E402
from b1_run import read_records  # noqa: E402
import resolver_models as RM  # noqa: E402

TOL, MIN_N = 1e-3, 20


def b1(pattern, model):
    inj = pd.read_csv(ROOT / "data/b1_injections.csv").set_index("inj_id")
    conf, ok, seen = [], [], set()
    for p in sorted(glob.glob(str(ROOT / f"results/kaggle/b1/*/b1/{pattern}.jsonl.gz"))):
        for r in read_records(p):
            if r["inj_id"] in seen or inj.loc[r["inj_id"], "split"] != "test":
                continue
            seen.add(r["inj_id"])
            if "error" in r:
                conf.append(0.0); ok.append(False)
                continue
            i, c = RM.decide(model, [(x["r"], x["P"], x["x"]) for x in r["rows"]])
            conf.append(c); ok.append(abs(r["rows"][i]["P"] / float(inj.loc[r["inj_id"], "P_true"]) - 1) < TOL)
    return np.array(conf), np.array(ok, bool)


def bins(conf, ok):
    e = np.linspace(0, 1, 11)
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "mean_confidence": float(conf[m].mean()), "accuracy": float(ok[m].mean())})
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v1, v2 = RM.load("v1")[0], RM.load("v2")[0]
    rep1 = json.loads((ROOT / "results/b1_v2_posthoc_report.json").read_text())
    b2r = json.loads((ROOT / "results/b2_report.json").read_text())["all"]["resolver"]["tls"]["v2"]
    b5r = json.loads((ROOT / "results/b5_report.json").read_text())["primary"]["resolver_seeded_tls"]["v2"]
    b2 = pd.read_csv(ROOT / "results/b2_per_target.csv")
    b5 = pd.read_csv(ROOT / "results/b5_per_target.csv")
    b5 = b5[b5["primary_sample"]]
    curves = [("a", "B1(i), wrong-alias seeds, v1", *b1("b1_wrong_*", v1), rep1["v1"]["B1_i_wrong_seed"]["all"]["ece"], "#1f4e79", "o", "-"),
              ("a", "B1(ii), TLS seeds, v1", *b1("b1_search_tls_*", v1), rep1["v1"]["B1_ii_tls_seed"]["all"]["ece"], "#c55a11", "s", "--"),
              ("a", "B1(ii), TLS seeds, v2 (post hoc)", *b1("b1_search_tls_*", v2), rep1["v2"]["B1_ii_tls_seed"]["all"]["ece"], "#548235", "D", "-"),
              ("b", "B2, TOIs, v2", b2["conf"].to_numpy(float), b2["ok_ours"].to_numpy(bool), b2r["ece"], "#1f4e79", "o", "-"),
              ("b", "B5, eclipsing binaries, v2", b5["confidence_v2"].to_numpy(float), b5["ok_resolver"].to_numpy(bool), b5r["ece"], "#c55a11", "s", "--")]
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.7})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    out = {}
    for panel, label, conf, ok, reported, col, mk, ls in curves:
        e = C.ece(conf, ok)
        assert abs(e - reported) < 1e-6, (label, e, reported)  # the legend value is the reported one
        bb = bins(conf, ok)
        out[label] = {"n": int(conf.size), "ece": e, "bins": bb}
        big = [b for b in bb if b["n"] >= MIN_N]
        a = ax[0 if panel == "a" else 1]
        a.plot([b["mean_confidence"] for b in big], [b["accuracy"] for b in big], ls, marker=mk, color=col, ms=3.5, lw=1.0, label=f"{label} (ECE {e:.3f})")
    for a, t in zip(ax, ("(a) injections", "(b) real targets")):
        a.plot([0, 1], [0, 1], ":", color="k", lw=0.6)
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.set_xlabel("Stated probability of the chosen alias")
        a.set_title(t, fontsize=9, loc="left")
        a.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax[0].set_ylabel("Fraction correct")
    fig.tight_layout()
    fig.savefig(ROOT / "paper/figures/fig_reliability.pdf")
    (ROOT / "results/fig_reliability.json").write_text(json.dumps({"min_targets_per_plotted_bin": MIN_N, "curves": out}, indent=1))
    for k, v in out.items():
        print(k, v["n"], round(v["ece"], 4), [(round(b["mean_confidence"], 2), round(b["accuracy"], 2), b["n"]) for b in v["bins"]])


if __name__ == "__main__":
    main()

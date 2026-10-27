"""Generate every number, table, figure and released data table used by paper/ms.tex from committed results files
(CLAUDE.md rules 1, 5, 8). Nothing in the manuscript is typed by hand.

Writes
  paper/generated_macros.tex           \\newcommand macros (one per reported number)
  paper/tables/*.tex                   deluxetables
  paper/figures/*.pdf                  figures; each asserts that what it plots equals the results file
  paper/data/b5_candidate_period_revisions.csv  catalog-truth audit cases decided by later data (machine-readable)
  paper/data/b2_per_target.csv         B2 per-TOI table (sectors, SNR, periods, correctness)
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
P = ROOT / "paper"
MACROS: dict[str, str] = {}


def J(name):
    return json.loads((R / name).read_text())


def mac(name, value):
    assert name.isalpha(), name
    assert name not in MACROS, f"duplicate macro {name}"
    MACROS[name] = value


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def f1(x):
    return f"{x:.1f}"


def pc(x):  # percent, one decimal
    return f"{100 * x:.1f}"


def n(x):
    return f"{int(x):,}".replace(",", "{,}")


def ci(c, fmt=f3):
    return f"[{fmt(c[0])}, {fmt(c[1])}]"


def cip(c):
    return f"[{pc(c[0])}, {pc(c[1])}]"


def pval(p):
    """Relation and value for math mode, e.g. '= 0.064' or '= 9\\times10^{-27}'; exact zero (float underflow) -> '< 10^{-300}'."""
    if p == 0:
        return "<10^{-300}"
    if p >= 0.001:
        return f"={p:.3f}" if p < 0.995 else "=1.00"
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    return f"={m:.0f}\\times10^{{{e}}}" if m < 9.95 else f"=1\\times10^{{{e + 1}}}"


def acc_cell(d):
    return f"{f3(d['accuracy'])} {ci(d.get('ci95', d.get('accuracy_ci95')))}"


# ------------------------------------------------------------------ macros
def macros():
    gen = J("b1_generation.json")
    pool = J("p08_screen_report.json")
    mac("poolFrozen", n(pool["pool_stars"]))
    mac("poolScreened", n(pool["A6_rule_in_effect"]["total"]))
    mac("poolPseven", n(pool["p07_exclude"]))
    mac("poolScreenedRaw", n(pool["raw_rule_V1_sensitivity"]["total"]))
    mac("bOneInj", n(gen["injections"]))
    mac("bOneHosts", n(gen["hosts"]))
    for s, k in (("fit", "Fit"), ("calib", "Calib"), ("test", "Test")):
        mac(f"bOneHosts{k}", n(gen["hosts_by_split"][s]))
        mac(f"bOneInj{k}", n(gen["injections_by_split"][s]))
    tr = J("b1_training_report.json")
    mac("tempVone", f3(tr["temperature"]))
    mac("thrVoneLearned", f2(tr["abstain_threshold"]["learned"]))
    mac("thrVonePrincipled", f2(tr["abstain_threshold"]["principled"]))
    v2c = J("v2_calibration_final.json")
    mac("thrVtwoLearned", f2(v2c["abstain_threshold"]["learned"]))
    mac("vtwoCalRuns", n(v2c["n_runs"]))
    diag = J("v2_calibration_diagnostic.json")
    cv = diag["cross_validated_ece"]
    mac("cvEceTempTruth", f3(cv["V2-a_temperature_truth_in_candidates"]["mean"]))
    mac("cvEceTempAll", f3(cv["temperature_all_runs"]["mean"]))
    mac("cvEceIso", f3(cv["isotonic_pmax_all_runs"]["mean"]))
    mac("calTruthAbsentFrac", pc(1 - diag["truth_in_candidates"] / diag["runs"]))
    mac("calTruthAbsentConf", f2(diag["v1_on_calibration_data"]["mean_conf_truth_absent"]))
    mac("calEceTruthIn", f3(diag["v1_on_calibration_data"]["ece_truth_in_candidates"]))

    b1 = J("b1_test_report.json")
    post = J("b1_v2_posthoc_report.json")
    for key, tag in (("B1_i_wrong_seed", "W"), ("B1_ii_bls_seed", "B"), ("B1_ii_tls_seed", "T")):
        b = b1[key]
        a = b["learned"]["all"]
        mac(f"bOne{tag}N", n(b["n"]))
        mac(f"bOne{tag}Seed", f3(b["seed_alone_accuracy"]))
        mac(f"bOne{tag}Acc", f3(a["accuracy"]))
        mac(f"bOne{tag}AccCI", ci(a["accuracy_ci95"]))
        mac(f"bOne{tag}EceVone", f3(a["ece"]))
        mac(f"bOne{tag}EceVtwo", f3(post["v2"][key]["all"]["ece"]))
        mac(f"bOne{tag}AbstVtwo", pc(post["v2"][key]["all"]["abstain_rate"]))
        mac(f"bOne{tag}NaVtwo", f3(post["v2"][key]["all"]["accuracy_non_abstained"]))
        mac(f"bOne{tag}Ceil", f3(b["truth_in_candidates"]))
        mac(f"bOne{tag}Pr", f3(b["principled"]["all"]["accuracy"]))
        mac(f"bOne{tag}PrEce", f3(b["principled"]["all"]["ece"]))
        mac(f"bOne{tag}AccNinety", f3(a["accuracy_at_90pct_coverage"]))
        m = b["mcnemar_learned_vs_seed"]
        mac(f"bOne{tag}McP", pval(m["p_value"]))
        mac(f"bOne{tag}McFix", n(m["only_learned_correct"]))
        mac(f"bOne{tag}McBreak", n(m["only_principled_correct"]))
        mac(f"bOne{tag}Err", n(b["errors_counted_wrong"]))
    mac("bOneTMissPc", pc(1 - b1["B1_ii_tls_seed"]["truth_in_candidates"]))
    mac("bOneBMissPc", pc(1 - b1["B1_ii_bls_seed"]["truth_in_candidates"]))
    w = b1["B1_i_wrong_seed"]["learned"]
    mac("bOneTwinAcc", f3(w["by_cls"]["eb_twin"]["accuracy"]))
    mac("bOnePlanetAcc", f3(w["by_cls"]["planet"]["accuracy"]))
    by_r = {k: v["accuracy"] for k, v in w["by_seed_r"].items()}
    mac("bOneAliasMin", f3(min(by_r.values())))
    mac("bOneAliasMax", f3(max(by_r.values())))
    mac("bOneTwinPr", f3(b1["B1_i_wrong_seed"]["principled"]["by_cls"]["eb_twin"]["accuracy"]))
    mac("bOneTwinTls", f3(b1["B1_ii_tls_seed"]["learned"]["by_cls"]["eb_twin"]["accuracy"]))
    mac("bOneStratLow", f3(w["by_stratum"]["2-3"]["accuracy"]))
    mac("bOneStratHigh", f3(w["by_stratum"]["13-inf"]["accuracy"]))
    mac("bOneTEpochFew", f3(b1["B1_ii_tls_seed"]["learned"]["by_epoch_bin"]["0-1"]["accuracy"]))
    mac("bOneEpochFew", f3(w["by_epoch_bin"]["0-1"]["accuracy"]))
    mac("bOneEpochMany", f3(w["by_epoch_bin"][">=4"]["accuracy"]))
    pm = b1["B1_i_wrong_seed"]["mcnemar"]
    mac("bOneLearnVsPrP", pval(pm["p_value"]))

    # B5
    b5 = J("b5_report.json")
    sel = J("b5_selection.json")
    mac("bFiveCands", n(sel["candidates_before_cap"]))
    mac("bFiveIncluded", n(b5["included"]))
    mac("bFiveHthree", n(b5["h3_pass"]))
    mac("bFiveN", n(b5["primary_sample"]))
    mac("bFiveAmbN", n(b5["tier_b_amb"]["n"]))
    pr = b5["primary"]
    mac("bFiveRawTls", f3(pr["raw"]["tls"]["accuracy"]))
    mac("bFiveRawTlsCI", ci(pr["raw"]["tls"]["ci95"]))
    mac("bFiveRawBls", f3(pr["raw"]["bls"]["accuracy"]))
    mac("bFiveRawBlsCI", ci(pr["raw"]["bls"]["ci95"]))
    v = pr["resolver_seeded_tls"]["v2"]
    mac("bFiveAcc", f3(v["accuracy"]))
    mac("bFiveAccCI", ci(v["ci95"]))
    mac("bFiveEce", f3(v["ece"]))
    mac("bFiveAbst", pc(v["abstain_rate"]))
    mac("bFiveNa", f3(v["accuracy_non_abstained"]["accuracy"]))
    mac("bFiveCeil", f3(pr["resolver_seeded_tls"]["ceiling"]["accuracy"]))
    mac("bFiveAccBlsSeed", f3(pr["resolver_seeded_bls"]["v2"]["accuracy"]))
    for key, tag in (("PRIMARY_v2_tls_vs_raw_tls", "Tls"), ("PRIMARY_v2_tls_vs_raw_bls", "Bls"), ("v2_bls_vs_raw_bls", "BlsSeed")):
        m = pr[key]
        mac(f"bFiveMc{tag}P", pval(m["p_value"]))
        mac(f"bFiveMc{tag}Fix", n(m["only_second_correct"]))
        mac(f"bFiveMc{tag}Break", n(m["only_first_correct"]))
    amb = b5["tier_b_amb"]
    mac("bFiveAmbAcc", f3(amb["resolver_seeded_tls"]["v2"]["accuracy"]))
    mac("bFiveAmbTls", f3(amb["raw"]["tls"]["accuracy"]))
    mac("bFiveAmbBls", f3(amb["raw"]["bls"]["accuracy"]))
    mac("bFiveAmbEce", f3(amb["resolver_seeded_tls"]["v2"]["ece"]))
    mac("bFiveShortBls", f3(b5["by_period"]["<1"]["raw"]["bls"]["accuracy"]))
    mac("bFiveShortAcc", f3(b5["by_period"]["<1"]["resolver_seeded_tls"]["v2"]["accuracy"]))
    mac("bFiveShortP", pval(b5["by_period"]["<1"]["PRIMARY_v2_tls_vs_raw_bls"]["p_value"]))
    mac("bFiveLongAcc", f3(b5["by_period"][">10"]["resolver_seeded_tls"]["v2"]["accuracy"]))
    mac("bFiveLongBls", f3(b5["by_period"][">10"]["raw"]["bls"]["accuracy"]))
    mac("bFiveLongP", pval(b5["by_period"][">10"]["PRIMARY_v2_tls_vs_raw_bls"]["p_value"]))
    for b, t in (("<1", "Short"), ("1-3", "OneThree"), ("3-10", "ThreeTen"), (">10", "Long")):
        bp = b5["by_period"][b]
        mac(f"bFive{t}N", n(bp["n"]))
        mac(f"bFive{t}Tls", f3(bp["raw"]["tls"]["accuracy"]))
        if t in ("OneThree", "ThreeTen"):
            mac(f"bFive{t}Acc", f3(bp["resolver_seeded_tls"]["v2"]["accuracy"]))
            mac(f"bFive{t}Bls", f3(bp["raw"]["bls"]["accuracy"]))
            mac(f"bFive{t}P", pval(bp["PRIMARY_v2_tls_vs_raw_bls"]["p_value"]))
        mac(f"bFive{t}TlsP", pval(bp["PRIMARY_v2_tls_vs_raw_tls"]["p_value"]))
    mac("bFiveChoiceErr", f3(pr["resolver_seeded_tls"]["ceiling"]["accuracy"] - v["accuracy"]))
    mac("bFiveCeilMiss", f3(1 - pr["resolver_seeded_tls"]["ceiling"]["accuracy"]))
    mac("bFiveTierB", n(sel["tier_included"]["B"]))
    mac("bFiveHthreeFail", n(b5["included"] - b5["h3_pass"]))
    mac("bFiveCatRows", n(sel["catalog_rows"]))
    mac("bFiveOneSecN", n(b5["by_n_early"]["1"]["n"]))
    mac("bFiveCovered", pc(1 - v["abstain_rate"]))
    mac("bFiveOneSecAcc", f3(b5["by_n_early"]["1"]["resolver_seeded_tls"]["v2"]["accuracy"]))
    mac("bFiveFourSecAcc", f3(b5["by_n_early"]["4"]["resolver_seeded_tls"]["v2"]["accuracy"]))

    au = J("b5_audit_report.json")
    a = au["all"]
    mac("auditN", n(a["n"]))
    mac("auditVtwo", n(a["later data support v2"]["count"]))
    mac("auditVtwoPc", pc(a["later data support v2"]["fraction"]))
    mac("auditCat", n(a["later data support the catalog"]["count"]))
    mac("auditCatPc", pc(a["later data support the catalog"]["fraction"]))
    mac("auditInc", n(a["inconclusive"]["count"]))
    mac("auditIncPc", pc(a["inconclusive"]["fraction"]))
    mac("auditIns", n(a["insufficient"]["count"]))
    bt = au["by_tier"]["B"]
    mac("auditBN", n(bt["n"]))
    mac("auditBVtwo", n(bt["later data support v2"]["count"]))
    mac("auditBCat", n(bt["later data support the catalog"]["count"]))
    ba = au["by_tier"]["B-amb"]
    mac("auditAmbN", n(ba["n"]))
    mac("auditAmbInc", n(ba["inconclusive"]["count"]))
    br = au["by_ratio"]
    mac("auditHalfN", n(br["1/2"]["n"]))
    mac("auditHalfInc", n(br["1/2"]["inconclusive"]["count"]))
    mac("auditDoubleN", n(br["2"]["n"]))
    mac("auditDoubleVtwo", n(br["2"]["later data support v2"]["count"]))
    mac("auditDoubleCat", n(br["2"]["later data support the catalog"]["count"]))
    mac("auditNonaliasN", n(br["non-alias"]["n"]))
    mac("auditNonaliasVtwo", n(br["non-alias"]["later data support v2"]["count"]))
    mac("auditNonaliasCat", n(br["non-alias"]["later data support the catalog"]["count"]))

    # B2
    b2 = J("b2_report.json")
    al = b2["all"]
    mac("bTwoCands", n(b2["records"]))
    mac("bTwoN", n(b2["primary_sample"]))
    mac("bTwoNA", n(b2["by_tier"]["A"]["n"]))
    mac("bTwoNB", n(b2["by_tier"]["B"]["n"]))
    r2 = al["resolver"]["tls"]["v2"]
    mac("bTwoAcc", f3(r2["accuracy"]))
    mac("bTwoAccCI", ci(r2["ci95"]))
    mac("bTwoEce", f3(r2["ece"]))
    mac("bTwoAbst", pc(r2["abstain_rate"]))
    mac("bTwoNa", f3(r2["accuracy_non_abstained"]["accuracy"]))
    mac("bTwoNaCI", ci(r2["accuracy_non_abstained"]["ci95"]))
    mac("bTwoCeil", f3(al["resolver"]["tls"]["truth_in_candidates_ceiling"]["accuracy"]))
    mac("bTwoEceVoneTls", f3(al["resolver"]["tls"]["v1"]["ece"]))
    mac("bTwoEceVoneBls", f3(al["resolver"]["bls"]["v1"]["ece"]))
    names = {"tls": "Tls", "bls": "Bls", "tschudi": "Tsch", "spoc1": "SpocOne", "spocE": "SpocE", "qlp_hist": "Qlp", "spoc_hist": "SpocHist"}
    for s, t in names.items():
        mac(f"bTwoRaw{t}", f3(al["baselines"][s]["accuracy"]))
        mac(f"bTwoCov{t}", pc(al["baselines"][s]["coverage"]))
        m = al["paired"][f"PRIMARY_v2_tls_vs_baseline_{s}"]
        mac(f"bTwoMc{t}P", pval(m["p_value"]))
        mac(f"bTwoMc{t}Fix", n(m["only_second_correct"]))
        mac(f"bTwoMc{t}Break", n(m["only_first_correct"]))
        g = al["paired"][f"PRIMARY_v2_tls_vs_baseline_{s}_where_period_given"]
        mac(f"bTwoMcGiven{t}P", pval(g["p_value"]))
        mac(f"bTwoMcGiven{t}Fix", n(g["only_second_correct"]))
        mac(f"bTwoMcGiven{t}Break", n(g["only_first_correct"]))
        mac(f"bTwoGiven{t}", f3(al["baselines"][s]["accuracy_where_period_given"]["accuracy"]))
        rs = al["paired"][f"resolver_v2_seeded_{s}_vs_raw_{s}"]
        mac(f"bTwoSelf{t}Fix", n(rs["only_second_correct"]))
        mac(f"bTwoSelf{t}Break", n(rs["only_first_correct"]))
        mac(f"bTwoSelf{t}P", pval(rs["p_value"]))
        mac(f"bTwoEceVtwo{t}", f3(al["resolver"][s]["v2"]["ece"]))
        mac(f"bTwoCeil{t}", f3(al["resolver"][s]["truth_in_candidates_ceiling"]["accuracy"]))
    for t, k in (("A", "A"), ("B", "B")):
        bb = b2["by_tier"][k]
        mac(f"bTwoTier{t}Acc", f3(bb["resolver"]["tls"]["v2"]["accuracy"]))
        mac(f"bTwoTier{t}Tls", f3(bb["baselines"]["tls"]["accuracy"]))
        mac(f"bTwoTier{t}P", pval(bb["paired"]["PRIMARY_v2_tls_vs_baseline_tls"]["p_value"]))
        mac(f"bTwoTier{t}Ece", f3(bb["resolver"]["tls"]["v2"]["ece"]))
    mac("bTwoMissTlsPc", pc(1 - al["resolver"]["tls"]["truth_in_candidates_ceiling"]["accuracy"]))
    one = b2["by_stratum"]["1"]
    mac("bTwoOneSecN", n(one["n"]))
    mac("bTwoOneSecAcc", f3(one["resolver"]["tls"]["v2"]["accuracy"]))
    mac("bTwoOneSecTls", f3(one["baselines"]["tls"]["accuracy"]))
    mac("bTwoOneSecP", pval(one["paired"]["PRIMARY_v2_tls_vs_baseline_tls"]["p_value"]))

    ph = J("b2_posthoc_report.json")
    A = ph["A_selective"]
    mac("selAnswered", n(A["answered"]))
    mac("selCov", pc(A["coverage"]))
    mac("selOurs", f3(A["ours"]["accuracy"]))
    for s, t in (("tls", "Tls"), ("bls", "Bls"), ("spoc1", "SpocOne"), ("spocE", "SpocE")):
        v_ = A["on_answered_subset"][s]
        mac(f"sel{t}", f3(v_["accuracy"]))
        mac(f"sel{t}P", pval(v_["mcnemar_vs_ours"]["p_value"]))
    ab = A["on_abstained_subset"]
    mac("abstN", n(ab["ours"]["n"]))
    mac("abstOurs", f3(ab["ours"]["accuracy"]))
    mac("abstTls", f3(ab["tls"]["accuracy"]))
    mac("abstBls", f3(ab["bls"]["accuracy"]))
    B = ph["B_risk_coverage"]["at_ours_coverage"]
    mac("rcOurs", f3(B["ours_v2_conf"]["accuracy"]))
    mac("rcTls", f3(B["tls_by_sde"]["accuracy"]))
    mac("rcBls", f3(B["bls_by_power"]["accuracy"]))
    mac("aucOurs", f3(B["ours_v2_conf"]["area_under_accuracy_coverage"]))
    mac("aucTls", f3(B["tls_by_sde"]["area_under_accuracy_coverage"]))
    mac("aucBls", f3(B["bls_by_power"]["area_under_accuracy_coverage"]))
    C = ph["C_B1_vs_B2"]
    mac("cmpTlsBOne", f3(C["raw_tls_accuracy"]["B1ii"]["accuracy"]))
    mac("cmpTlsBTwo", f3(C["raw_tls_accuracy"]["B2"]["accuracy"]))
    mac("cmpEbBOne", pc(C["eb_fraction"]["B1ii"]))
    mac("cmpEbBTwo", pc(C["eb_fraction"]["B2"]))
    mac("cmpOneSecBOne", pc(C["single_sector_fraction"]["B1ii"]))
    mac("cmpOneSecBTwo", pc(C["single_sector_fraction"]["B2"]))
    D = ph["D_B3_go_no_go"]["current"]
    mac("nogoN", n(D["n_confident_disagreements"]))
    mac("nogoRight", f3(D["v2_right"]["accuracy"]))
    mac("nogoCatRight", f3(D["reference_right"]["accuracy"]))

    fa = J("false_alarm_report.json")["variants"]
    for v_, t in (("scrambled", "Scr"), ("inverted", "Inv")):
        mac(f"fa{t}N", n(fa[v_]["n"]))
        for c, ct in (("v2", "Vtwo"), ("v1", "Vone"), ("principled", "Pr")):
            mac(f"fa{t}{ct}", pc(fa[v_][c]["rate"]))
            mac(f"fa{t}{ct}CI", cip(fa[v_][c]["ci95"]))
    mac("faScrSeedMed", f1(fa["scrambled"]["seed_periods_median_d"]))
    mac("faInvSeedMed", f2(fa["inverted"]["seed_periods_median_d"]))
    c = J("fa_context_report.json")  # post hoc, decision FA-ctx
    for v_, t in (("scrambled", "Scr"), ("inverted", "Inv")):
        for k, kt in (("a6_sde_ge9", "Asix"), ("raw_sde_ge9", "Raw")):
            mac(f"ctx{t}{kt}", pc(c[v_][k]["rate"]))
            mac(f"ctx{t}{kt}CI", cip(c[v_][k]["ci95"]))
        mac(f"ctx{t}AsixN", n(c[v_]["a6_sde_ge9"]["count"]))
        mac(f"ctx{t}Both", pc(c[v_]["resolver_confident_given_a6_sde_ge9"]["rate"]))
        mac(f"ctx{t}BothCI", cip(c[v_]["resolver_confident_given_a6_sde_ge9"]["ci95"]))
        mac(f"ctx{t}Low", pc(c[v_]["resolver_confident_given_a6_sde_lt9"]["rate"]))
        mac(f"ctx{t}LowCI", cip(c[v_]["resolver_confident_given_a6_sde_lt9"]["ci95"]))
        mac(f"ctx{t}Joint", pc(c[v_]["resolver_confident_and_a6_sde_ge9"]["rate"]))

    ms = json.loads((P / "data" / "mast" / "summary.json").read_text())  # scripts/make_mast_doi_lists.py
    mac("mastFiles", n(ms["all_files"]))
    mac("mastStars", n(ms["all_stars"]))
    mac("mastSecMin", str(ms["sector_min"]))
    mac("mastSecMax", str(ms["sector_max"]))
    mac("mastSubFiles", n(ms["subset_files"]))
    mac("mastSubStars", n(ms["subset_stars"]))

    rr = J("b1_raw_rule_sensitivity.json")  # amendment A6 sensitivity
    mac("rawHostsBoth", n(rr["test_hosts"]["pass_raw_rule"]))
    mac("rawHostsOnly", n(rr["test_hosts"]["a6_only"]))
    for sub, t in (("pass_raw_rule", "Both"), ("a6_only", "Only")):
        for key, kt in (("B1_i_wrong_seed", "W"), ("B1_ii_tls_seed", "T"), ("B1_ii_bls_seed", "B")):
            le = rr[sub][key]["learned"]
            mac(f"raw{t}{kt}Acc", f3(le["accuracy"]))
            mac(f"raw{t}{kt}AccCI", ci(le["accuracy_ci95"]))
            mac(f"raw{t}{kt}Ece", f3(le["ece"]))

    lc = J("fig_lc_example.json")  # written by scripts/make_lc_figure.py
    mac("lcTic", str(lc["tic"]))
    mac("lcPcat", f"{lc['P_catalog']:.4f}")
    mac("lcPres", f"{lc['P_resolver']:.4f}")
    mac("lcPseed", f"{lc['seed_period_tls']:.4f}")
    mac("lcProb", f3(lc["p_v2_choice"]))
    mac("lcDchi", f"{lc['delta_chi2_later']:.0f}")
    for k, t in (("F1", "One"), ("F2", "Two"), ("F3", "Three")):
        mac(f"lcDF{t}", f"{lc['terms_catalog_minus_choice'][k]:+.0f}".replace("-", "$-$"))
    mac("lcEarlySec", f"{lc['early_sectors'][0]}--{lc['early_sectors'][-1]}")
    mac("lcEarlyN", str(len(lc["early_sectors"])))
    mac("lcLaterSec", f"{lc['later_sectors_shown'][0]}--{lc['later_sectors_shown'][-1]}")
    mac("lcLaterN", str(len(lc["later_sectors_shown"])))

    rt = J("b4_runtime.json")
    pl = rt["per_light_curve"]
    mac("bFourN", n(rt["complete_pipeline"]))
    mac("bFourPinned", n(rt["pinned"]))
    for k, t in (("total", "Total"), ("resolver", "Res"), ("gpu_bls", "Bls"), ("prepare", "Prep"), ("load", "Load")):
        mac(f"bFour{t}Med", f2(pl[k]["median_s"]))
        mac(f"bFour{t}Pnf", f2(pl[k]["p95_s"]))
    mac("bFourTlsMed", f1(rt["tls_quoted_from_B2"]["median_s"]))
    mac("bFourTlsPnf", f"{rt['tls_quoted_from_B2']['p95_s']:.0f}")
    mac("bFourGpu", rt["hardware"]["bls"]["gpu"].replace(" ", "~"))
    mac("bFourCpuRes", rt["hardware"]["resolver"]["cpu_model"].replace("(R)", "").replace(" CPU", "").replace("  ", " ").strip())
    mac("bFourCpuBls", rt["hardware"]["bls"]["cpu_model"].replace("(R)", "").replace(" CPU", "").replace("  ", " ").strip())
    mac("bFourCores", n(rt["hardware"]["resolver"]["cpu_count"]))
    sw = rt["stage_wall_h"]
    mac("bFourResWallH", f1(sum(v for k, v in sw.items() if k.startswith("resolver"))))
    mac("bFourBlsWallH", f1(sum(v for k, v in sw.items() if k.startswith("bls"))))
    mac("bFourTlsMax", f"{rt['tls_quoted_from_B2']['max_s']:.0f}")

    lines = ["% AUTO-GENERATED by scripts/make_paper_assets.py from results/*.json -- do not edit",
             f"% git commit at generation: {subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (P / "generated_macros.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ tables
def table(path, caption, label, colspec, head, rows, notes=""):
    t = [f"\\begin{{deluxetable*}}{{{colspec}}}", "\\tablecaption{" + caption + f" \\label{{{label}}}}}", "\\tablewidth{0pt}",
         "\\tablehead{" + " & ".join(f"\\colhead{{{h}}}" for h in head) + "}", "\\startdata"]
    t += [" & ".join(r) + " \\\\" for r in rows]
    t[-1] = t[-1][:-3]
    t += ["\\enddata"]
    if notes:
        t += [f"\\tablecomments{{{notes}}}"]
    t += ["\\end{deluxetable*}"]
    (P / "tables" / path).write_text("\n".join(t) + "\n")


def tables():
    (P / "tables").mkdir(exist_ok=True)
    gen, sel, b5, b2, b4 = J("b1_generation.json"), J("b5_selection.json"), J("b5_report.json"), J("b2_report.json"), J("b4_runtime.json")
    pool = J("p08_screen_report.json")
    table("samples.tex", "Benchmark samples.", "tab:samples", "llrl",
          ["Benchmark", "Source", "Scored", "Truth"],
          [["B1 injections (test split)", f"{n(gen['hosts_by_split']['test'])} of {n(gen['hosts'])} quiet stars", n(gen["injections_by_split"]["test"]), "injected period"],
           ["B5 eclipsing binaries", "Pr\\v{s}a et al.\\ (2022) catalog", n(b5["primary_sample"]), "catalog period"],
           ["B2 TOI temporal holdout", "TOI catalog snapshot", n(b2["primary_sample"]), "Exoplanet Archive / EB catalog"],
           ["False-alarm test", "B2 light curves, inverted or scrambled", n(J("false_alarm_report.json")["variants"]["scrambled"]["n"]) + " each", "no coherent signal"],
           ["B4 runtime", "TESS Sector 49", n(b4["complete_pipeline"]), "\\nodata"]],
          "Counts after all pre-registered selection and eligibility rules (Section~\\ref{sec:data}).")

    pr = b5["primary"]
    v, vb, v1 = pr["resolver_seeded_tls"]["v2"], pr["resolver_seeded_bls"]["v2"], pr["resolver_seeded_tls"]["v1"]
    table("b5.tex", "B5: alias resolution on TESS eclipsing binaries (primary sample).", "tab:b5", "lcccc",
          ["Method", "Accuracy [95\\% CI]", "ECE", "Abstain", "McNemar vs.\\ raw seed"],
          [["Raw TLS peak", acc_cell(pr["raw"]["tls"]), "\\nodata", "\\nodata", "\\nodata"],
           ["Raw BLS peak", acc_cell(pr["raw"]["bls"]), "\\nodata", "\\nodata", "\\nodata"],
           ["Resolver v2, TLS seed (primary)", acc_cell(v), f3(v["ece"]), pc(v["abstain_rate"]) + "\\%",
            f"${pr['PRIMARY_v2_tls_vs_raw_tls']['only_second_correct']}/{pr['PRIMARY_v2_tls_vs_raw_tls']['only_first_correct']}$, $p{pval(pr['PRIMARY_v2_tls_vs_raw_tls']['p_value'])}$"],
           ["Resolver v2, BLS seed", acc_cell(vb), f3(vb["ece"]), pc(vb["abstain_rate"]) + "\\%",
            f"${pr['v2_bls_vs_raw_bls']['only_second_correct']}/{pr['v2_bls_vs_raw_bls']['only_first_correct']}$, $p{pval(pr['v2_bls_vs_raw_bls']['p_value'])}$"],
           ["Resolver v1, TLS seed", acc_cell(v1), f3(v1["ece"]), pc(v1["abstain_rate"]) + "\\%", "\\nodata"]],
          f"Primary sample: {n(b5['primary_sample'])} eclipsing binaries with at least two early-window primary eclipses, excluding near-equal eclipses (B-amb). Accuracy is top-1 (no abstention); "
          "a period is correct if within 0.1\\% of the catalog period. McNemar columns give discordant pairs won by the resolver / "
          f"by the raw seed and the exact two-sided $p$. Against raw BLS the TLS-seeded resolver gives "
          f"${pr['PRIMARY_v2_tls_vs_raw_bls']['only_second_correct']}/{pr['PRIMARY_v2_tls_vs_raw_bls']['only_first_correct']}$, "
          f"$p{pval(pr['PRIMARY_v2_tls_vs_raw_bls']['p_value'])}$.")

    au = J("b5_audit_report.json")
    lab = {"1/2": "$P_{\\rm res}=P_{\\rm cat}/2$", "2": "$P_{\\rm res}=2P_{\\rm cat}$", "non-alias": "not an alias", "3": "$P_{\\rm res}=3P_{\\rm cat}$",
           "4": "$P_{\\rm res}=4P_{\\rm cat}$"}
    rows = []
    for k, t in au["by_ratio"].items():
        rows.append([lab.get(k, k), n(t["n"]), n(t["later data support v2"]["count"]), n(t["later data support the catalog"]["count"]),
                     n(t["inconclusive"]["count"]), n(t["insufficient"]["count"])])
    a = au["all"]
    rows.append(["All", n(a["n"]), n(a["later data support v2"]["count"]), n(a["later data support the catalog"]["count"]),
                 n(a["inconclusive"]["count"]), n(a["insufficient"]["count"])])
    table("audit.tex", "Catalog-truth audit of B5 disagreements (secondary analysis).", "tab:audit", "lrrrrr",
          ["Relation", "$N$", "Favors resolver", "Favors catalog", "Inconclusive", "Insufficient"], rows,
          "Each disagreement between the resolver and the catalog period was tested on the later-sector light curve by fitting "
          "the same eclipsing-binary model (primary and free-phase secondary) at both periods; a decision requires "
          "$|\\Delta\\chi^2|>10$. The decided cases are released as candidate period revisions, not confirmations.")

    al = b2["all"]
    names = (("tls", "Raw TLS peak"), ("bls", "Raw BLS peak"), ("tschudi", "Tschudi (2026a) correction"), ("spoc1", "SPOC TCE, single sector"),
             ("spocE", "SPOC TCE, multi-sector"), ("qlp_hist", "QLP (G21 catalog)"), ("spoc_hist", "SPOC (G21 catalog)"))
    rows = [["Resolver v2, TLS seed", pc(1.0) + "\\%", acc_cell(al["resolver"]["tls"]["v2"]), "\\nodata", "\\nodata"]]
    for s, lbl in names:
        b = al["baselines"][s]
        m = al["paired"][f"PRIMARY_v2_tls_vs_baseline_{s}"]
        g = al["paired"][f"PRIMARY_v2_tls_vs_baseline_{s}_where_period_given"]
        rows.append([lbl, pc(b["coverage"]) + "\\%", acc_cell(b), f"${m['only_second_correct']}/{m['only_first_correct']}$, $p{pval(m['p_value'])}$",
                     f"${g['only_second_correct']}/{g['only_first_correct']}$, $p{pval(g['p_value'])}$"])
    table("b2.tex", "B2: temporal holdout on TOIs (primary sample).", "tab:b2", "lcccc",
          ["Method", "Coverage", "Accuracy [95\\% CI]", "Resolver vs.\\ method", "Same, where method gives a period"], rows,
          f"{n(b2['primary_sample'])} TOIs with Tier A or B truth and at least two transits in the early window. A method that gives no "
          "period counts as wrong; the last column restricts to TOIs where it gives one. McNemar entries: discordant pairs won by "
          "the resolver / by the method, exact two-sided $p$.")

    b1 = J("b1_test_report.json")
    post = J("b1_v2_posthoc_report.json")
    rows = []
    for key, lbl in (("B1_i_wrong_seed", "Wrong alias seed"), ("B1_ii_bls_seed", "BLS search seed"), ("B1_ii_tls_seed", "TLS search seed")):
        b = b1[key]
        rows.append([lbl, n(b["n"]), f3(b["seed_alone_accuracy"]), acc_cell(b["learned"]["all"]), f3(b["learned"]["all"]["ece"]),
                     f3(post["v2"][key]["all"]["ece"]), f3(b["truth_in_candidates"])])
    table("b1.tex", "B1: injection-recovery test split.", "tab:b1", "lrccccc",
          ["Seed", "$N$", "Seed alone", "Resolver accuracy [95\\% CI]", "ECE v1", "ECE v2$^a$", "Ceiling"], rows,
          "Accuracy and v1 ECE are pre-registered results scored after the v1 freeze. $^a$v2 ECE is post hoc: v2 was defined after "
          "the v1 search-seed results were known. Accuracy is identical for v1 and v2 by construction. Ceiling: fraction of "
          "injections for which one of the eleven refined candidates matches the truth.")

    fa = J("false_alarm_report.json")["variants"]
    rows = []
    for v_, lbl in (("scrambled", "Scrambled"), ("inverted", "Inverted")):
        r_ = [lbl, n(fa[v_]["n"]), f"{pc(fa[v_]['v2']['rate'])}\\% {cip(fa[v_]['v2']['ci95'])}", f"{pc(fa[v_]['v1']['rate'])}\\%", f"{pc(fa[v_]['principled']['rate'])}\\%"]
        c = J("fa_context_report.json")[v_]
        r_ += [f"{pc(c['a6_sde_ge9']['rate'])}\\%", f"{pc(c['resolver_confident_given_a6_sde_ge9']['rate'])}\\%"]
        rows.append(r_)
    table("fa.tex", "False-alarm test: fraction of light curves resolved with confidence.", "tab:fa", "lrccccc",
          ["Light curves", "$N$", "v2 [95\\% CI]", "v1", "Principled", "BLS SDE$\\ge9^a$", "v2 if SDE$\\ge9^a$"], rows,
          "The pre-registered criterion is a confident-resolution rate below 5\\% on scrambled light curves. $^a$Post hoc: the "
          "fraction of light curves on which the BLS search that supplied the seed reaches SDE~$\\ge9$ (SDE on the detrended "
          "spectrum at TLS's trial periods), and the \\Vtwo\\ confident rate among those light curves.")

    pl = b4["per_light_curve"]
    rows = [[lbl, f2(pl[k]["median_s"]), f2(pl[k]["p95_s"]), hw] for k, lbl, hw in (
        ("load", "Read light curve", "GPU-session CPU"), ("prepare", "Detrend and bin", "GPU-session CPU"),
        ("gpu_bls", "BLS seed search", b4["hardware"]["bls"]["gpu"]), ("resolver", "Alias resolver (v2)", "CPU session"),
        ("total", "Total", "\\nodata"))]
    table("b4.tex", "B4: processing time per light curve, TESS Sector 49.", "tab:b4", "lccl",
          ["Stage", "Median (s)", "95th percentile (s)", "Hardware"], rows,
          f"{n(b4['complete_pipeline'])} light curves. GPU session: one process per NVIDIA {b4['hardware']['bls']['gpu']}; CPU session: "
          f"{b4['hardware']['resolver']['cpu_count']} vCPU ({MACROS['bFourCpuRes']}), four worker processes with one BLAS thread each; "
          f"GPU-session CPU: {MACROS['bFourCpuBls']}.")


# ------------------------------------------------------------------ figures
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (P / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.7})

    b5 = J("b5_report.json")
    bins = ["<1", "1-3", "3-10", ">10"]
    labels = ["$<1$ d", "1\u20133 d", "3\u201310 d", "$>10$ d"]
    series = [("raw", "tls", "Raw TLS peak", "#9e9e9e", "o"), ("raw", "bls", "Raw BLS peak", "#616161", "s"),
              ("res", "tls", "Resolver v2 (TLS seed)", "#1f4e79", "D")]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(bins))
    for i, (kind, key, lbl, col, mk) in enumerate(series):
        accs, lo, hi = [], [], []
        for b in bins:
            d = b5["by_period"][b]["raw"][key] if kind == "raw" else b5["by_period"][b]["resolver_seeded_tls"]["v2"]
            accs.append(d["accuracy"]); lo.append(d["accuracy"] - d["ci95"][0]); hi.append(d["ci95"][1] - d["accuracy"])
        ax.errorbar(x + (i - 1) * 0.18, accs, yerr=[lo, hi], fmt=mk, color=col, ms=4, capsize=2, lw=0.8, label=lbl)
        assert np.allclose(accs, [(b5["by_period"][b]["raw"][key] if kind == "raw" else b5["by_period"][b]["resolver_seeded_tls"]["v2"])["accuracy"] for b in bins])
    ax.set_xticks(x, [f"{l}\n($N$={b5['by_period'][b]['n']})" for l, b in zip(labels, bins)])
    ax.set_ylabel("Period accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(P / "figures" / "fig_b5_period.pdf")
    plt.close(fig)

    rc = J("b2_posthoc_report.json")["B_risk_coverage"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, lbl, col, ls in (("ours_v2_conf", "Resolver v2 confidence", "#1f4e79", "-"), ("tls_by_sde", "TLS period ranked by SDE", "#c55a11", "--"),
                            ("bls_by_power", "BLS period ranked by power", "#7f7f7f", ":")):
        c = rc["curves"][k]
        ax.plot([p["coverage"] for p in c], [p["accuracy"] for p in c], ls, color=col, lw=1.2, label=lbl)
    cov = rc["at_ours_coverage"]["coverage"]
    ax.axvline(cov, color="k", lw=0.5, alpha=0.5)
    assert abs(cov - J("b2_posthoc_report.json")["A_selective"]["coverage"]) < 1e-12
    ax.set_xlabel("Coverage (fraction of TOIs answered)")
    ax.set_ylabel("Accuracy of answered TOIs")
    ax.set_ylim(0.75, 1.0)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(P / "figures" / "fig_b2_riskcov.pdf")
    plt.close(fig)

    b1 = J("b1_test_report.json")["B1_i_wrong_seed"]["learned"]["by_seed_r"]
    order = ["1/5", "1/4", "1/3", "1/2", "2/3", "1", "3/2", "2", "3", "4", "5"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    accs = [b1[r]["accuracy"] for r in order]
    ax.errorbar(range(len(order)), accs, yerr=[[a - b1[r]["accuracy_ci95"][0] for a, r in zip(accs, order)],
                                               [b1[r]["accuracy_ci95"][1] - a for a, r in zip(accs, order)]],
                fmt="o", color="#1f4e79", ms=4, capsize=2, lw=0.8)
    ax.set_xticks(range(len(order)), [f"${r}$" if "/" not in r else f"${r.split('/')[0]}/{r.split('/')[1]}$" for r in order])
    ax.set_xlabel("Seed period / true period")
    ax.set_ylabel("Resolver accuracy")
    ax.set_ylim(0.7, 1.0)
    fig.tight_layout()
    fig.savefig(P / "figures" / "fig_b1_alias.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ data products
def data():
    (P / "data").mkdir(exist_ok=True)
    d = pd.read_csv(R / "kaggle/b5_audit/b5audit/b5_audit.csv")
    keep = d[d["verdict"].isin(["later data support v2", "later data support the catalog"])].copy()
    keep["supported_period"] = np.where(keep["verdict"] == "later data support v2", keep["P_v2"], keep["P_catalog"])
    keep["status"] = np.where(keep["verdict"] == "later data support v2", "candidate revision (resolver period favored by later data)",
                              "catalog period favored by later data")
    cols = ["tic", "tier", "P_catalog", "P_v2", "ratio", "delta_chi2", "n_ecl_catalog", "n_ecl_v2", "supported_period", "status"]
    keep[cols].sort_values(["status", "delta_chi2"], ascending=[True, False]).to_csv(P / "data" / "b5_candidate_period_revisions.csv", index=False,
                                                                                     float_format="%.8g")
    au = J("b5_audit_report.json")["all"]
    assert (keep["verdict"] == "later data support v2").sum() == au["later data support v2"]["count"]
    assert (keep["verdict"] == "later data support the catalog").sum() == au["later data support the catalog"]["count"]
    shutil.copy(R / "b2_per_target.csv", P / "data" / "b2_per_target.csv")


def main():
    macros()
    tables()
    figures()
    data()
    print(f"{len(MACROS)} macros; tables, figures and data written")


if __name__ == "__main__":
    main()

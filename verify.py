"""Reproduction check.

Recomputes the numbers reported in the manuscript (version 8) and its
supplementary material and compares them, as printed, with the published
values. Run ``python verify.py``; the script exits with status 1 if any check
fails.

The first part checks the numbers quoted in the text, the flow chart and the
tables one by one: the computed value, rounded as in the publication, is
compared with the published value (counts and estimates to the printed number
of decimals, p-values in the journal format: ``< 0.001``, three decimals below
0.1, otherwise two). The second part checks the tables row by row: every
published row of Tables 1-3 and Supplementary Tables S1-S16 (values in
``published_tables.py``) must be reproduced exactly by the corresponding table
of this repository. Firth estimates use profile penalised-likelihood confidence
intervals and penalised likelihood-ratio p-values.

Checks that need the auxiliary file ``repeated_examinations.csv`` are reported
as skipped when it is not available.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import analysis_supplementary as supp
import analysis_tables as main_tables
import prep
from analysis_supplementary import MODEL_A, MODEL_B, _quantreg
from analysis_tables import firth_or, fmt_p, gmr, gmr_categorical
from firth import cochran_armitage, firth_profile, logit_mle, wilson_ci
from published_tables import PUBLISHED
from robustness import REPEAT_LABEL, analysis_sets, leave_one_out, tableS13

M1, M2, M3 = (), ("disease_duration_years",), ("disease_duration_years", "steroid_courses", "antitnf")


def main() -> int:
    pat_att, pat, ref = prep.load_all()
    ref_att = prep.load_reference()
    aza_users = pat[pat["aza_user"] == 1]
    thio_users = pat[pat["thio_user"] == 1]
    marked = pat[pat["ssm_ge_40"] == 1]
    checks: list[tuple[str, str, str, str]] = []
    skipped: list[tuple[str, str, str]] = []
    section = [""]

    def head(title):
        section[0] = title

    def num(desc, got, want, dp=2):
        g = f"{float(got):.{dp}f}"
        w = want if isinstance(want, str) else f"{float(want):.{dp}f}"
        checks.append((section[0], desc, g, w))

    def count(desc, got, want):
        num(desc, got, want, 0)

    def pval(desc, got, want):
        checks.append((section[0], desc, fmt_p(float(got)), want))

    def est(desc, r, want, dp=2, keys=("or", "low", "high")):
        """Estimate with 95 % CI, compared as printed, e.g. '1.39 (1.13-1.72)'."""
        a, b, c = (float(r[k]) for k in keys)
        checks.append((section[0], desc, f"{a:.{dp}f} ({b:.{dp}f}-{c:.{dp}f})", want))

    def med_iqr(x, dp=0):
        x = pd.Series(x).dropna()
        return f"{x.median():.{dp}f} ({x.quantile(.25):.{dp}f}-{x.quantile(.75):.{dp}f})"

    def text(desc, got, want):
        checks.append((section[0], desc, got, want))

    def cat_counts(df, cat, outcome="ssm_ge_40"):
        d = df.dropna(subset=[cat])
        g = d.groupby(cat, observed=False).agg(n=("ssm", "size"), e=(outcome, "sum"))
        return g.loc[list(d[cat].cat.categories)]

    def trend(df, cat, outcome="ssm_ge_40"):
        g = cat_counts(df, cat, outcome)
        return cochran_armitage(g["e"], g["n"])[1], cochran_armitage(g["e"][1:], g["n"][1:])[1]

    def logit30(df, col, unit, extra=()):
        cols = [col, "age", "male", "bmi"] + list(extra)
        d = df.dropna(subset=["ssm_ge_30"] + cols).copy()
        d[col] = d[col] / unit
        m = logit_mle(d[cols].astype(float), d["ssm_ge_30"].values)
        t = m["table"].set_index("term").loc[col]
        return dict(or_=t.OR, low=t.ci_low, high=t.ci_high, p=t.p, n=m["n"], events=m["events"])

    # ----------------------------------------------------------------------
    head("Study population")
    count("patients in the ELASTIBD baseline cohort file", prep.n_cohort(), 358)
    count("patients with an attempted spleen examination", len(pat_att), 310)
    count("patients with a valid measurement (analysed)", len(pat), 271)
    count("reference participants examined", len(ref_att), 64)
    count("reference participants analysed", len(ref), 57)
    no_value = ref_att["ssm"].isna()
    count("reference: no valid value", (no_value & (ref_att["ssm_success"] != 1)).sum(), 6)
    count("reference: value not recorded", (no_value & (ref_att["ssm_success"] == 1)).sum(), 1)
    count("Crohn's disease", (pat["uc"] == 0).sum(), 173)
    count("ulcerative colitis", (pat["uc"] == 1).sum(), 98)
    count("men", pat["male"].sum(), 164)
    num("men (%)", 100 * pat["male"].mean(), 60.5, 1)
    count("median age (years)", pat["age"].median(), 38)
    count("patients with a medication history", pat["thiopurine"].notna().sum(), 266)
    count("thiopurine-exposed", pat["thiopurine"].sum(), 194)
    num("thiopurine-exposed (%)", 100 * pat["thiopurine"].mean(), 72.9, 1)
    count("azathioprine-exposed", (pat["aza"] == 1).sum(), 193)
    count("mercaptopurine-exposed", (pat["mercaptopurine"] == 1).sum(), 3)
    count("mercaptopurine only", ((pat["mercaptopurine"] == 1) & (pat["aza"] == 0)).sum(), 1)
    count("anti-TNF exposed", (pat["antitnf"] == 1).sum(), 183)
    num("anti-TNF exposed (%)", 100 * pat["antitnf"].mean(), 68.5, 1)
    count("azathioprine plus anti-TNF", (pat["aza_combo"] == "AZA + anti-TNF").sum(), 150)
    count("maintenance dose 100 mg/day", (aza_users["aza_mg_day"] == 100).sum(), 121)
    count("maintenance dose <= 75 mg/day", (aza_users["aza_mg_day"] <= 75).sum(), 41)
    count("maintenance dose > 100 mg/day", (aza_users["aza_mg_day"] > 100).sum(), 30)
    text("weight-adjusted dose, median (IQR)", med_iqr(aza_users["aza_mg_kg"], 2), "1.39 (1.16-1.70)")
    text("thiopurine duration (months), median (IQR)", med_iqr(thio_users["thio_months"]),
         "50 (15-90)")
    text("cumulative thiopurine dose (g), median (IQR)", med_iqr(thio_users["thio_cumulative_g"]),
         "129 (45-282)")
    count("thiopurine users with a cumulative dose", thio_users["thio_cumulative_g"].notna().sum(),
          192)
    count("reliability subset (IQR/median <= 30%)", (pat["reliable"] == 1).sum(), 186)
    count("patients with a recorded IQR/median", pat["ssm_iqr_median"].notna().sum(), 256)
    num("reliability subset (% of recorded)", 100 * (pat["reliable"] == 1).sum()
        / pat["ssm_iqr_median"].notna().sum(), 72.7, 1)
    exposed, unexposed = pat[pat["thiopurine"] == 1], pat[pat["thiopurine"] == 0]
    num("median LSM, exposed (kPa)", exposed["lsm"].median(), 4.7, 1)
    num("median LSM, unexposed (kPa)", unexposed["lsm"].median(), 5.1, 1)
    pval("LSM, exposed vs unexposed", stats.mannwhitneyu(exposed["lsm"], unexposed["lsm"]).pvalue,
         "0.041")
    pval("ulcerative colitis, exposed vs unexposed",
         stats.fisher_exact(pd.crosstab(pat.dropna(subset=["thiopurine"])["thiopurine"],
                                        pat.dropna(subset=["thiopurine"])["uc"])).pvalue, "< 0.001")

    # ----------------------------------------------------------------------
    head("Distribution and reference values")
    text("SSM, patients, median (IQR)", med_iqr(pat["ssm"], 1), "19.8 (14.6-25.4)")
    text("SSM, reference, median (IQR)", med_iqr(ref["ssm"], 1), "19.3 (11.5-22.5)")
    num("reference 90th percentile", ref["ssm"].quantile(0.90), 27.9, 1)
    num("reference 95th percentile", ref["ssm"].quantile(0.95), 29.6, 1)
    num("reference maximum", ref["ssm"].max(), 35.5, 1)
    num("reference mean", ref["ssm"].mean(), 18.3, 1)
    num("geometric mean, patients", np.exp(np.log(pat["ssm"]).mean()), 20.3, 1)
    num("geometric mean, reference", np.exp(np.log(ref["ssm"]).mean()), 16.7, 1)
    count("SSM >= 30 kPa, patients", pat["ssm_ge_30"].sum(), 49)
    count("SSM >= 30 kPa, reference", ref["ssm_ge_30"].sum(), 2)
    count("SSM >= 40 kPa, patients", pat["ssm_ge_40"].sum(), 20)
    num("SSM >= 40 kPa, patients (%)", 100 * pat["ssm_ge_40"].mean(), 7.4, 1)
    count("SSM >= 40 kPa, reference", ref["ssm_ge_40"].sum(), 0)
    count("SSM >= 50 kPa, patients", pat["ssm_ge_50"].sum(), 11)
    num("SSM >= 21 kPa, patients (%)", 100 * pat["ssm_ge_21"].mean(), 43.9, 1)
    num("SSM >= 21 kPa, reference (%)", 100 * ref["ssm_ge_21"].mean(), 42.1, 1)
    count("patients above 75 kPa", (pat["ssm"] > 75).sum(), 4)
    num("upper 95% limit for 0 of 57 (%)", 100 * wilson_ci(0, len(ref))[1], 6.3, 1)
    both = pd.concat([pat.assign(patient=1.0), ref.assign(patient=0.0)], ignore_index=True)
    g = gmr(both, "patient", covars=("age", "male", "bmi"))
    est("GMR patients vs reference (age, sex, BMI)", g | {"or": g["gmr"]}, "1.19 (1.06-1.34)")
    pval("GMR patients vs reference: p", g["p"], "0.004")
    g = gmr(both, "patient", covars=("age", "male", "bmi", "lsm"))
    est("GMR patients vs reference (+ LSM)", g | {"or": g["gmr"]}, "1.19 (1.06-1.35)")
    for r, want in zip(_quantreg(both, "patient"), ["0.030", "0.13", "0.21", "0.081"]):
        pval(f"quantile regression vs reference, {int(r['quantile'] * 100)}th percentile: p",
             r["p"], want)
    r = _quantreg(both, "patient")[-1]
    est("quantile regression vs reference, 90th percentile", r | {"or": r["ratio"]},
        "1.23 (0.97-1.56)")

    # ----------------------------------------------------------------------
    head("Factors associated with SSM")
    num("median SSM, men", pat.loc[pat["male"] == 1, "ssm"].median(), 20.9, 1)
    num("median SSM, women", pat.loc[pat["male"] == 0, "ssm"].median(), 16.8, 1)
    pval("men vs women", stats.mannwhitneyu(pat.loc[pat["male"] == 1, "ssm"],
                                            pat.loc[pat["male"] == 0, "ssm"]).pvalue, "0.002")
    for col, want, dp in [("bmi", 0.35, 2), ("lsm", 0.12, 2), ("plt", -0.12, 2)]:
        d = pat[[col, "ssm"]].dropna()
        num(f"Spearman rho, SSM vs {col}", stats.spearmanr(d[col], d["ssm"])[0], want, dp)
    d = pat[["plt", "ssm"]].dropna()
    pval("Spearman, SSM vs platelets: p", stats.spearmanr(d["plt"], d["ssm"])[1], "0.046")
    d = pat[(pat["uc"] == 0)][["cdai", "ssm"]].dropna()
    num("Spearman rho, SSM vs CDAI (CD)", stats.spearmanr(d["cdai"], d["ssm"])[0], -0.22)
    pval("Spearman, SSM vs CDAI (CD): p", stats.spearmanr(d["cdai"], d["ssm"])[1], "0.006")
    d0 = pat.assign(plt10=pat["plt"] / 10.0)
    fits = {}
    for name, covars in [("A", MODEL_A), ("B", MODEL_B)]:
        d = d0.dropna(subset=["log_ssm", "ssm_ge_30", "ssm_ge_40"] + covars)
        lin = sm.OLS(d["log_ssm"].astype(float), sm.add_constant(d[covars].astype(float))).fit(
            cov_type="HC3")
        f40 = firth_profile(d[covars].astype(float), d["ssm_ge_40"].values, terms=covars)
        fits[name] = (d, lin, f40)
        count(f"Model {name}: n", len(d), {"A": 219, "B": 218}[name])
        count(f"Model {name}: events >= 40 kPa", d["ssm_ge_40"].sum(), 14)
    d, lin, f40 = fits["A"]
    for v, want, dp in [("male", "1.14 (1.01-1.29)", 2), ("bmi", "1.023 (1.009-1.038)", 3)]:
        ci = lin.conf_int().loc[v]
        est(f"Model A, GMR for {v}", dict(or_=np.exp(lin.params[v]), lo=np.exp(ci.iloc[0]),
                                           hi=np.exp(ci.iloc[1])), want, dp, ("or_", "lo", "hi"))
    d, lin, f40 = fits["B"]
    t = f40["terms"]
    est("Model B, OR for platelets per 10 x10^3/uL (>= 40 kPa)", t["plt10"], "0.85 (0.76-0.94)",
        keys=("or", "pl_low", "pl_high"))
    pval("Model B, platelets: p", t["plt10"]["plrt_p"], "< 0.001")
    est("Model B, OR per corticosteroid course (>= 40 kPa)", t["steroid_courses"],
        "1.90 (1.05-3.42)", keys=("or", "pl_low", "pl_high"))

    # ----------------------------------------------------------------------
    head("Thiopurine exposure and the distribution (Table 2)")
    num("median SSM, exposed", exposed["ssm"].median(), 19.8, 1)
    num("median SSM, unexposed", unexposed["ssm"].median(), 20.1, 1)
    pval("SSM, exposed vs unexposed (crude)",
         stats.mannwhitneyu(exposed["ssm"], unexposed["ssm"]).pvalue, "0.98")
    g = gmr(pat, "thiopurine")
    est("adjusted GMR, thiopurine exposure", g | {"or": g["gmr"]}, "0.99 (0.87-1.13)")
    count("adjusted GMR, thiopurine exposure: n", g["n"], 224)
    g = gmr(pat, "thiopurine", covars=("age", "male", "bmi", "lsm"))
    est("adjusted GMR without alcohol use", g | {"or": g["gmr"]}, "1.02 (0.91-1.14)")
    count("adjusted GMR without alcohol use: n", g["n"], 266)
    count("patients with a medication history but no recorded alcohol use",
          (pat["thiopurine"].notna() & pat["alcohol_any"].isna()).sum(), 42)
    d = pat.dropna(subset=["aza_combo"])
    pval("treatment regimen (Kruskal-Wallis)", stats.kruskal(
        *[d.loc[d["aza_combo"] == c, "ssm"] for c in d["aza_combo"].unique()]).pvalue, "0.90")
    d = aza_users[["aza_mg_day", "ssm"]].dropna()
    num("Spearman rho, daily dose vs SSM", stats.spearmanr(d["aza_mg_day"], d["ssm"])[0], 0.22)
    pval("Spearman, daily dose vs SSM: p", stats.spearmanr(d["aza_mg_day"], d["ssm"])[1], "0.003")
    groups = {k: aza_users[aza_users["aza_dose_cat"] == k] for k in ["<= 75 mg", "100 mg", "> 100 mg"]}
    for k, want in [("> 100 mg", 24.6), ("100 mg", 19.9), ("<= 75 mg", 16.5)]:
        num(f"median SSM, daily dose {k}", groups[k]["ssm"].median(), want, 1)
    pval("daily dose groups (Kruskal-Wallis)",
         stats.kruskal(*[g["ssm"] for g in groups.values()]).pvalue, "0.012")
    for k, want in [("<= 75 mg", 65), ("100 mg", 71), ("> 100 mg", 78)]:
        count(f"mean body weight, daily dose {k}", round(groups[k]["weight"].mean()), want)
    d = aza_users[["aza_mg_day", "weight"]].dropna()
    num("Spearman rho, daily dose vs body weight", stats.spearmanr(d["aza_mg_day"], d["weight"])[0],
        0.28)
    d = aza_users[["aza_mg_kg", "ssm"]].dropna()
    num("Spearman rho, weight-adjusted dose vs SSM", stats.spearmanr(d["aza_mg_kg"], d["ssm"])[0],
        -0.03)
    for desc, df, col, unit, want in [
            ("adjusted GMR per 25 mg/day", aza_users, "aza_mg_day", 25, "1.05 (0.98-1.13)"),
            ("adjusted GMR per 12 months of thiopurine", thio_users, "thio_months", 12,
             "1.00 (0.98-1.02)"),
            ("adjusted GMR per 100 g of thiopurine", thio_users, "thio_cumulative_g", 100,
             "1.00 (0.95-1.05)")]:
        g = gmr(df, col, unit)
        est(desc, g | {"or": g["gmr"]}, want)
    for df, col, ref_level, expected, crude in [
            (pat, "aza_combo", "Neither", {"Anti-TNF without AZA": "1.00 (0.82-1.22)",
                                           "AZA monotherapy": "0.99 (0.81-1.23)",
                                           "AZA + anti-TNF": "1.05 (0.87-1.26)"}, None),
            (aza_users, "aza_dose_cat", "100 mg", {"<= 75 mg": "1.04 (0.88-1.23)",
                                                   "> 100 mg": "1.20 (0.93-1.54)"}, None),
            (aza_users, "aza_mg_kg_cat", "< 1.5 mg/kg", {"1.5-1.99 mg/kg": "1.06 (0.89-1.27)",
                                                         ">= 2.0 mg/kg": "0.93 (0.72-1.19)"},
             "0.38"),
            (thio_users, "thio_months_cat", "<= 12 months", {"13-48 months": "1.01 (0.85-1.22)",
                                                             "> 48 months": "1.01 (0.86-1.19)"},
             "0.11"),
            (thio_users, "thio_cumulative_cat", "<= 100 g", {"101-300 g": "0.92 (0.79-1.08)",
                                                             "> 300 g": "0.95 (0.77-1.15)"},
             "0.71")]:
        res = gmr_categorical(df, col, ref_level)
        for lvl, want in expected.items():
            est(f"Table 2, {col}: {lvl} vs {ref_level}", res[lvl] | {"or": res[lvl]["gmr"]}, want)
        if crude:
            d = df.dropna(subset=[col])
            pval(f"Table 2, {col}: crude p", stats.kruskal(
                *[d.loc[d[col] == c, "ssm"] for c in d[col].cat.categories]).pvalue, crude)
    for col in ["thio_months_cat", "thio_cumulative_cat"]:
        for lvl, n in thio_users[col].value_counts().items():
            count(f"Table 2, {col}: n in {lvl}", n, {"<= 12 months": 44, "13-48 months": 50,
                                                      "> 48 months": 98, "<= 100 g": 73,
                                                      "101-300 g": 75, "> 300 g": 44}[lvl])

    # ----------------------------------------------------------------------
    head("Markedly elevated SSM (Table 3)")
    for cat, n_exp, e_exp, t40, t30 in [
            ("thio_cumulative_cat4", [71, 73, 75, 44], [4, 3, 5, 8], ("0.027", "0.011"), ("0.36", "0.73")),
            ("thio_months_cat4", [71, 44, 50, 98], [4, 0, 5, 11], ("0.066", "0.037"), ("0.13", "0.18")),
            ("aza_dose_cat4", [72, 41, 121, 30], [4, 2, 8, 6], ("0.074", "0.034"), ("0.030", "0.014"))]:
        g = cat_counts(pat, cat)
        for (lab, row), n, e in zip(g.iterrows(), n_exp, e_exp):
            count(f"Table 3A, {cat} {lab}: n", row["n"], n)
            count(f"Table 3A, {cat} {lab}: events", row["e"], e)
        p_all, p_users = trend(pat, cat)
        pval(f"Table 3A, {cat}: trend p (all)", p_all, t40[0])
        pval(f"Table 3A, {cat}: trend p (users)", p_users, t40[1])
        p_all, p_users = trend(pat, cat, "ssm_ge_30")
        pval(f"Table 3A, {cat}: trend p for >= 30 kPa (all)", p_all, t30[0])
        pval(f"Table 3A, {cat}: trend p for >= 30 kPa (users)", p_users, t30[1])
    table3b = [
        (pat, "thio_cumulative_g", 100, M1, "263/20", "1.39 (1.13-1.72)", "0.002"),
        (pat, "thio_cumulative_g", 100, M2, "262/20", "1.34 (1.08-1.68)", "0.008"),
        (pat, "thio_cumulative_g", 100, M3, "256/18", "1.25 (0.99-1.57)", "0.062"),
        (pat, "thio_months", 12, M1, "263/20", "1.12 (1.03-1.23)", "0.013"),
        (pat, "thio_months", 12, M2, "262/20", "1.10 (1.00-1.21)", "0.044"),
        (pat, "thio_months", 12, M3, "256/18", "1.08 (0.98-1.19)", "0.13"),
        (thio_users, "thio_cumulative_g", 100, M1, "192/16", "1.49 (1.16-1.94)", "0.002"),
        (thio_users, "thio_cumulative_g", 100, M2, "192/16", "1.44 (1.12-1.89)", "0.005"),
        (thio_users, "thio_cumulative_g", 100, M3, "188/14", "1.34 (1.02-1.77)", "0.034"),
        (thio_users, "thio_months", 12, M1, "192/16", "1.14 (1.03-1.28)", "0.016"),
        (thio_users, "thio_months", 12, M2, "192/16", "1.12 (1.00-1.27)", "0.041"),
        (thio_users, "thio_months", 12, M3, "188/14", "1.10 (0.98-1.24)", "0.089"),
    ]
    for df, col, unit, extra, nev, want, p in table3b:
        r = firth_or(df, col, unit, extra)
        label = (f"Table 3B, {'users' if df is thio_users else 'all'}, {col}, "
                 f"Model {1 + len(extra) if len(extra) < 2 else 3}")
        text(f"{label}: n / events", f"{r['n']}/{r['events']}", nev)
        est(f"{label}: OR (95% CI)", r, want)
        pval(f"{label}: p", r["p"], p)
    same = pat.dropna(subset=list(M3) + ["thio_cumulative_g"])
    est("Model 1 in the 256 patients of Model 3", firth_or(same, "thio_cumulative_g", 100),
        "1.31 (1.05-1.63)")
    drop3 = pat.dropna(subset=["thio_cumulative_g"])
    drop3 = drop3[drop3[list(M3)].isna().any(axis=1)]
    count("patients lost in Model 3", len(drop3), 7)
    count("patients lost in Model 3 with SSM >= 40 kPa", drop3["ssm_ge_40"].sum(), 2)
    r = logit30(pat, "thio_cumulative_g", 100)
    est("OR per 100 g for SSM >= 30 kPa (conventional logistic)", r, "1.12 (0.94-1.32)",
        keys=("or_", "low", "high"))

    # ----------------------------------------------------------------------
    head("Shape of the association (Supplementary Table S16)")
    for t, want in zip([21, 25, 30, 35, 40, 45, 50],
                       ["0.97", "1.10", "1.11", "1.14", "1.39", "1.35", "1.47"]):
        r = firth_or(pat.assign(_y=(pat["ssm"] >= t).astype(float)), "thio_cumulative_g", 100,
                     outcome="_y", ci=False)
        num(f"threshold scan: OR per 100 g at >= {t} kPa", r["or"], want)
    q = _quantreg(pat, "thio_cumulative_g", unit=100)
    est("quantile regression, 90th percentile, per 100 g", q[3] | {"or": q[3]["ratio"]},
        "1.11 (1.04-1.18)")
    pval("quantile regression, 90th percentile: p", q[3]["p"], "0.001")
    pval("quantile regression, median: p", q[1]["p"], "0.86")
    d = pat.dropna(subset=["ssm_ge_40", "thio_cumulative_g", "age", "male", "bmi"]).copy()
    x = d["thio_cumulative_g"].to_numpy(float)
    k1, k2, k3 = np.percentile(x[x > 0], [10, 50, 90])
    text("spline knots (g)", f"{k1:.0f}, {k2:.0f}, {k3:.0f}", "9, 129, 420")
    pos = lambda u: np.clip(u, 0, None) ** 3
    d["c100"] = x / 100
    d["nl"] = (pos(x - k1) - pos(x - k2) * (k3 - k1) / (k3 - k2)
               + pos(x - k3) * (k2 - k1) / (k3 - k2)) / (k3 - k1) ** 2 / 100
    r = firth_profile(d[["c100", "nl", "age", "male", "bmi"]].astype(float), d["ssm_ge_40"].values,
                      terms=["nl"], ci=False)
    pval("restricted cubic spline: test for non-linearity", r["terms"]["nl"]["plrt_p"], "0.54")
    d["gt300"] = (x > 300).astype(float)
    r = firth_profile(d[["gt300", "age", "male", "bmi"]].astype(float), d["ssm_ge_40"].values,
                      terms=["gt300"])["terms"]["gt300"]
    est("cumulative dose > 300 g vs <= 300 g or none", r, "3.30 (1.23-8.44)",
        keys=("or", "pl_low", "pl_high"))

    # ----------------------------------------------------------------------
    head("Dose, duration and correlations (Supplementary Tables S10, S11 and S15)")
    r = firth_or(aza_users, "aza_mg_day", 25)
    est("AZA users, OR per 25 mg/day (Model 1)", r, "1.79 (1.13-2.88)")
    d = aza_users.dropna(subset=["ssm_ge_40", "aza_mg_day", "aza_months", "age", "male", "bmi"]).copy()
    d["aza_mg_day"] /= 25
    d["aza_months"] /= 12
    r = firth_profile(d[["aza_mg_day", "aza_months", "age", "male", "bmi"]].astype(float),
                      d["ssm_ge_40"].values, terms=["aza_mg_day", "aza_months"])["terms"]
    est("joint model, AZA users: OR per 25 mg/day", r["aza_mg_day"], "1.62 (1.02-2.61)",
        keys=("or", "pl_low", "pl_high"))
    est("joint model, AZA users: OR per 12 months", r["aza_months"], "1.13 (1.01-1.27)",
        keys=("or", "pl_low", "pl_high"))
    X = sm.add_constant(d[["aza_mg_day", "aza_months", "age", "male", "bmi"]].astype(float))
    vif = max(1 / (1 - sm.OLS(X[c], X.drop(columns=c)).fit().rsquared)
              for c in ["aza_mg_day", "aza_months"])
    text("joint model: variance inflation factors < 1.2", str(vif < 1.2), "True")
    for a, b, want in [("aza_mg_day", "aza_months", 0.28), ("aza_cumulative_g", "aza_months", 0.96),
                       ("aza_cumulative_g", "aza_mg_day", 0.49),
                       ("aza_cumulative_g", "disease_duration_years", 0.56)]:
        d = aza_users[[a, b]].dropna()
        num(f"Spearman rho among AZA users, {a} vs {b}", stats.spearmanr(d[a], d[b])[0], want)
    for a, b, want in [("aza_mg_day", "aza_months", 0.74),
                       ("aza_cumulative_g", "disease_duration_years", 0.43)]:
        d = pat[[a, b]].dropna()
        num(f"Spearman rho, all patients, {a} vs {b}", stats.spearmanr(d[a], d[b])[0], want)
    count("AZA users with a recorded daily dose", aza_users["aza_mg_day"].notna().sum(), 192)
    count("AZA users with a recorded duration", aza_users["aza_months"].notna().sum(), 191)

    # ----------------------------------------------------------------------
    head("Robustness (Supplementary Tables S12 and S13)")
    sets = dict(analysis_sets(pat)[0])
    for label, want, p in [
            ("Reliability subset (IQR/median <= 30%)", "1.42 (1.11-1.81)", "0.005"),
            ("Excluding the measurement at the 100-kPa ceiling", "1.33 (1.06-1.66)", "0.015"),
            ("Reliability subset, excluding the ceiling measurement", "1.29 (0.99-1.70)", "0.063"),
            ("Excluding the two highest cumulative doses (864 and 870 g)", "1.26 (0.98-1.61)", "0.066"),
            ("LSM < 7.2 kPa only", "1.29 (1.00-1.66)", "0.051"),
            ("Crohn's disease only", "1.30 (1.02-1.66)", "0.032"),
            ("Ulcerative colitis only", "1.46 (0.88-2.45)", "0.13"),
            ("Men only", "1.39 (1.08-1.82)", "0.013"),
            ("Women only", "1.36 (0.97-1.92)", "0.073"),
            ("BMI < 30 kg/m2", "1.35 (1.08-1.71)", "0.010"),
            ("Non-drinkers only", "1.25 (0.91-1.69)", "0.16")]:
        r = firth_or(sets[label], "thio_cumulative_g", 100)
        est(f"{label}: OR per 100 g", r, want)
        pval(f"{label}: p", r["p"], p)
    count("reliability subset: events", firth_or(
        sets["Reliability subset (IQR/median <= 30%)"], "thio_cumulative_g", 100, ci=False)["events"], 13)
    count("ulcerative colitis: events", firth_or(
        sets["Ulcerative colitis only"], "thio_cumulative_g", 100, ci=False)["events"], 4)
    if REPEAT_LABEL in sets:
        est("excluding repeated examinations: OR per 100 g", firth_or(
            sets[REPEAT_LABEL], "thio_cumulative_g", 100), "1.48 (1.19-1.86)")
    else:
        skipped.append((section[0], "excluding repeated examinations: OR per 100 g",
                        "requires repeated_examinations.csv"))
    est("additional adjustment for LSM", firth_or(pat, "thio_cumulative_g", 100, ("lsm",)),
        "1.39 (1.13-1.71)")
    alc = pat[pat["alcohol_any"].notna()]
    r = firth_or(alc, "thio_cumulative_g", 100, ("alcohol_any",))
    est("additional adjustment for alcohol use", r, "1.31 (1.02-1.67)")
    count("additional adjustment for alcohol use: n", r["n"], 222)
    est("Model 1 in the 222 patients with recorded alcohol use",
        firth_or(alc, "thio_cumulative_g", 100), "1.31 (1.02-1.68)")
    est("azathioprine dose alone", firth_or(pat, "aza_cumulative_g", 100), "1.41 (1.14-1.74)")
    est("excluding the mercaptopurine-exposed patients",
        firth_or(pat[pat["mercaptopurine"] != 1], "thio_cumulative_g", 100), "1.40 (1.14-1.74)")
    est("mercaptopurine converted with the molar mass ratio (x 1.82)", firth_or(
        prep.add_thiopurine_equivalents(pat, factor=prep.MP_TO_AZA_MOLAR), "thio_cumulative_g", 100),
        "1.39 (1.13-1.73)")
    p_all, p_users = trend(pat, "aza_cumulative_cat4")
    pval("azathioprine dose alone: trend p (all)", p_all, "0.022")
    pval("azathioprine dose alone: trend p (users)", p_users, "0.009")
    d = pat.dropna(subset=["ssm_ge_40", "thio_cumulative_g", "uc", "age", "male", "bmi"]).copy()
    d["cum"] = d["thio_cumulative_g"] / 100
    d["cum_x_uc"] = d["cum"] * d["uc"]
    r = firth_profile(d[["cum", "uc", "cum_x_uc", "age", "male", "bmi"]].astype(float),
                      d["ssm_ge_40"].values, terms=["cum_x_uc"], ci=False)
    pval("interaction cumulative dose x disease type", r["terms"]["cum_x_uc"]["plrt_p"], "0.92")
    loo = leave_one_out(pat)
    num("leave-one-out: lowest OR", loo["_or"].min(), "1.33")
    num("leave-one-out: highest OR", loo["_or"].max(), "1.44")
    count("leave-one-out: fits with p < 0.05", (loo["_p"] < 0.05).sum(), 20)
    count("leave-one-out: fits with trend p < 0.05", (loo["_trend"] < 0.05).sum(), 12)
    num("leave-one-out, users: lowest OR", loo["_or_u"].min(), "1.39")
    num("leave-one-out, users: highest OR", loo["_or_u"].max(), "1.57")

    # ----------------------------------------------------------------------
    head("Specificity (Supplementary Table S9)")
    for col, unit, want in [("antitnf_months", 12, "0.99 (0.86-1.10)"),
                            ("disease_duration_years", 1, "1.05 (0.99-1.11)"),
                            ("steroid_courses", 1, "1.34 (0.82-2.03)"),
                            ("alcohol_any", 1, "1.77 (0.60-5.13)"),
                            ("resection", 1, "1.47 (0.51-3.81)")]:
        est(f"OR for {col}", firth_or(pat, col, unit), want)
    r = firth_or(pat.assign(plt10=pat["plt"] / 10), "plt10", 1)
    est("OR per 10 x10^3/uL platelets", r, "0.89 (0.82-0.96)")
    pval("platelets: p", r["p"], "< 0.001")

    # ----------------------------------------------------------------------
    head("Discordance and haematology")
    count("SSM >= 40 kPa with LSM < 10 kPa", (marked["lsm"] < 10).sum(), 20)
    count("SSM >= 40 kPa with LSM < 7.2 kPa", (marked["lsm"] < 7.2).sum(), 16)
    num("median LSM in marked cases", marked["lsm"].median(), 5.1, 1)
    num("minimum SSM/LSM ratio in marked cases", (marked["ssm"] / marked["lsm"]).min(), 6.2, 1)
    num("median SSM/LSM ratio in marked cases", (marked["ssm"] / marked["lsm"]).median(), 10.6, 1)
    count("patients with LSM >= 10 kPa", (pat["lsm"] >= 10).sum(), 6)
    count("LSM >= 10 kPa and SSM >= 40 kPa", ((pat["lsm"] >= 10) & (pat["ssm_ge_40"] == 1)).sum(), 0)
    count("neither", ((pat["lsm"] < 10) & (pat["ssm"] < 40)).sum(), 245)
    for lo, hi, want in [(0, 21, 274), (40, 50, 242), (50, np.inf, 224)]:
        count(f"median platelets, SSM {lo}-{hi} kPa",
              pat.loc[(pat["ssm"] >= lo) & (pat["ssm"] < hi), "plt"].median(), want)
    for lo, hi, want in [(0, 21, 0), (50, np.inf, 18)]:
        count(f"thrombocytopenia (%), SSM {lo}-{hi} kPa",
              round(100 * pat.loc[(pat["ssm"] >= lo) & (pat["ssm"] < hi), "thrombocytopenia"].mean()),
              want)
    low = pat[pat["thrombocytopenia"] == 1]
    count("thrombocytopenic patients", len(low), 7)
    count("thrombocytopenic and thiopurine-exposed", low["thiopurine"].sum(), 5)
    text("SSM range, thrombocytopenic exposed (kPa)",
         f"{low.loc[low['thiopurine'] == 1, 'ssm'].min():.0f}-"
         f"{low.loc[low['thiopurine'] == 1, 'ssm'].max():.0f}", "31-90")
    count("median SSM, thrombocytopenic exposed (kPa)",
          low.loc[low["thiopurine"] == 1, "ssm"].median(), 45)
    text("SSM, thrombocytopenic unexposed (kPa)",
         ", ".join(f"{v:.0f}" for v in sorted(low.loc[low["thiopurine"] == 0, "ssm"])), "23, 24")
    count("median platelets, patients", pat["plt"].median(), 270)
    count("median platelets, reference", ref["plt"].median(), 246)
    ge50 = pat[pat["ssm_ge_50"] == 1]
    count("SSM >= 50 kPa and thiopurine-exposed", ge50["thiopurine"].sum(), 10)
    pval("SSM >= 50 kPa vs below: thiopurine exposure (Fisher)", stats.fisher_exact(pd.crosstab(
        pat.dropna(subset=["thiopurine"])["ssm_ge_50"],
        pat.dropna(subset=["thiopurine"])["thiopurine"])).pvalue, "0.30")
    count("patients with abdominal imaging", pat["splenomegaly"].notna().sum(), 152)
    count("marked cases with imaging", marked["splenomegaly"].notna().sum(), 8)
    count("marked cases with splenomegaly", marked["splenomegaly"].sum(), 1)

    # ----------------------------------------------------------------------
    head("Patients with SSM >= 40 kPa (Table 4)")
    count("median SSM (kPa)", marked["ssm"].median(), 54)
    count("men", marked["male"].sum(), 14)
    count("medication history available", marked["thiopurine"].notna().sum(), 20)
    count("thiopurine-exposed", marked["thiopurine"].sum(), 16)
    count("azathioprine with anti-TNF", (marked["aza_combo"] == "AZA + anti-TNF").sum(), 14)
    exposed_marked = marked[marked["thio_user"] == 1]
    text("cumulative dose range among exposed (g)",
         f"{exposed_marked['thio_cumulative_g'].min():.0f}-"
         f"{exposed_marked['thio_cumulative_g'].max():.0f}", "54-870")
    count("cumulative dose > 300 g", (marked["thio_cumulative_g"] > 300).sum(), 8)
    count("mercaptopurine-exposed (azathioprine and thiopurine values coincide)",
          (marked["mercaptopurine"] == 1).sum(), 0)
    text("platelet counts < 150 x10^3/uL",
         ", ".join(f"{v:.0f}" for v in sorted(marked.loc[marked["plt"] < 150, "plt"])), "79, 85, 130")

    # ----------------------------------------------------------------------
    # Tables, row by row
    # ----------------------------------------------------------------------
    def norm(x):
        x = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
        for a, b in [("\u2013", "-"), ("\u2014", "-"), ("\u2212", "-")]:
            x = x.replace(a, b)
        return " ".join(x.split())

    def computed_rows(df, drop=()):
        cols = [c for c in df.columns if c not in drop]
        out = []
        for r in df[cols].itertuples(index=False):
            vals = tuple(v for v in (norm(c) for c in r[1:]) if v)
            out.append(vals)
        return out

    tables = {
        "Table 1": computed_rows(main_tables.table1(pat)),
        "Table 2": computed_rows(main_tables.table2a(pat)),
        "Table 3A": computed_rows(main_tables.table3a(pat)),
        "Table 3B": computed_rows(main_tables.table3b(pat), drop=("OR (Wald 95% CI)", "p (Wald)")),
        "S1": computed_rows(supp.s1(pat, ref)),
        "S2": computed_rows(supp.s2(pat)),
        "S3": computed_rows(supp.s3(pat, ref)),
        "S4": computed_rows(supp.s4(pat)),
        "S5": computed_rows(supp.s5(pat), drop=("n (thiopurine users)",)),
        "S6": computed_rows(supp.s6(pat)),
        "S7": computed_rows(supp.s7(pat)),
        "S8": computed_rows(supp.s8(pat)),
        "S9": computed_rows(supp.s9(pat)),
        "S10": computed_rows(supp.s10(pat)),
        "S11": computed_rows(supp.s11(pat)),
        "S13": computed_rows(tableS13(pat)),
        "S15": computed_rows(supp.s15(pat)),
        "S16": computed_rows(supp.s16(pat)),
    }
    tables["S14"] = tables["Table 1"]
    loo_rows = [tuple(norm(c) for c in r) for r in loo.iloc[:, 2:7].itertuples(index=False)]
    for name, rows in PUBLISHED.items():
        head(f"{name}, row by row")
        if name == "S12":
            for (label, want), got in zip(rows, loo_rows):
                text(f"{name}: {label}", " | ".join(got), " | ".join(want))
            continue
        available = set(tables[name])
        last_values = {r[-1] for r in tables[name] if r}
        for label, want in rows:
            if name == "S13" and label.startswith(REPEAT_LABEL[:40]) and REPEAT_LABEL not in sets:
                skipped.append((section[0], f"{name}: {label}", "requires repeated_examinations.csv"))
                continue
            if len(want) == 1:   # section rows that carry only a p-value
                ok = want[0] in last_values
            else:
                ok = want in available
            text(f"{name}: {label}", " | ".join(want) if ok else "not reproduced", " | ".join(want))

    # --- report -------------------------------------------------------------
    failed = 0
    width = min(max(len(c[1]) for c in checks), 78) + 2
    current = None
    for sec, desc, got, want in checks:
        if sec != current:
            print(f"\n{sec}")
            current = sec
        ok = got == want
        failed += not ok
        if len(got) <= 20 and len(want) <= 20:
            print(f"  {'PASS' if ok else 'FAIL'}  {desc[:78]:<{width}} computed {got:>20}   "
                  f"published {want:>20}")
        else:
            print(f"  {'PASS' if ok else 'FAIL'}  {desc[:78]}")
            if not ok:
                print(f"        computed:  {got}\n        published: {want}")
    for sec, desc, reason in skipped:
        print(f"  SKIP  {desc} ({reason})")
    print(f"\n{len(checks) - failed} of {len(checks)} checks passed"
          + (f"; {len(skipped)} skipped." if skipped else "."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

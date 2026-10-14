"""Supplementary Tables S1-S11, S15 and S16 of the ELASTIBD spleen-stiffness manuscript.

Supplementary Tables S12 (leave-one-out) and S13 (robustness) are produced by
``robustness.py``. Supplementary Table S14 (additional baseline characteristics)
is part of ``table1_characteristics.csv``.

All Firth models report profile penalised-likelihood 95 % confidence intervals
and penalised likelihood-ratio p-values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import prep
from analysis_tables import (MODELS, OUT, firth_or, fisher, fmt_p, gmr, mannwhitney,
                             med_iqr, n_pct)
from firth import cochran_armitage, firth_profile, logit_mle, wilson_ci


# ---------------------------------------------------------------------------
# S1 - patients versus the reference sample
# ---------------------------------------------------------------------------
def s1(pat: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def cont(label, col, d=1):
        rows.append([label, med_iqr(pat[col], d), med_iqr(ref[col], d),
                     fmt_p(mannwhitney(pat[col], ref[col]))])

    def binr(label, col):
        rows.append([label, n_pct(pat[col]), n_pct(ref[col]),
                     fmt_p(fisher(pat[col], ref[col]))])

    cont("Age (years)", "age", 0)
    binr("Male sex", "male")
    cont("BMI (kg/m2)", "bmi")
    cont("Waist circumference (cm)", "waist", 0)
    binr("Hypertension", "hypertension")
    binr("Diabetes mellitus", "diabetes")
    for lab, col in [("AST (U/L)", "ast"), ("ALT (U/L)", "alt"), ("ALP (U/L)", "alp"),
                     ("GGT (U/L)", "ggt"), ("Platelets (x10^3/uL)", "plt")]:
        cont(lab, col, 0)
    binr("Platelets < 150 x10^3/uL", "thrombocytopenia")
    cont("LSM (kPa)", "lsm")
    binr("LSM >= 7.2 kPa", "lsm_ge_72")
    binr("LSM >= 10 kPa", "lsm_ge_10")
    cont("CAP (dB/m)", "cap", 0)
    binr("Steatosis (CAP >= 248 dB/m)", "steatosis")
    binr("XL probe used for LSM", "xl_probe")
    cont("SSM (kPa)", "ssm")
    rows.append(["SSM (kPa), mean +/- SD",
                 f"{pat['ssm'].mean():.1f} +/- {pat['ssm'].std():.1f}",
                 f"{ref['ssm'].mean():.1f} +/- {ref['ssm'].std():.1f}",
                 fmt_p(stats.ttest_ind(np.log(pat["ssm"]), np.log(ref["ssm"]), equal_var=False).pvalue)])
    cont("SSM IQR/median (%)", "ssm_iqr_median", 0)
    for lab, col in [("SSM >= 21 kPa", "ssm_ge_21"), ("SSM >= 30 kPa", "ssm_ge_30"),
                     ("SSM >= 40 kPa", "ssm_ge_40"), ("SSM >= 50 kPa", "ssm_ge_50")]:
        binr(lab, col)
    cont("SSM/LSM ratio", "ssm_lsm_ratio")

    both = pd.concat([pat.assign(patient=1.0), ref.assign(patient=0.0)], ignore_index=True)
    adj = gmr(both, "patient", covars=("age", "male", "bmi"))
    adj_lsm = gmr(both, "patient", covars=("age", "male", "bmi", "lsm"))
    lo, hi = wilson_ci(0, len(ref))
    footer = [
        ["Adjusted GMR (age, sex, BMI)",
         f"{adj['gmr']:.2f} ({adj['low']:.2f}-{adj['high']:.2f})", f"p = {fmt_p(adj['p'])}", ""],
        ["Adjusted GMR (+ LSM)",
         f"{adj_lsm['gmr']:.2f} ({adj_lsm['low']:.2f}-{adj_lsm['high']:.2f})", "", ""],
        ["Geometric mean SSM (kPa)", f"{np.exp(np.log(pat['ssm']).mean()):.1f}",
         f"{np.exp(np.log(ref['ssm']).mean()):.1f}", ""],
        ["Reference distribution: p90 / p95 / maximum", "",
         f"{ref['ssm'].quantile(.90):.1f} / {ref['ssm'].quantile(.95):.1f} / {ref['ssm'].max():.1f}", ""],
        [f"Upper 95% confidence limit for 0 of {len(ref)} >= 40 kPa", "", f"{100 * hi:.1f}%", ""],
    ]
    return pd.DataFrame(rows + footer, columns=[
        "Variable", f"Patients with IBD (n = {len(pat)})",
        f"Reference sample (n = {len(ref)})", "p"])


# ---------------------------------------------------------------------------
# S2 - univariable associations with SSM
# ---------------------------------------------------------------------------
CONTINUOUS_S2 = [
    ("Age (years)", "age"), ("BMI (kg/m2)", "bmi"), ("Waist circumference (cm)", "waist"),
    ("Alcohol (g/week)", "alcohol_g_week"), ("Smoking (pack-years)", "pack_years"),
    ("Age at diagnosis (years)", "age_at_diagnosis"),
    ("Disease duration (months)", "disease_duration_months"),
    ("CDAI (CD only)", "cdai"), ("SCCAI (UC only)", "sccai"), ("Mayo score (UC only)", "mayo"),
    ("Mean CRP, past 12 months (mg/L)", "crp_mean"), ("AST (U/L)", "ast"), ("ALT (U/L)", "alt"),
    ("ALP (U/L)", "alp"), ("GGT (U/L)", "ggt"), ("Total bilirubin (mg/dL)", "bilirubin"),
    ("Albumin (g/dL)", "albumin"), ("Haemoglobin (g/dL)", "hgb"),
    ("White cell count (x10^3/uL)", "wbc"), ("Platelets (x10^3/uL)", "plt"),
    ("FIB-4", "fib4"), ("APRI", "apri"), ("LSM (kPa)", "lsm"), ("CAP (dB/m)", "cap"),
    ("Systemic corticosteroid courses (n)", "steroid_courses"),
    ("Anti-TNF duration (months)", "antitnf_months"),
]
BINARY_S2 = [
    ("Male sex", "male"), ("Any alcohol use", "alcohol_any"), ("Hypertension", "hypertension"),
    ("Diabetes mellitus", "diabetes"), ("Ulcerative colitis", "uc"),
    ("Clinical remission", "remission"), ("Documented flare in past 12 months", "flare_any"),
    ("Extraintestinal manifestations", "eim"), ("Prior bowel resection", "resection"),
    ("Platelets < 150 x10^3/uL", "thrombocytopenia"), ("LSM >= 7.2 kPa", "lsm_ge_72"),
    ("Steatosis (CAP >= 248 dB/m)", "steatosis"), ("XL probe used for LSM", "xl_probe"),
    ("Splenomegaly on imaging", "splenomegaly"), ("Hepatomegaly on imaging", "hepatomegaly"),
    ("Thiopurine (AZA or mercaptopurine)", "thiopurine"), ("Azathioprine", "aza"), ("Anti-TNF", "antitnf"),
    ("Methotrexate", "methotrexate"), ("Vedolizumab or ustekinumab", "vedo_uste"),
    ("Budesonide", "budesonide"), ("Oral mesalazine", "mesalazine"),
    ("Isoniazid prophylaxis", "isoniazid"),
]


def s2(pat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in BINARY_S2:
        d = pat.dropna(subset=[col])
        p = mannwhitney(d.loc[d[col] == 1, "ssm"], d.loc[d[col] == 0, "ssm"])
        rows.append([label, len(d), "", "", "", fmt_p(p)])
        for val, name in [(0.0, "No"), (1.0, "Yes")]:
            g = d[d[col] == val]
            rows.append([f"  {name}", len(g), med_iqr(g["ssm"]),
                         f"{g['ssm'].mean():.1f} +/- {g['ssm'].std():.1f}",
                         f"{100 * g['ssm_ge_30'].mean():.1f}%", ""])
    for label, col in CONTINUOUS_S2:
        d = pat.dropna(subset=[col, "ssm"])
        if col == "cdai":
            d = d[d["uc"] == 0]
        if col in ("sccai", "mayo"):
            d = d[d["uc"] == 1]
        rho, p = stats.spearmanr(d[col], d["ssm"])
        rows.append([label, len(d), f"rho = {rho:+.3f}", "", "", fmt_p(p)])
    # characteristics with three levels (Kruskal-Wallis test)
    for label, col, levels, subset in [
            ("Smoking status", "smoking", [(0.0, "Never"), (1.0, "Current"), (2.0, "Former")], None),
            ("CD phenotype (Montreal B), Crohn's disease only", "cd_phenotype",
             [(0.0, "Inflammatory"), (1.0, "Penetrating"), (2.0, "Stricturing")], 0.0)]:
        d = pat.dropna(subset=[col])
        if subset is not None:
            d = d[d["uc"] == subset]
        groups = [d.loc[d[col] == v, "ssm"] for v, _ in levels]
        rows.append([label, len(d), "", "", "", fmt_p(stats.kruskal(*groups).pvalue)])
        for v, name in levels:
            g = d[d[col] == v]
            rows.append([f"  {name}", len(g), med_iqr(g["ssm"]),
                         f"{g['ssm'].mean():.1f} +/- {g['ssm'].std():.1f}",
                         f"{100 * g['ssm_ge_30'].mean():.1f}%", ""])
    return pd.DataFrame(rows, columns=[
        "Variable", "n", "SSM median (IQR) or Spearman rho", "SSM mean +/- SD",
        "SSM >= 30 kPa", "p"])


# ---------------------------------------------------------------------------
# S3 / S6 - quantile regression
# ---------------------------------------------------------------------------
QUANTILES = [0.25, 0.50, 0.75, 0.90]


def _quantreg(df, exposure, covars=("age", "male", "bmi"), unit=1.0):
    d = df.dropna(subset=["log_ssm", exposure] + list(covars)).copy()
    d = d.rename(columns={exposure: "expo"})
    d["expo"] = d["expo"] / unit
    formula = "log_ssm ~ expo + " + " + ".join(covars)
    out = []
    for q in QUANTILES:
        res = smf.quantreg(formula, d).fit(q=q, max_iter=5000)
        ci = res.conf_int().loc["expo"]
        out.append(dict(quantile=q, ratio=np.exp(res.params["expo"]),
                        low=np.exp(ci.iloc[0]), high=np.exp(ci.iloc[1]),
                        p=res.pvalues["expo"], n=int(res.nobs)))
    return out


def s3(pat, ref) -> pd.DataFrame:
    both = pd.concat([pat.assign(patient=1.0), ref.assign(patient=0.0)], ignore_index=True)
    rows = [[f"{int(r['quantile'] * 100)}th percentile",
             f"{r['ratio']:.2f} ({r['low']:.2f}-{r['high']:.2f})", fmt_p(r["p"])]
            for r in _quantreg(both, "patient")]
    return pd.DataFrame(rows, columns=[
        "SSM quantile", "Ratio of quantile, patients vs reference sample (95% CI)", "p"])


def s6(pat) -> pd.DataFrame:
    users = pat[pat["aza_user"] == 1]
    a = _quantreg(pat, "thiopurine")
    b = _quantreg(users, "aza_mg_day", unit=25)
    rows = [[f"{int(x['quantile'] * 100)}th percentile",
             f"{x['ratio']:.2f} ({x['low']:.2f}-{x['high']:.2f})", fmt_p(x["p"]),
             f"{y['ratio']:.3f} ({y['low']:.3f}-{y['high']:.3f})", fmt_p(y["p"])]
            for x, y in zip(a, b)]
    return pd.DataFrame(rows, columns=[
        "SSM quantile", "Thiopurine exposure vs none: ratio (95% CI)", "p",
        "AZA daily dose per 25 mg (users): ratio (95% CI)", "p "])


# ---------------------------------------------------------------------------
# S4 - multivariable models
# ---------------------------------------------------------------------------
MODEL_A = ["age", "male", "bmi", "alcohol_any", "uc", "disease_duration_years",
           "thiopurine", "antitnf", "steroid_courses"]
MODEL_B = MODEL_A + ["lsm", "plt10", "steatosis"]
LABELS = {"age": "Age (per year)", "male": "Male sex", "bmi": "BMI (per kg/m2)",
          "alcohol_any": "Any alcohol use", "uc": "Ulcerative colitis (vs Crohn's disease)",
          "disease_duration_years": "Disease duration (per year)",
          "thiopurine": "Thiopurine exposure", "antitnf": "Anti-TNF exposure",
          "steroid_courses": "Corticosteroid courses (per course)", "lsm": "LSM (per kPa)",
          "plt10": "Platelets (per 10 x10^3/uL)", "steatosis": "Steatosis (CAP >= 248 dB/m)"}


def s4(pat: pd.DataFrame) -> pd.DataFrame:
    d0 = pat.assign(plt10=pat["plt"] / 10.0)
    rows = []
    for model_name, covars in [("Model A (clinical model)", MODEL_A),
                               ("Model B (clinical + hepatic/haematological)", MODEL_B)]:
        d = d0.dropna(subset=["log_ssm", "ssm_ge_30", "ssm_ge_40"] + covars)
        X = sm.add_constant(d[covars].astype(float))
        lin = sm.OLS(d["log_ssm"].astype(float), X).fit(cov_type="HC3")
        log30 = logit_mle(d[covars].astype(float), d["ssm_ge_30"].values)
        log40 = firth_profile(d[covars].astype(float), d["ssm_ge_40"].values, terms=covars)
        t30 = log30["table"].set_index("term")
        t40 = log40["terms"]
        rows.append([model_name, f"n = {int(lin.nobs)}",
                     f"{int(d['ssm_ge_30'].sum())} events >= 30 kPa",
                     f"{int(d['ssm_ge_40'].sum())} events >= 40 kPa", "", "", ""])
        for v in covars:
            ci = lin.conf_int().loc[v]
            rows.append([
                f"  {LABELS[v]}",
                f"{np.exp(lin.params[v]):.3f} ({np.exp(ci.iloc[0]):.3f}-{np.exp(ci.iloc[1]):.3f})",
                fmt_p(lin.pvalues[v]),
                f"{t30.loc[v, 'OR']:.2f} ({t30.loc[v, 'ci_low']:.2f}-{t30.loc[v, 'ci_high']:.2f})",
                fmt_p(t30.loc[v, "p"]),
                f"{t40[v]['or']:.2f} ({t40[v]['pl_low']:.2f}-{t40[v]['pl_high']:.2f})",
                fmt_p(t40[v]["plrt_p"])])
    return pd.DataFrame(rows, columns=[
        "Variable", "GMR for log-SSM (95% CI)", "p", "OR for SSM >= 30 kPa (95% CI)", "p ",
        "OR for SSM >= 40 kPa (95% CI), Firth (profile penalised likelihood)", "p  "])


# ---------------------------------------------------------------------------
# S5 - sensitivity analyses for the distribution models
# ---------------------------------------------------------------------------
def _subsets(pat: pd.DataFrame):
    return [
        ("Primary analysis (all successful SSM)", pat),
        ("Reliable SSM only (IQR/median <= 30%)", pat[pat["reliable"] == 1]),
        ("Excluding LSM >= 10 kPa", pat[pat["lsm"] < prep.LSM_CACLD]),
        ("Excluding LSM >= 7.2 kPa", pat[pat["lsm"] < prep.LSM_SIGNIFICANT_FIBROSIS]),
        ("Excluding any alcohol use", pat[pat["alcohol_any"] == 0]),
        ("Excluding XL-probe (LSM) examinations", pat[pat["xl_probe"] == 0]),
        ("Excluding SSM > 75 kPa", pat[pat["ssm"] <= 75]),
        ("Men only", pat[pat["male"] == 1]),
        ("Crohn's disease only", pat[pat["uc"] == 0]),
        ("BMI < 30 kg/m2", pat[pat["bmi"] < 30]),
    ]


GMR_COVARS = ("age", "male", "bmi", "alcohol_any", "lsm")
GMR_COVARS_NO_ALCOHOL = ("age", "male", "bmi", "lsm")


def _s5_row(label, sub, covars):
    aza_users = sub[sub["aza_user"] == 1]
    thio_users = sub[sub["thio_user"] == 1]
    t = gmr(sub, "thiopurine", covars=covars)
    cells = [label, t["n"], f"{t['gmr']:.2f} ({t['low']:.2f}-{t['high']:.2f}); p = {fmt_p(t['p'])}"]
    n_aza = None
    for df, col, unit in [(aza_users, "aza_mg_day", 25), (aza_users, "aza_mg_kg", 0.5),
                          (thio_users, "thio_cumulative_g", 100)]:
        r = gmr(df, col, unit, covars=covars)
        if col == "aza_mg_day":
            n_aza = r["n"]
        if col == "thio_cumulative_g":
            n_thio = r["n"]
        cells.append(f"{r['gmr']:.3f} ({r['low']:.3f}-{r['high']:.3f}); p = {fmt_p(r['p'])}")
    return cells[:3] + [n_aza] + cells[3:5] + [n_thio, cells[5]]


def s5(pat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, (label, sub) in enumerate(_subsets(pat)):
        rows.append(_s5_row(label, sub, GMR_COVARS))
        if k == 0:
            # alcohol use was not recorded in 42 of the 266 patients with a medication
            # history: the primary model refitted without this covariate
            rows.append(_s5_row("Primary model without alcohol use", sub, GMR_COVARS_NO_ALCOHOL))
    return pd.DataFrame(rows, columns=[
        "Analysis set", "n", "Thiopurine exposure: adjusted GMR (95% CI)", "n (AZA users)",
        "AZA daily dose, per 25 mg: adjusted GMR", "AZA dose per 0.5 mg/kg: adjusted GMR",
        "n (thiopurine users)", "Cumulative thiopurine dose per 100 g (thiopurine users): adjusted GMR"])


# ---------------------------------------------------------------------------
# S7 / S8 - spleen stiffness categories
# ---------------------------------------------------------------------------
SSM_BINS = [(0, 21, "< 21"), (21, 30, "21-29.9"), (30, 40, "30-39.9"),
            (40, 50, "40-49.9"), (50, 1e6, ">= 50")]


def _ssm_category(x):
    for lo, hi, lab in SSM_BINS:
        if lo <= x < hi:
            return lab
    return None


def _n_pct_of(x, denom, digits=1):
    """n (%) with the category size as denominator, as printed in the manuscript."""
    k = int(pd.Series(x).fillna(0).sum())
    return f"{k} ({100 * k / denom:.{digits}f}%)" if denom else ""


def s7(pat: pd.DataFrame) -> pd.DataFrame:
    d = pat.assign(cat=pat["ssm"].map(_ssm_category))
    rows = []
    for _, _, lab in SSM_BINS:
        g = d[d["cat"] == lab]
        rows.append([lab, len(g), med_iqr(g["plt"], 0), _n_pct_of(g["thrombocytopenia"], len(g)),
                     med_iqr(g["lsm"]), med_iqr(g["albumin"], 2), med_iqr(g["bilirubin"], 2)])
    rho, p = stats.spearmanr(*pat[["ssm", "plt"]].dropna().T.values)
    rows.append([f"Spearman SSM vs platelets: rho = {rho:.2f}, p = {fmt_p(p)}", "", "", "", "", "", ""])
    return pd.DataFrame(rows, columns=[
        "SSM category (kPa)", "n", "Platelets, median (IQR)", "Platelets < 150, n (%)",
        "LSM, median (IQR)", "Albumin", "Total bilirubin"])


def s8(pat: pd.DataFrame) -> pd.DataFrame:
    d = pat.assign(cat=pat["ssm"].map(_ssm_category))
    rows, ev, tot = [], [], []
    for _, _, lab in SSM_BINS:
        g = d[d["cat"] == lab]
        thio_users = g[g["thio_user"] == 1]
        rows.append([lab, len(g), _n_pct_of(g["thiopurine"], len(g)),
                     _n_pct_of((g["aza_mg_day"] >= 100).astype(float), len(g)),
                     med_iqr(thio_users["thio_months"], 0), int(g["thrombocytopenia"].sum()),
                     int(g["lsm_ge_10"].sum()), _n_pct_of(g["male"], len(g), 0)])
        ev.append(g["thiopurine"].fillna(0).sum())
        tot.append(len(g))
    _, p = cochran_armitage(ev, tot)
    rows.append([f"Cochran-Armitage trend p for thiopurine exposure across categories: {fmt_p(p)}",
                 "", "", "", "", "", "", ""])
    return pd.DataFrame(rows, columns=[
        "SSM category (kPa)", "n", "Thiopurine-exposed, n (%)", "AZA >= 100 mg/day, n (%)",
        "Thiopurine duration in users, months, median (IQR)", "Platelets < 150, n", "LSM >= 10 kPa, n",
        "Male, n (%)"])


# ---------------------------------------------------------------------------
# S9 - specificity analysis
# ---------------------------------------------------------------------------
S9_VARS = [
    ("Anti-TNF duration, per 12 months", "antitnf_months", 12),
    ("Anti-TNF exposure", "antitnf", 1),
    ("Methotrexate exposure", "methotrexate", 1),
    ("Vedolizumab/ustekinumab exposure", "vedo_uste", 1),
    ("Corticosteroid courses, per course", "steroid_courses", 1),
    ("Ulcerative colitis (vs Crohn's disease)", "uc", 1),
    ("Disease duration, per year", "disease_duration_years", 1),
    ("Prior bowel resection", "resection", 1),
    ("Documented flare in past 12 months", "flare_any", 1),
    ("Number of flares in past 12 months, per flare", "flare_n", 1),
    ("Clinical remission", "remission", 1),
    ("Mean CRP over past 12 months, per mg/L", "crp_mean", 1),
    ("Extraintestinal manifestations", "eim", 1),
    ("Any alcohol use", "alcohol_any", 1),
    ("Platelet count, per 10 x10^3/uL", "plt10", 1),
    ("Hepatic steatosis (CAP >= 248 dB/m)", "steatosis", 1),
]


def s9(pat: pd.DataFrame) -> pd.DataFrame:
    d0 = pat.assign(plt10=pat["plt"] / 10.0)
    rows = []
    for label, col, unit in S9_VARS:
        r = firth_or(d0, col, unit)
        rows.append([label, f"{r['n']} / {r['events']}",
                     f"{r['or']:.2f} ({r['low']:.2f}-{r['high']:.2f}); "
                     f"{'p ' if fmt_p(r['p']).startswith('<') else 'p = '}{fmt_p(r['p'])}"])
    return pd.DataFrame(rows, columns=[
        "Exposure or characteristic (each adjusted for age, sex and BMI)", "n / events",
        "OR for SSM >= 40 kPa (95% CI); p"])


# ---------------------------------------------------------------------------
# S10 - daily dose and weight-adjusted dose versus the two thresholds
# ---------------------------------------------------------------------------
def s10(pat: pd.DataFrame) -> pd.DataFrame:
    users = pat[pat["aza_user"] == 1]
    rows = []
    combos = [("All patients (non-users = 0)", pat, "Azathioprine daily dose, per 25 mg/day",
               "aza_mg_day", 25),
              ("Azathioprine users only", users, "Azathioprine daily dose, per 25 mg/day",
               "aza_mg_day", 25),
              ("Azathioprine users only",
               users, "Azathioprine weight-adjusted dose, per 0.5 mg/kg/day", "aza_mg_kg", 0.5)]
    for pop, df, exp_label, col, unit in combos:
        for model_label, extra in MODELS:
            r40 = firth_or(df, col, unit, extra, outcome="ssm_ge_40")
            cols = [col, "age", "male", "bmi"] + list(extra)
            d = df.dropna(subset=["ssm_ge_30"] + cols).copy()
            d[col] = d[col] / unit
            m30 = logit_mle(d[cols].astype(float), d["ssm_ge_30"].values)
            t = m30["table"].set_index("term").loc[col]
            rows.append([pop, exp_label, model_label, f"{r40['n']} / {r40['events']}",
                         f"{r40['or']:.2f} ({r40['low']:.2f}-{r40['high']:.2f})", fmt_p(r40["p"]),
                         f"{m30['n']} / {m30['events']}",
                         f"{t.OR:.2f} ({t.ci_low:.2f}-{t.ci_high:.2f})", fmt_p(t.p)])
    # cumulative thiopurine dose at the >= 30 kPa threshold (quoted in the Table 3B footnote)
    d = pat.dropna(subset=["ssm_ge_30", "thio_cumulative_g", "age", "male", "bmi"]).copy()
    d["thio_cumulative_g"] = d["thio_cumulative_g"] / 100.0
    m = logit_mle(d[["thio_cumulative_g", "age", "male", "bmi"]].astype(float),
                  d["ssm_ge_30"].values)
    t = m["table"].set_index("term").loc["thio_cumulative_g"]
    rows.append(["All patients (non-users = 0)", "Cumulative thiopurine dose, per 100 g",
                 "Model 1: age, sex, BMI", "", "", "", f"{m['n']} / {m['events']}",
                 f"{t.OR:.2f} ({t.ci_low:.2f}-{t.ci_high:.2f})", fmt_p(t.p)])
    return pd.DataFrame(rows, columns=[
        "Population", "Exposure metric", "Adjustment", "n / events (SSM >= 40)",
        "OR for SSM >= 40 kPa (95% CI)", "p", "n / events (SSM >= 30)",
        "OR for SSM >= 30 kPa (95% CI)", "p "])


# ---------------------------------------------------------------------------
# S11 - correlations among exposure metrics
# ---------------------------------------------------------------------------
S11_VARS = [("Cumulative azathioprine dose (g)", "aza_cumulative_g"),
            ("Duration of azathioprine (months)", "aza_months"),
            ("Daily dose (mg/day)", "aza_mg_day"),
            ("Weight-adjusted dose (mg/kg/day)", "aza_mg_kg"),
            ("Disease duration (years)", "disease_duration_years"),
            ("Age (years)", "age"), ("Body weight (kg)", "weight"),
            ("Anti-TNF duration (months)", "antitnf_months"),
            ("Corticosteroid courses (n)", "steroid_courses")]


def s11(pat: pd.DataFrame) -> pd.DataFrame:
    users = pat[pat["aza_user"] == 1]
    labels = [l for l, _ in S11_VARS]
    cols = [c for _, c in S11_VARS]
    out = pd.DataFrame(index=labels, columns=labels, dtype=object)
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i == j:
                out.iloc[i, j] = "-"
                continue
            d = users[[ci, cj]].dropna()
            rho, p = stats.spearmanr(d[ci], d[cj])
            out.iloc[i, j] = f"{rho:.2f}" + ("*" if p < 0.05 else "")
    n_pairs = [len(users[[a, b]].dropna()) for a in cols for b in cols if a != b]
    out.index.name = (f"Spearman correlations among azathioprine users (n = {len(users)}; pairwise "
                      f"complete cases, n = {min(n_pairs)}-{max(n_pairs)})")
    res = out.reset_index()
    d = pat[["aza_cumulative_g", "disease_duration_years"]].dropna()
    rho, _ = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    res.loc[len(res)] = [f"All patients (non-users = 0, n = {len(d)}): cumulative azathioprine dose "
                         f"vs disease duration, rho = {rho:.2f}"] + [""] * (res.shape[1] - 1)
    return res


# ---------------------------------------------------------------------------
# S15 - azathioprine daily dose and duration entered simultaneously
# ---------------------------------------------------------------------------
def _joint(df, covars):
    cols = ["aza_mg_day", "aza_months"] + list(covars)
    d = df.dropna(subset=["ssm_ge_40"] + cols).copy()
    cols = [c for c in cols if c in ("aza_mg_day", "aza_months") or d[c].nunique() > 1]
    d["aza_mg_day"] = d["aza_mg_day"] / 25.0
    d["aza_months"] = d["aza_months"] / 12.0
    return firth_profile(d[cols].astype(float), d["ssm_ge_40"].values,
                         terms=["aza_mg_day", "aza_months"])


def s15(pat: pd.DataFrame) -> pd.DataFrame:
    users = pat[pat["aza_user"] == 1]
    m1, m2 = ("age", "male", "bmi"), ("age", "male", "bmi", "disease_duration_years")
    sets = [
        ("All patients (non-users = 0), Model 1", pat, m1),
        ("Azathioprine users only, Model 1", users, m1),
        ("Azathioprine users only, Model 2 (+ disease duration)", users, m2),
        ("Users, reliability subset (IQR/median <= 30%)", users[users["reliable"] == 1], m1),
        ("Users, excluding the measurement at the 100-kPa ceiling",
         users[users["ssm"] < prep.DEVICE_CEILING], m1),
        ("Users, excluding both cumulative doses above 800 g",
         users[users["aza_cumulative_g"].isna() | (users["aza_cumulative_g"] <= 800)], m1),
        ("Users, BMI < 30 kg/m2", users[users["bmi"] < 30], m1),
    ]
    rows = []
    for label, df, cov in sets:
        r = _joint(df, cov)
        a, b = r["terms"]["aza_mg_day"], r["terms"]["aza_months"]
        rows.append([label, f"{r['n']} / {r['events']}",
                     f"{a['or']:.2f} ({a['pl_low']:.2f}-{a['pl_high']:.2f})", fmt_p(a["plrt_p"]),
                     f"{b['or']:.2f} ({b['pl_low']:.2f}-{b['pl_high']:.2f})", fmt_p(b["plrt_p"])])
    return pd.DataFrame(rows, columns=[
        "Analysis set", "n / events", "AZA daily dose, per 25 mg/day: OR (95% CI)", "p",
        "AZA duration, per 12 months: OR (95% CI)", "p "])


# ---------------------------------------------------------------------------
# S16 - threshold scan, quantile regression and shape of the association
# ---------------------------------------------------------------------------
SCAN_THRESHOLDS = [21, 25, 30, 35, 40, 45, 50]


def _rcs3(x, knots):
    """Non-linear basis of a restricted cubic spline with three knots."""
    k1, k2, k3 = knots
    pos = lambda u: np.clip(u, 0, None) ** 3
    return (pos(x - k1) - pos(x - k2) * (k3 - k1) / (k3 - k2)
            + pos(x - k3) * (k2 - k1) / (k3 - k2)) / (k3 - k1) ** 2


def s16(pat: pd.DataFrame) -> pd.DataFrame:
    cum = "thio_cumulative_g"
    rows = [["(A) Threshold scan: OR for SSM at or above each threshold per 100 g (Model 1)", "", ""]]
    for t in SCAN_THRESHOLDS:
        d = pat.assign(_y=(pat["ssm"] >= t).astype(float))
        r = firth_or(d, cum, 100, (), outcome="_y")
        p = fmt_p(r["p"])
        rows.append([f"  SSM >= {t} kPa" + (" (primary threshold)" if t == 40 else ""),
                     f"{r['n']} / {r['events']}",
                     f"{r['or']:.2f} ({r['low']:.2f}-{r['high']:.2f}); "
                     f"{'p ' if p.startswith('<') else 'p = '}{p}"])
    rows.append(["(B) Quantile regression of log-SSM: ratio of the SSM quantile per 100 g", "", ""])
    for label, df in [("All patients", pat), ("Thiopurine users", pat[pat["thio_user"] == 1])]:
        for r in _quantreg(df, cum, unit=100):
            p = fmt_p(r["p"])
            rows.append([f"  {label}, {int(r['quantile'] * 100)}th percentile", str(r["n"]),
                         f"{r['ratio']:.3f} ({r['low']:.3f}-{r['high']:.3f}); "
                         f"{'p ' if p.startswith('<') else 'p = '}{p}"])
    rows.append(["(C) Shape of the association (Model 1)", "", ""])
    d = pat.dropna(subset=["ssm_ge_40", cum, "age", "male", "bmi"]).copy()
    x = d[cum].to_numpy(float)
    knots = np.percentile(x[x > 0], [10, 50, 90])
    d["c100"] = x / 100.0
    d["nl"] = _rcs3(x, knots) / 100.0
    d["c2"] = (x / 100.0) ** 2
    nev = f"{len(d)} / {int(d['ssm_ge_40'].sum())}"
    spline = firth_profile(d[["c100", "nl", "age", "male", "bmi"]].astype(float),
                           d["ssm_ge_40"].values, terms=["nl"], ci=False)["terms"]["nl"]
    quad = firth_profile(d[["c100", "c2", "age", "male", "bmi"]].astype(float),
                         d["ssm_ge_40"].values, terms=["c2"], ci=False)["terms"]["c2"]
    rows.append([f"  Restricted cubic spline (knots at {knots[0]:.0f}, {knots[1]:.0f} and "
                 f"{knots[2]:.0f} g): test for non-linearity", nev, f"p = {fmt_p(spline['plrt_p'])}"])
    rows.append(["  Quadratic term: test for non-linearity", nev, f"p = {fmt_p(quad['plrt_p'])}"])
    d["gt300"] = (x > 300).astype(float)
    g = firth_profile(d[["gt300", "age", "male", "bmi"]].astype(float), d["ssm_ge_40"].values,
                      terms=["gt300"])["terms"]["gt300"]
    hi_ev, hi_n = int(d.loc[d["gt300"] == 1, "ssm_ge_40"].sum()), int(d["gt300"].sum())
    lo_ev, lo_n = int(d.loc[d["gt300"] == 0, "ssm_ge_40"].sum()), int((d["gt300"] == 0).sum())
    rows.append([f"  Cumulative dose > 300 g vs <= 300 g or none ({hi_ev}/{hi_n} vs {lo_ev}/{lo_n} events)",
                 nev, f"{g['or']:.2f} ({g['pl_low']:.2f}-{g['pl_high']:.2f}); p = {fmt_p(g['plrt_p'])}"])
    d["capped"] = np.minimum(x, 500.0) / 100.0
    c = firth_profile(d[["capped", "age", "male", "bmi"]].astype(float), d["ssm_ge_40"].values,
                      terms=["capped"])["terms"]["capped"]
    rows.append(["  Cumulative dose capped at 500 g, per 100 g", nev,
                 f"{c['or']:.2f} ({c['pl_low']:.2f}-{c['pl_high']:.2f}); p = {fmt_p(c['plrt_p'])}"])
    return pd.DataFrame(rows, columns=["Analysis", "n / events", "Estimate (95% CI); p"])


def main():
    OUT.mkdir(exist_ok=True)
    _, pat, ref = prep.load_all()
    tables = {
        "tableS1_patients_vs_reference.csv": s1(pat, ref),
        "tableS2_univariable.csv": s2(pat),
        "tableS3_quantile_patients_vs_reference.csv": s3(pat, ref),
        "tableS4_multivariable_models.csv": s4(pat),
        "tableS5_sensitivity_distribution.csv": s5(pat),
        "tableS6_quantile_exposure.csv": s6(pat),
        "tableS7_platelets_by_ssm.csv": s7(pat),
        "tableS8_exposure_by_ssm.csv": s8(pat),
        "tableS9_specificity.csv": s9(pat),
        "tableS10_dose_thresholds.csv": s10(pat),
        "tableS11_correlations.csv": s11(pat),
        "tableS15_dose_and_duration_jointly.csv": s15(pat),
        "tableS16_threshold_quantile_linearity.csv": s16(pat),
    }
    for name, df in tables.items():
        df.to_csv(OUT / name, index=False)
        print(f"wrote {name:44s} ({len(df)} rows)")


if __name__ == "__main__":
    main()

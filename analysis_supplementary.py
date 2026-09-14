"""Supplementary Tables S1-S11 of the ELASTIBD spleen-stiffness manuscript."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import prep
from analysis_tables import (OUT, firth_or, fisher, gmr, mannwhitney, med_iqr,
                             n_pct)
from firth import cochran_armitage, firth_logit, logit_mle, wilson_ci


# ---------------------------------------------------------------------------
# S1 - patients versus the healthy reference sample
# ---------------------------------------------------------------------------
def s1(pat: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def cont(label, col, d=1):
        rows.append([label, med_iqr(pat[col], d), med_iqr(ref[col], d),
                     f"{mannwhitney(pat[col], ref[col]):.3g}"])

    def binr(label, col):
        rows.append([label, n_pct(pat[col]), n_pct(ref[col]),
                     f"{fisher(pat[col], ref[col]):.3g}"])

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
                 f"{stats.ttest_ind(np.log(pat['ssm']), np.log(ref['ssm']), equal_var=False).pvalue:.3g}"])
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
         f"{adj['gmr']:.2f} ({adj['low']:.2f}-{adj['high']:.2f})", f"p = {adj['p']:.3g}", ""],
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
    ("Thiopurine (AZA or 6-MP)", "thiopurine"), ("Azathioprine", "aza"), ("Anti-TNF", "antitnf"),
    ("Methotrexate", "methotrexate"), ("Vedolizumab or ustekinumab", "vedo_uste"),
    ("Budesonide", "budesonide"), ("Oral mesalazine", "mesalazine"),
    ("Isoniazid prophylaxis", "isoniazid"),
]


def s2(pat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in BINARY_S2:
        d = pat.dropna(subset=[col])
        p = mannwhitney(d.loc[d[col] == 1, "ssm"], d.loc[d[col] == 0, "ssm"])
        rows.append([label, len(d), "", "", "", f"{p:.3g}"])
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
        rows.append([label, len(d), f"rho = {rho:+.3f}", "", "", f"{p:.3g}"])
    # smoking (three levels)
    d = pat.dropna(subset=["smoking"])
    groups = [d.loc[d["smoking"] == v, "ssm"] for v in sorted(d["smoking"].unique())]
    rows.append(["Smoking status", len(d), "", "", "", f"{stats.kruskal(*groups).pvalue:.3g}"])
    for v, name in [(0.0, "Never"), (1.0, "Current"), (2.0, "Former")]:
        g = d[d["smoking"] == v]
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
        res = smf.quantreg(formula, d).fit(q=q)
        ci = res.conf_int().loc["expo"]
        out.append(dict(quantile=q, ratio=np.exp(res.params["expo"]),
                        low=np.exp(ci.iloc[0]), high=np.exp(ci.iloc[1]),
                        p=res.pvalues["expo"], n=int(res.nobs)))
    return out


def s3(pat, ref) -> pd.DataFrame:
    both = pd.concat([pat.assign(patient=1.0), ref.assign(patient=0.0)], ignore_index=True)
    rows = [[f"{int(r['quantile'] * 100)}th percentile",
             f"{r['ratio']:.2f} ({r['low']:.2f}-{r['high']:.2f})", f"{r['p']:.3g}"]
            for r in _quantreg(both, "patient")]
    return pd.DataFrame(rows, columns=[
        "SSM quantile", "Ratio of quantile, patients vs reference sample (95% CI)", "p"])


def s6(pat) -> pd.DataFrame:
    users = pat[pat["aza_user"] == 1]
    a = _quantreg(pat, "thiopurine")
    b = _quantreg(users, "aza_mg_day", unit=25)
    rows = [[f"{int(x['quantile'] * 100)}th percentile",
             f"{x['ratio']:.2f} ({x['low']:.2f}-{x['high']:.2f})", f"{x['p']:.3g}",
             f"{y['ratio']:.3f} ({y['low']:.3f}-{y['high']:.3f})", f"{y['p']:.3g}"]
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
        log40 = firth_logit(d[covars].astype(float), d["ssm_ge_40"].values)
        t30 = log30["table"].set_index("term")
        t40 = log40["table"].set_index("term")
        rows.append([model_name, f"n = {int(lin.nobs)}",
                     f"{int(d['ssm_ge_30'].sum())} events >= 30 kPa",
                     f"{int(d['ssm_ge_40'].sum())} events >= 40 kPa", "", "", ""])
        for v in covars:
            ci = lin.conf_int().loc[v]
            rows.append([
                f"  {LABELS[v]}",
                f"{np.exp(lin.params[v]):.3f} ({np.exp(ci.iloc[0]):.3f}-{np.exp(ci.iloc[1]):.3f})",
                f"{lin.pvalues[v]:.3g}",
                f"{t30.loc[v, 'OR']:.2f} ({t30.loc[v, 'ci_low']:.2f}-{t30.loc[v, 'ci_high']:.2f})",
                f"{t30.loc[v, 'p']:.3g}",
                f"{t40.loc[v, 'OR']:.2f} ({t40.loc[v, 'ci_low']:.2f}-{t40.loc[v, 'ci_high']:.2f})",
                f"{t40.loc[v, 'p']:.3g}"])
    return pd.DataFrame(rows, columns=[
        "Variable", "GMR for log-SSM (95% CI)", "p", "OR for SSM >= 30 kPa (95% CI)", "p ",
        "OR for SSM >= 40 kPa (95% CI), Firth", "p  "])


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


def s5(pat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in _subsets(pat):
        users = sub[sub["aza_user"] == 1]
        t = gmr(sub, "thiopurine")
        cells = [label, t["n"], f"{t['gmr']:.2f} ({t['low']:.2f}-{t['high']:.2f}); p = {t['p']:.3g}"]
        n_users = None
        for col, unit in [("aza_mg_day", 25), ("aza_mg_kg", 0.5), ("aza_cumulative_g", 100)]:
            r = gmr(users, col, unit)
            n_users = r["n"]
            cells.append(f"{r['gmr']:.3f} ({r['low']:.3f}-{r['high']:.3f}); p = {r['p']:.3g}")
        rows.append(cells[:2] + [cells[2], n_users] + cells[3:])
    return pd.DataFrame(rows, columns=[
        "Analysis set", "n", "Thiopurine exposure: adjusted GMR (95% CI)", "n (AZA users)",
        "AZA daily dose, per 25 mg: adjusted GMR", "AZA dose per 0.5 mg/kg: adjusted GMR",
        "AZA cumulative dose per 100 g: adjusted GMR"])


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


def _n_pct_of(x, denom):
    """n (%) with the category size as denominator, as printed in the manuscript."""
    k = int(pd.Series(x).fillna(0).sum())
    return f"{k} ({100 * k / denom:.1f}%)" if denom else ""


def s7(pat: pd.DataFrame) -> pd.DataFrame:
    d = pat.assign(cat=pat["ssm"].map(_ssm_category))
    rows = []
    for _, _, lab in SSM_BINS:
        g = d[d["cat"] == lab]
        rows.append([lab, len(g), med_iqr(g["plt"], 0), _n_pct_of(g["thrombocytopenia"], len(g)),
                     med_iqr(g["lsm"]), med_iqr(g["albumin"], 2), med_iqr(g["bilirubin"], 2)])
    rho, p = stats.spearmanr(*pat[["ssm", "plt"]].dropna().T.values)
    rows.append([f"Spearman SSM vs platelets: rho = {rho:.2f}, p = {p:.3g}", "", "", "", "", "", ""])
    return pd.DataFrame(rows, columns=[
        "SSM category (kPa)", "n", "Platelets, median (IQR)", "Platelets < 150, n (%)",
        "LSM, median (IQR)", "Albumin", "Total bilirubin"])


def s8(pat: pd.DataFrame) -> pd.DataFrame:
    d = pat.assign(cat=pat["ssm"].map(_ssm_category))
    rows, ev, tot = [], [], []
    for _, _, lab in SSM_BINS:
        g = d[d["cat"] == lab]
        users = g[g["aza_user"] == 1]
        rows.append([lab, len(g), _n_pct_of(g["thiopurine"], len(g)),
                     _n_pct_of((g["aza_mg_day"] >= 100).astype(float), len(g)),
                     med_iqr(users["aza_months"], 0), int(g["thrombocytopenia"].sum()),
                     int(g["lsm_ge_10"].sum()), _n_pct_of(g["male"], len(g))])
        ev.append(g["thiopurine"].fillna(0).sum())
        tot.append(len(g))
    _, p = cochran_armitage(ev, tot)
    rows.append([f"Cochran-Armitage trend p for thiopurine exposure across categories: {p:.3g}",
                 "", "", "", "", "", "", ""])
    return pd.DataFrame(rows, columns=[
        "SSM category (kPa)", "n", "Thiopurine-exposed, n (%)", "AZA >= 100 mg/day, n (%)",
        "AZA duration in users, months, median (IQR)", "Platelets < 150, n", "LSM >= 10 kPa, n",
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
                     f"{r['or']:.2f} ({r['low']:.2f}-{r['high']:.2f}); p = {r['p']:.3g}"])
    return pd.DataFrame(rows, columns=[
        "Exposure or characteristic (each adjusted for age, sex and BMI)", "n / events",
        "OR for SSM >= 40 kPa (95% CI); p"])


# ---------------------------------------------------------------------------
# S10 - daily dose and weight-adjusted dose versus the two thresholds
# ---------------------------------------------------------------------------
def s10(pat: pd.DataFrame) -> pd.DataFrame:
    from analysis_tables import MODELS
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
                         f"{r40['or']:.2f} ({r40['low']:.2f}-{r40['high']:.2f})", f"{r40['p']:.3f}",
                         f"{m30['n']} / {m30['events']}",
                         f"{t.OR:.2f} ({t.ci_low:.2f}-{t.ci_high:.2f})", f"{t.p:.3f}"])
    # cumulative dose at the >= 30 kPa threshold (quoted in the footnote of Table 3B)
    d = pat.dropna(subset=["ssm_ge_30", "aza_cumulative_g", "age", "male", "bmi"]).copy()
    d["aza_cumulative_g"] = d["aza_cumulative_g"] / 100.0
    m = logit_mle(d[["aza_cumulative_g", "age", "male", "bmi"]].astype(float),
                  d["ssm_ge_30"].values)
    t = m["table"].set_index("term").loc["aza_cumulative_g"]
    rows.append(["All patients (non-users = 0)", "Cumulative dose, per 100 g",
                 "Model 1: age, sex, BMI", "", "", "", f"{m['n']} / {m['events']}",
                 f"{t.OR:.2f} ({t.ci_low:.2f}-{t.ci_high:.2f})", f"{t.p:.3f}"])
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
    out.index.name = f"Spearman correlations among azathioprine users (n = {len(users)})"
    return out.reset_index()


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
    }
    for name, df in tables.items():
        df.to_csv(OUT / name, index=False)
        print(f"wrote {name:44s} ({len(df)} rows)")


if __name__ == "__main__":
    main()

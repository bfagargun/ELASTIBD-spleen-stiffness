"""Main tables of the ELASTIBD spleen-stiffness manuscript.

Produces, in ``outputs/``:

* ``table1_characteristics.csv``      - patient characteristics by thiopurine exposure
* ``table2a_categorical_exposure.csv`` - exposure categories and the SSM distribution
* ``table2b_continuous_exposure.csv``  - continuous exposure metrics (geometric mean ratios)
* ``table3a_thresholds_by_category.csv`` - proportions above thresholds, trend tests
* ``table3b_firth_models.csv``         - adjusted odds ratios (Firth)
* ``table4_marked_cases.csv``          - the patients with SSM >= 40 kPa

Run with ``python run_all.py`` or directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import prep
from firth import cochran_armitage, firth_logit

_HERE = Path(__file__).resolve().parent
OUT = (_HERE.parent if _HERE.name == "code" else _HERE) / "outputs"
Z = 1.959963984540054


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def med_iqr(x, d=1):
    x = pd.Series(x).dropna()
    if x.empty:
        return ""
    return f"{x.median():.{d}f} ({x.quantile(.25):.{d}f}-{x.quantile(.75):.{d}f})"


def n_pct(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return ""
    return f"{int(x.sum())} ({100 * x.mean():.1f}%)"


def mannwhitney(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.mannwhitneyu(a, b).pvalue


def fisher(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    tbl = [[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]]
    return stats.fisher_exact(tbl).pvalue


def _drop_constant(d, cols, exposure):
    """Covariates with no variation in a subset are dropped (e.g. sex in a
    sex-stratified analysis); the exposure itself is always retained."""
    return [c for c in cols if c == exposure or d[c].nunique(dropna=True) > 1]


def gmr(df, exposure, unit=1.0, covars=("age", "male", "bmi", "alcohol_any", "lsm"),
        outcome="log_ssm"):
    """Geometric mean ratio per `unit` of exposure (OLS on log SSM, HC3 robust SE)."""
    cols = [exposure] + list(covars)
    d = df.dropna(subset=[outcome] + cols).copy()
    cols = _drop_constant(d, cols, exposure)
    d[exposure] = d[exposure] / unit
    X = sm.add_constant(d[cols].astype(float))
    res = sm.OLS(d[outcome].astype(float), X).fit(cov_type="HC3")
    ci = res.conf_int().loc[exposure]
    return dict(
        n=int(res.nobs),
        gmr=float(np.exp(res.params[exposure])),
        low=float(np.exp(ci.iloc[0])),
        high=float(np.exp(ci.iloc[1])),
        p=float(res.pvalues[exposure]),
    )


def gmr_categorical(df, cat_col, reference, covars=("age", "male", "bmi", "alcohol_any", "lsm")):
    """Geometric mean ratios for each level of a categorical exposure."""
    d = df.dropna(subset=["log_ssm", cat_col] + list(covars)).copy()
    dummies = pd.get_dummies(d[cat_col].astype(str), prefix="lvl", dtype=float)
    ref_col = f"lvl_{reference}"
    dummies = dummies.drop(columns=[ref_col])
    X = sm.add_constant(pd.concat([dummies, d[list(covars)].astype(float)], axis=1))
    res = sm.OLS(d["log_ssm"].astype(float), X).fit(cov_type="HC3")
    out = {}
    for col in dummies.columns:
        ci = res.conf_int().loc[col]
        out[col.replace("lvl_", "")] = dict(
            gmr=float(np.exp(res.params[col])), low=float(np.exp(ci.iloc[0])),
            high=float(np.exp(ci.iloc[1])), p=float(res.pvalues[col]))
    return out


def firth_or(df, exposure, unit, extra=(), outcome="ssm_ge_40"):
    """Adjusted OR per `unit` of exposure from Firth penalised logistic regression."""
    cols = [exposure, "age", "male", "bmi"] + list(extra)
    d = df.dropna(subset=[outcome] + cols).copy()
    cols = _drop_constant(d, cols, exposure)
    d[exposure] = d[exposure] / unit
    fit = firth_logit(d[cols].astype(float), d[outcome].astype(float).values)
    row = fit["table"].set_index("term").loc[exposure]
    return dict(n=fit["n"], events=fit["events"], **{
        "or": float(row.OR), "low": float(row.ci_low), "high": float(row.ci_high),
        "p": float(row.p)})


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------
def table1(pat: pd.DataFrame) -> pd.DataFrame:
    exposed = pat[pat["thiopurine"] == 1]
    unexposed = pat[pat["thiopurine"] == 0]
    rows = []

    def cont(label, col, d=1):
        rows.append([label, med_iqr(pat[col], d), med_iqr(exposed[col], d),
                     med_iqr(unexposed[col], d),
                     f"{mannwhitney(exposed[col], unexposed[col]):.3g}"])

    def binr(label, col):
        rows.append([label, n_pct(pat[col]), n_pct(exposed[col]), n_pct(unexposed[col]),
                     f"{fisher(exposed[col], unexposed[col]):.3g}"])

    cont("Age (years)", "age", 0)
    binr("Male sex", "male")
    cont("BMI (kg/m2)", "bmi")
    cont("Waist circumference (cm)", "waist", 0)
    binr("Any alcohol use", "alcohol_any")
    binr("Hypertension", "hypertension")
    binr("Diabetes mellitus", "diabetes")
    binr("Ulcerative colitis", "uc")
    cont("Disease duration (months)", "disease_duration_months", 0)
    binr("Clinical remission", "remission")
    binr("Prior bowel resection", "resection")
    binr("Extraintestinal manifestations", "eim")
    cont("Mean CRP, past 12 months (mg/L)", "crp_mean")
    binr("Anti-TNF exposure", "antitnf")
    cont("Systemic corticosteroid courses (n)", "steroid_courses", 0)
    binr("Methotrexate exposure", "methotrexate")
    binr("Vedolizumab or ustekinumab exposure", "vedo_uste")
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
    cont("SSM IQR/median (%)", "ssm_iqr_median", 0)
    for lab, col in [("SSM >= 21 kPa", "ssm_ge_21"), ("SSM >= 30 kPa", "ssm_ge_30"),
                     ("SSM >= 40 kPa", "ssm_ge_40"), ("SSM >= 50 kPa", "ssm_ge_50")]:
        binr(lab, col)
    cont("SSM/LSM ratio", "ssm_lsm_ratio")

    return pd.DataFrame(rows, columns=[
        "Variable", f"All patients (n = {len(pat)})",
        f"Thiopurine-exposed (n = {len(exposed)})",
        f"Not exposed (n = {len(unexposed)})", "p"])


# ---------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------
def table2a(pat: pd.DataFrame) -> pd.DataFrame:
    users = pat[pat["aza_user"] == 1]
    blocks = [
        ("Thiopurine exposure", pat, "thiopurine", {0.0: "Not exposed", 1.0: "Exposed"}, "Not exposed"),
        ("Azathioprine exposure", pat, "aza", {0.0: "Not exposed", 1.0: "Exposed"}, "Not exposed"),
        ("Treatment regimen", pat, "aza_combo", None, "Neither"),
        ("Azathioprine daily dose (users only)", users, "aza_dose_cat", None, "100 mg"),
        ("Azathioprine duration (users only)", users, "aza_duration_cat", None, "<= 12 months"),
        ("Azathioprine cumulative dose (users only)", users, "aza_cumulative_cat", None, "<= 100 g"),
    ]
    rows = []
    for name, df, col, mapping, ref in blocks:
        series = df[col].map(mapping) if mapping else df[col].astype("object")
        d = df.assign(_lvl=series).dropna(subset=["_lvl"])
        groups = [g["ssm"].dropna().values for _, g in d.groupby("_lvl", observed=True)]
        crude = (stats.kruskal(*groups).pvalue if len(groups) > 2
                 else stats.mannwhitneyu(groups[0], groups[1]).pvalue)
        adj = gmr_categorical(d, "_lvl", ref)
        rows.append([name, "", "", "", "", "", f"{crude:.3g}", ""])
        order = [ref] + [l for l in d["_lvl"].unique() if l != ref]
        for lvl in order:
            g = d[d["_lvl"] == lvl]
            a = adj.get(lvl)
            rows.append([
                f"  {lvl}", len(g), med_iqr(g["ssm"]),
                f"{g['ssm'].mean():.1f} +/- {g['ssm'].std():.1f}",
                f"{100 * g['ssm_ge_30'].mean():.1f}%", f"{100 * g['ssm_ge_40'].mean():.1f}%", "",
                "1 (reference)" if lvl == ref else
                f"{a['gmr']:.2f} ({a['low']:.2f}-{a['high']:.2f}); p = {a['p']:.2f}"])
    return pd.DataFrame(rows, columns=[
        "Exposure", "n", "SSM median (IQR)", "SSM mean +/- SD", "SSM >= 30 kPa",
        "SSM >= 40 kPa", "Crude p", "Adjusted GMR (95% CI)"])


def table2b(pat: pd.DataFrame) -> pd.DataFrame:
    users = pat[pat["aza_user"] == 1]
    metrics = [("Daily dose (mg/day)", "aza_mg_day", 25, "per 25 mg"),
               ("Weight-adjusted dose (mg/kg/day)", "aza_mg_kg", 0.5, "per 0.5 mg/kg"),
               ("Duration (months)", "aza_months", 12, "per 12 months"),
               ("Cumulative dose (g)", "aza_cumulative_g", 100, "per 100 g")]
    rows = []
    for label, col, unit, unit_label in metrics:
        d = users.dropna(subset=[col, "ssm"])
        rho, p_rho = stats.spearmanr(d[col], d["ssm"])
        crude = gmr(users, col, unit, covars=())
        adj = gmr(users, col, unit)
        rows.append([label, len(d), f"rho = {rho:+.3f}; p = {p_rho:.3g}", unit_label,
                     f"{crude['gmr']:.3f} ({crude['low']:.3f}-{crude['high']:.3f}); p = {crude['p']:.3g}",
                     f"{adj['gmr']:.3f} ({adj['low']:.3f}-{adj['high']:.3f}); p = {adj['p']:.3g}"])
    return pd.DataFrame(rows, columns=[
        "Azathioprine exposure metric (users only)", "n", "Spearman correlation with SSM",
        "Unit", "Crude GMR (95% CI)", "Adjusted GMR (95% CI)"])


# ---------------------------------------------------------------------------
# Table 3
# ---------------------------------------------------------------------------
CAT4 = [("Cumulative azathioprine dose", "aza_cumulative_cat4"),
        ("Azathioprine duration", "aza_duration_cat4"),
        ("Azathioprine daily dose", "aza_dose_cat4")]


def table3a(pat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in CAT4:
        d = pat.dropna(subset=[col])
        grp = d.groupby(col, observed=False)
        counts = grp.agg(n=("ssm", "size"), e40=("ssm_ge_40", "sum"), e30=("ssm_ge_30", "sum"))
        counts = counts.loc[[c for c in d[col].cat.categories]]
        t40_all = cochran_armitage(counts["e40"], counts["n"])[1]
        t40_u = cochran_armitage(counts["e40"][1:], counts["n"][1:])[1]
        t30_all = cochran_armitage(counts["e30"], counts["n"])[1]
        t30_u = cochran_armitage(counts["e30"][1:], counts["n"][1:])[1]
        rows.append([label, "", "", "", "", f"{t40_all:.3g} / {t40_u:.3g}",
                     f"{t30_all:.3g} / {t30_u:.3g}"])
        for lvl in counts.index:
            g = d[d[col] == lvl]
            n = int(counts.loc[lvl, "n"])
            rows.append([
                f"  {'No azathioprine' if lvl == 'none' else lvl}", n, med_iqr(g["ssm"]),
                f"{int(counts.loc[lvl, 'e40'])} ({100 * counts.loc[lvl, 'e40'] / n:.1f}%)",
                f"{int(counts.loc[lvl, 'e30'])} ({100 * counts.loc[lvl, 'e30'] / n:.1f}%)", "", ""])
    return pd.DataFrame(rows, columns=[
        "Exposure category", "n", "SSM median (IQR)", "SSM >= 40 kPa, n (%)",
        "SSM >= 30 kPa, n (%)", "Trend p for SSM >= 40 (all / users)",
        "Trend p for SSM >= 30 (all / users)"])


MODELS = [("Model 1: age, sex, BMI", ()),
          ("Model 2: + disease duration", ("disease_duration_years",)),
          ("Model 3: + corticosteroid courses + anti-TNF",
           ("disease_duration_years", "steroid_courses", "antitnf"))]


def table3b(pat: pd.DataFrame) -> pd.DataFrame:
    users = pat[pat["aza_user"] == 1]
    rows = []
    for pop_label, df in [("All patients (non-users = 0)", pat), ("Azathioprine users only", users)]:
        for exp_label, col, unit in [("Cumulative azathioprine dose, per 100 g", "aza_cumulative_g", 100),
                                     ("Azathioprine duration, per 12 months", "aza_months", 12)]:
            for model_label, extra in MODELS:
                r = firth_or(df, col, unit, extra)
                rows.append([pop_label, exp_label, model_label, f"{r['n']} / {r['events']}",
                             f"{r['or']:.2f} ({r['low']:.2f}-{r['high']:.2f})", f"{r['p']:.3f}"])
    return pd.DataFrame(rows, columns=[
        "Population", "Exposure metric", "Adjustment", "n / events",
        "OR for SSM >= 40 kPa (95% CI)", "p"])


def table4(pat: pd.DataFrame) -> pd.DataFrame:
    cases = pat[pat["ssm_ge_40"] == 1].sort_values("ssm", ascending=False)
    out = pd.DataFrame({
        "Case": [f"#{i}" for i in range(1, len(cases) + 1)],
        "Age (y)": cases["age"].astype("Int64").values,
        "Sex": np.where(cases["male"] == 1, "M", "F"),
        "IBD": np.where(cases["uc"] == 1, "UC", "CD"),
        "BMI (kg/m2)": cases["bmi"].round(1).values,
        "SSM (kPa)": cases["ssm"].round(1).values,
        "IQR/median (%)": cases["ssm_iqr_median"].values,
        "LSM (kPa)": cases["lsm"].round(1).values,
        "SSM/LSM": (cases["ssm"] / cases["lsm"]).round(1).values,
        "Platelets (x10^3/uL)": cases["plt"].values,
        "ALT (U/L)": cases["alt"].values,
        "GGT (U/L)": cases["ggt"].values,
        "Splenomegaly (imaging)": cases["splenomegaly"].map({0.0: "No", 1.0: "Yes"}).fillna("NA").values,
        "Thiopurine": cases["thiopurine"].map({0.0: "No", 1.0: "Yes"}).fillna("NA").values,
        "AZA mg/day": cases["aza_mg_day"].values,
        "AZA mg/kg": cases["aza_mg_kg"].round(2).values,
        "AZA months": cases["aza_months"].values,
        "AZA cumulative (g)": cases["aza_cumulative_g"].round(0).values,
        "Anti-TNF": cases["antitnf"].map({0.0: "No", 1.0: "Yes"}).fillna("NA").values,
    })
    return out


def main():
    OUT.mkdir(exist_ok=True)
    _, pat, _ref = prep.load_all()
    outputs = {
        "table1_characteristics.csv": table1(pat),
        "table2a_categorical_exposure.csv": table2a(pat),
        "table2b_continuous_exposure.csv": table2b(pat),
        "table3a_thresholds_by_category.csv": table3a(pat),
        "table3b_firth_models.csv": table3b(pat),
        "table4_marked_cases.csv": table4(pat),
    }
    for name, df in outputs.items():
        df.to_csv(OUT / name, index=False)
        print(f"wrote {name:42s} ({len(df)} rows)")


if __name__ == "__main__":
    main()

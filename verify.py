"""Reproduction check.

Recomputes the numbers reported in the manuscript and compares them with the
published values. Run ``python verify.py``; the script exits with status 1 if
any check fails.

Each check is (description, computed value, published value, tolerance).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import prep
from analysis_tables import firth_or, gmr
from firth import cochran_armitage, logit_mle, wilson_ci


def main() -> int:
    pat_att, pat, ref = prep.load_all()
    ref_att = prep.load_reference()
    users = pat[pat["aza_user"] == 1]
    checks: list[tuple[str, float, float, float]] = []

    def chk(desc, got, want, tol=0.005):
        checks.append((desc, float(got), float(want), tol))

    # --- study population -------------------------------------------------
    chk("patients with an attempted examination", len(pat_att), 310, 0)
    chk("patients analysed", len(pat), 271, 0)
    chk("reference participants examined", len(ref_att), 64, 0)
    chk("reference participants analysed", len(ref), 57, 0)
    chk("Crohn's disease", (pat["uc"] == 0).sum(), 173, 0)
    chk("ulcerative colitis", (pat["uc"] == 1).sum(), 98, 0)
    chk("men", pat["male"].sum(), 164, 0)
    chk("median age", pat["age"].median(), 38, 0)
    chk("thiopurine-exposed", pat["thiopurine"].sum(), 194, 0)
    chk("patients with a medication history", pat["thiopurine"].notna().sum(), 266, 0)
    chk("azathioprine-exposed", (pat["aza"] == 1).sum(), 193, 0)
    chk("mercaptopurine-exposed", (pat["mercaptopurine"] == 1).sum(), 3, 0)
    chk("anti-TNF exposed", (pat["antitnf"] == 1).sum(), 183, 0)
    chk("azathioprine plus anti-TNF", (pat["aza_combo"] == "AZA + anti-TNF").sum(), 150, 0)
    chk("maintenance dose 100 mg/day", (users["aza_mg_day"] == 100).sum(), 121, 0)
    chk("median weight-adjusted dose", users["aza_mg_kg"].median(), 1.39, 0.005)
    chk("median duration of exposure (months)", users["aza_months"].median(), 50, 0)
    chk("median cumulative dose (g)", users["aza_cumulative_g"].median(), 123, 0.5)
    chk("reliability subset (IQR/median <= 30%)", (pat["reliable"] == 1).sum(), 186, 0)
    chk("patients with a recorded IQR/median", pat["ssm_iqr_median"].notna().sum(), 256, 0)

    # --- distribution ------------------------------------------------------
    chk("median SSM, patients (kPa)", pat["ssm"].median(), 19.8, 0.05)
    chk("median SSM, reference (kPa)", ref["ssm"].median(), 19.3, 0.05)
    chk("reference 90th percentile", ref["ssm"].quantile(0.90), 27.9, 0.05)
    chk("reference 95th percentile", ref["ssm"].quantile(0.95), 29.6, 0.05)
    chk("reference maximum", ref["ssm"].max(), 35.5, 0.05)
    chk("SSM >= 30 kPa, patients", pat["ssm_ge_30"].sum(), 49, 0)
    chk("SSM >= 40 kPa, patients", pat["ssm_ge_40"].sum(), 20, 0)
    chk("SSM >= 40 kPa, reference", ref["ssm_ge_40"].sum(), 0, 0)
    chk("SSM >= 50 kPa, patients", pat["ssm_ge_50"].sum(), 11, 0)
    chk("SSM >= 21 kPa, patients (%)", 100 * pat["ssm_ge_21"].mean(), 43.9, 0.1)
    chk("SSM >= 21 kPa, reference (%)", 100 * ref["ssm_ge_21"].mean(), 42.1, 0.1)
    chk("upper 95% limit for 0 of 57 (%)", 100 * wilson_ci(0, len(ref))[1], 6.3, 0.1)
    both = pd.concat([pat.assign(patient=1.0), ref.assign(patient=0.0)], ignore_index=True)
    g = gmr(both, "patient", covars=("age", "male", "bmi"))
    chk("GMR patients vs reference", g["gmr"], 1.19, 0.005)
    chk("GMR lower limit", g["low"], 1.06, 0.005)
    chk("GMR upper limit", g["high"], 1.34, 0.005)

    # --- thiopurine exposure and the distribution --------------------------
    t = gmr(pat, "thiopurine")
    chk("thiopurine GMR", t["gmr"], 0.99, 0.005)
    chk("thiopurine GMR lower", t["low"], 0.87, 0.005)
    chk("thiopurine GMR upper", t["high"], 1.13, 0.005)
    rho, p = stats.spearmanr(*users[["aza_mg_day", "ssm"]].dropna().T.values)
    chk("Spearman daily dose vs SSM", rho, 0.216, 0.002)
    rho_w, _ = stats.spearmanr(*users[["aza_mg_kg", "ssm"]].dropna().T.values)
    chk("Spearman weight-adjusted dose vs SSM", rho_w, -0.034, 0.002)
    d = gmr(users, "aza_mg_day", 25)
    chk("adjusted GMR per 25 mg/day", d["gmr"], 1.05, 0.005)
    c = gmr(users, "aza_cumulative_g", 100)
    chk("adjusted GMR per 100 g", c["gmr"], 1.00, 0.005)

    # --- tail analyses -----------------------------------------------------
    grp = pat.dropna(subset=["aza_cumulative_cat4"]).groupby("aza_cumulative_cat4", observed=False)
    counts = grp.agg(n=("ssm", "size"), e=("ssm_ge_40", "sum"))
    counts = counts.loc[["none", "<= 100 g", "101-300 g", "> 300 g"]]
    for lab, (n_exp, e_exp) in zip(counts.index, [(72, 4), (73, 3), (76, 5), (42, 8)]):
        chk(f"cumulative dose category {lab}: n", counts.loc[lab, "n"], n_exp, 0)
        chk(f"cumulative dose category {lab}: events", counts.loc[lab, "e"], e_exp, 0)
    chk("trend p, all patients", cochran_armitage(counts["e"], counts["n"])[1], 0.022, 0.001)
    chk("trend p, users only", cochran_armitage(counts["e"][1:], counts["n"][1:])[1], 0.009, 0.001)

    for desc, df, col, unit, extra, want, lo, hi in [
        ("Model 1, cumulative dose", pat, "aza_cumulative_g", 100, (), 1.41, 1.14, 1.74),
        ("Model 2, cumulative dose", pat, "aza_cumulative_g", 100,
         ("disease_duration_years",), 1.36, 1.09, 1.70),
        ("Model 3, cumulative dose", pat, "aza_cumulative_g", 100,
         ("disease_duration_years", "steroid_courses", "antitnf"), 1.26, 1.00, 1.59),
        ("Model 1, users only", users, "aza_cumulative_g", 100, (), 1.51, 1.16, 1.96),
        ("Model 1, duration", pat, "aza_months", 12, (), 1.12, 1.03, 1.23),
        ("Model 2, duration", pat, "aza_months", 12, ("disease_duration_years",),
         1.10, 1.01, 1.21),
    ]:
        r = firth_or(df, col, unit, extra)
        chk(f"{desc}: OR", r["or"], want, 0.005)
        chk(f"{desc}: lower limit", r["low"], lo, 0.005)
        chk(f"{desc}: upper limit", r["high"], hi, 0.005)

    dd = pat.dropna(subset=["ssm_ge_30", "aza_cumulative_g", "age", "male", "bmi"]).copy()
    dd["aza_cumulative_g"] /= 100.0
    m30 = logit_mle(dd[["aza_cumulative_g", "age", "male", "bmi"]].astype(float),
                    dd["ssm_ge_30"].values)
    row = m30["table"].set_index("term").loc["aza_cumulative_g"]
    chk("OR per 100 g for SSM >= 30 kPa", row.OR, 1.13, 0.005)

    # --- discordance and haematology ---------------------------------------
    marked = pat[pat["ssm_ge_40"] == 1]
    chk("patients >= 40 kPa with LSM < 10 kPa", (marked["lsm"] < 10).sum(), 20, 0)
    chk("patients >= 40 kPa with LSM < 7.2 kPa", (marked["lsm"] < 7.2).sum(), 16, 0)
    chk("median LSM in marked cases", marked["lsm"].median(), 5.1, 0.05)
    chk("minimum SSM/LSM ratio in marked cases", (marked["ssm"] / marked["lsm"]).min(), 6.2, 0.05)
    chk("median SSM/LSM ratio in marked cases",
        (marked["ssm"] / marked["lsm"]).median(), 10.6, 0.05)
    chk("patients with LSM >= 10 kPa and SSM >= 40 kPa",
        ((pat["lsm"] >= 10) & (pat["ssm_ge_40"] == 1)).sum(), 0, 0)
    chk("patients with neither", ((pat["lsm"] < 10) & (pat["ssm"] < 40)).sum(), 245, 0)
    chk("thrombocytopenic patients", pat["thrombocytopenia"].sum(), 7, 0)
    chk("median platelets, patients", pat["plt"].median(), 270, 0)
    chk("median platelets, reference", ref["plt"].median(), 246, 0.5)
    chk("marked cases: thiopurine-exposed", marked["thiopurine"].sum(), 16, 0)
    chk("marked cases: men", marked["male"].sum(), 14, 0)
    chk("marked cases: median SSM", marked["ssm"].median(), 54, 0.5)
    chk("marked cases with cumulative dose > 300 g",
        (marked["aza_cumulative_g"] > 300).sum(), 8, 0)
    chk("patients with SSM >= 50 kPa who were thiopurine-exposed",
        pat.loc[pat["ssm_ge_50"] == 1, "thiopurine"].sum(), 10, 0)
    chk("patients above 75 kPa", (pat["ssm"] > 75).sum(), 4, 0)

    # --- report -------------------------------------------------------------
    failed = 0
    width = max(len(c[0]) for c in checks) + 2
    for desc, got, want, tol in checks:
        ok = abs(got - want) <= tol + 1e-9
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {desc:<{width}} computed {got:>10.3f}   "
              f"published {want:>10.3f}")
    print(f"\n{len(checks) - failed} of {len(checks)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

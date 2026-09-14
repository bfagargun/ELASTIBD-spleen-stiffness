"""Robustness analyses: Table 3C and Supplementary Table S12 (leave-one-out).

All models are Firth penalised logistic regressions for SSM >= 40 kPa per 100 g
of cumulative azathioprine (Model 1: age, sex, BMI; Model 2: additionally
disease duration). Sex is dropped automatically in sex-stratified subsets.

One row of Table 3C ("excluding examinations repeated within the session because
the first acquisition had an absolute IQR above 25 kPa", n = 6) cannot be derived
from the two SPSS files, because they already contain the repeat measurements.
It is reproduced only if an auxiliary file ``repeated_examinations.csv`` with a
single column ``row`` (0-based row positions in the patient file, held by the
corresponding author) is placed in the data directory; otherwise the row is
reported as not available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import prep
from analysis_tables import OUT, firth_or
from firth import cochran_armitage

CUM = "aza_cumulative_g"
CAT = "aza_cumulative_cat4"


def _counts(df: pd.DataFrame) -> tuple[str, float, float]:
    d = df.dropna(subset=[CAT])
    g = d.groupby(CAT, observed=False).agg(n=("ssm", "size"), e=("ssm_ge_40", "sum"))
    g = g.loc[list(d[CAT].cat.categories)]
    cells = " / ".join(f"{int(r.e)}/{int(r.n)}" for r in g.itertuples())
    p_all = cochran_armitage(g["e"], g["n"])[1]
    p_users = cochran_armitage(g["e"][1:], g["n"][1:])[1]
    return cells, p_all, p_users


def _row(label: str, df: pd.DataFrame) -> list:
    users = df[df["aza_user"] == 1]
    cells, p_all, p_users = _counts(df)
    m1 = firth_or(df, CUM, 100, ())
    m2 = firth_or(df, CUM, 100, ("disease_duration_years",))
    u1 = firth_or(users, CUM, 100, ())
    fmt = lambda r: f"{r['or']:.2f} ({r['low']:.2f}-{r['high']:.2f}); p = {r['p']:.3f}"
    return [label, cells, f"{p_all:.3g} / {p_users:.3g}", f"{m1['n']}/{m1['events']}",
            fmt(m1), fmt(m2), f"{u1['n']}/{u1['events']}", fmt(u1)]


def analysis_sets(pat: pd.DataFrame, data_dir: Path | None = None):
    d = Path(data_dir) if data_dir is not None else prep.DATA_DIR
    sets = [
        ("All patients (primary analysis)", pat),
        ("Reliability subset (IQR/median <= 30%)", pat[pat["reliable"] == 1]),
        ("Strict reliability subset (IQR/median < 25%)", pat[pat["reliable_strict"] == 1]),
        ("Excluding the measurement at the 100-kPa ceiling",
         pat[pat["ssm"] < prep.DEVICE_CEILING]),
        ("Reliability subset, excluding the ceiling measurement",
         pat[(pat["reliable"] == 1) & (pat["ssm"] < prep.DEVICE_CEILING)]),
        ("Excluding SSM > 75 kPa", pat[pat["ssm"] <= 75]),
    ]
    repeat_file = d / "repeated_examinations.csv"
    if repeat_file.exists():
        rows = pd.read_csv(repeat_file)["row"].tolist()
        keep = ~pat.reset_index(drop=True).index.isin(rows)
        sets.append(("Excluding examinations repeated for an absolute IQR > 25 kPa",
                     pat.reset_index(drop=True)[keep]))
    sets += [
        ("Excluding cumulative doses above 800 g", pat[(pat[CUM].isna()) | (pat[CUM] <= 800)]),
        ("LSM < 7.2 kPa only", pat[pat["lsm"] < prep.LSM_SIGNIFICANT_FIBROSIS]),
        ("Crohn's disease only", pat[pat["uc"] == 0]),
        ("Ulcerative colitis only", pat[pat["uc"] == 1]),
        ("Men only", pat[pat["male"] == 1]),
        ("Women only", pat[pat["male"] == 0]),
        ("BMI < 30 kg/m2", pat[pat["bmi"] < 30]),
        ("Non-drinkers only", pat[pat["alcohol_any"] == 0]),
    ]
    return sets, repeat_file.exists()


def table3c(pat: pd.DataFrame) -> pd.DataFrame:
    sets, have_repeat = analysis_sets(pat)
    rows = [_row(label, df) for label, df in sets]
    if not have_repeat:
        rows.append(["Excluding examinations repeated for an absolute IQR > 25 kPa",
                     "not available without repeated_examinations.csv", "", "", "", "", "", ""])
    # leave-one-out summary
    loo = leave_one_out(pat)
    rows.append([
        "Leave-one-out (each patient with SSM >= 40 kPa omitted in turn)", "", "", "",
        f"OR range {loo['OR (all patients)'].str.slice(0, 4).astype(float).min():.2f}-"
        f"{loo['OR (all patients)'].str.slice(0, 4).astype(float).max():.2f}; "
        f"p < 0.05 in {int((loo['p'] < 0.05).sum())} of {len(loo)} fits "
        f"(trend p < 0.05 in {int((loo['trend p'] < 0.05).sum())} of {len(loo)})",
        "", "",
        f"OR range {loo['OR (users only)'].str.slice(0, 4).astype(float).min():.2f}-"
        f"{loo['OR (users only)'].str.slice(0, 4).astype(float).max():.2f}; "
        f"p < 0.05 in {int((loo['p (users)'] < 0.05).sum())} of {len(loo)}"])
    return pd.DataFrame(rows, columns=[
        "Analysis set", "SSM >= 40 kPa, events/n by cumulative-dose category "
        "(none / <= 100 / 101-300 / > 300 g)", "Trend p (all / users)", "n / events",
        "OR per 100 g, Model 1 (95% CI); p", "OR per 100 g, Model 2 (95% CI); p",
        "n / events, users", "OR per 100 g, users only, Model 1 (95% CI); p"])


def leave_one_out(pat: pd.DataFrame) -> pd.DataFrame:
    cases = pat[pat["ssm_ge_40"] == 1].sort_values("ssm", ascending=False)
    rows = []
    for idx, case in cases.iterrows():
        sub = pat.drop(index=idx)
        users = sub[sub["aza_user"] == 1]
        m1 = firth_or(sub, CUM, 100, ())
        u1 = firth_or(users, CUM, 100, ())
        _, p_all, _ = _counts(sub)
        rows.append([
            round(case["ssm"], 1), round(case[CUM], 0) if pd.notna(case[CUM]) else np.nan,
            f"{m1['or']:.2f} ({m1['low']:.2f}-{m1['high']:.2f})", m1["p"], p_all,
            f"{u1['or']:.2f} ({u1['low']:.2f}-{u1['high']:.2f})", u1["p"]])
    return pd.DataFrame(rows, columns=[
        "Omitted patient: SSM (kPa)", "Omitted patient: cumulative azathioprine dose (g)",
        "OR (all patients)", "p", "trend p", "OR (users only)", "p (users)"])


def main():
    OUT.mkdir(exist_ok=True)
    _, pat, _ref = prep.load_all()
    t3c = table3c(pat)
    loo = leave_one_out(pat)
    t3c.to_csv(OUT / "table3c_robustness.csv", index=False)
    loo.to_csv(OUT / "tableS12_leave_one_out.csv", index=False)
    print(f"wrote table3c_robustness.csv               ({len(t3c)} rows)")
    print(f"wrote tableS12_leave_one_out.csv           ({len(loo)} rows)")


if __name__ == "__main__":
    main()

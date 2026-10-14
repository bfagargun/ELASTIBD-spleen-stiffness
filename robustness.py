"""Robustness analyses: Supplementary Tables S13 (robustness) and S12 (leave-one-out).

All models are Firth penalised logistic regressions for SSM >= 40 kPa per 100 g
of cumulative thiopurine dose in azathioprine equivalents (mercaptopurine dose
x 2.08), with profile penalised-likelihood 95 % confidence intervals and
penalised likelihood-ratio p-values (Model 1: age, sex, BMI; Model 2:
additionally disease duration). Sex is dropped automatically in sex-stratified
subsets.

One row of Supplementary Table S13 ("excluding examinations repeated within the
session", n = 6) cannot be derived from the two SPSS files, because they
already contain the repeat measurements. It is reproduced only if an auxiliary
file ``repeated_examinations.csv`` with a single column ``row`` (0-based row
positions among the analysed patients, held by the corresponding author) is
placed in the data directory; otherwise the row is reported as not available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import prep
from analysis_tables import OUT, firth_or, fmt_p
from firth import cochran_armitage

CUM = "thio_cumulative_g"
CAT = "thio_cumulative_cat4"
USER = "thio_user"
REPEAT_LABEL = "Excluding examinations repeated within the session (n = 6)"


def _p(p: float) -> str:
    s = fmt_p(p)
    return f"p {s}" if s.startswith("<") else f"p = {s}"


def _fmt(r: dict) -> str:
    return f"{r['or']:.2f} ({r['low']:.2f}-{r['high']:.2f}); {_p(r['p'])}"


def _counts(df: pd.DataFrame, cat: str = CAT) -> tuple[str, float, float]:
    d = df.dropna(subset=[cat])
    g = d.groupby(cat, observed=False).agg(n=("ssm", "size"), e=("ssm_ge_40", "sum"))
    g = g.loc[list(d[cat].cat.categories)]
    cells = " / ".join(f"{int(r.e)}/{int(r.n)}" for r in g.itertuples())
    p_all = cochran_armitage(g["e"], g["n"])[1]
    p_users = cochran_armitage(g["e"][1:], g["n"][1:])[1]
    return cells, p_all, p_users


def _row(label: str, df: pd.DataFrame, cum: str = CUM, cat: str = CAT, user: str = USER,
         extra: tuple = ()) -> list:
    cells, p_all, p_users = _counts(df, cat)
    m1 = firth_or(df, cum, 100, tuple(extra))
    m2 = firth_or(df, cum, 100, ("disease_duration_years",) + tuple(extra))
    u1 = firth_or(df[df[user] == 1], cum, 100, tuple(extra))
    return [label, cells, f"{fmt_p(p_all)} / {fmt_p(p_users)}", f"{m1['n']}/{m1['events']}",
            _fmt(m1), _fmt(m2), f"{u1['n']}/{u1['events']}", _fmt(u1)]


def analysis_sets(pat: pd.DataFrame, data_dir: Path | None = None):
    """Subsets of the analysed patients; the exposure definition is unchanged."""
    d = Path(data_dir) if data_dir is not None else prep.DATA_DIR
    sets = [
        ("All patients (primary analysis)", pat),
        ("Reliability subset (IQR/median <= 30%)", pat[pat["reliable"] == 1]),
        ("Strict reliability subset (IQR/median < 25%)", pat[pat["reliable_strict"] == 1]),
        ("Excluding the measurement at the 100-kPa ceiling",
         pat[pat["ssm"] < prep.DEVICE_CEILING]),
        ("Reliability subset, excluding the ceiling measurement",
         pat[(pat["reliable"] == 1) & (pat["ssm"] < prep.DEVICE_CEILING)]),
        ("Excluding SSM > 75 kPa (n = 4)", pat[pat["ssm"] <= 75]),
    ]
    repeat_file = d / "repeated_examinations.csv"
    if repeat_file.exists():
        rows = pd.read_csv(repeat_file)["row"].tolist()
        keep = ~pat.reset_index(drop=True).index.isin(rows)
        sets.append((REPEAT_LABEL, pat.reset_index(drop=True)[keep]))
    sets += [
        ("Excluding the two highest cumulative doses (864 and 870 g)",
         pat[(pat[CUM].isna()) | (pat[CUM] <= 800)]),
        ("LSM < 7.2 kPa only", pat[pat["lsm"] < prep.LSM_SIGNIFICANT_FIBROSIS]),
        ("Crohn's disease only", pat[pat["uc"] == 0]),
        ("Ulcerative colitis only", pat[pat["uc"] == 1]),
        ("Men only", pat[pat["male"] == 1]),
        ("Women only", pat[pat["male"] == 0]),
        ("BMI < 30 kg/m2", pat[pat["bmi"] < 30]),
        ("Non-drinkers only", pat[pat["alcohol_any"] == 0]),
    ]
    return sets, repeat_file.exists()


def tableS13(pat: pd.DataFrame) -> pd.DataFrame:
    sets, have_repeat = analysis_sets(pat)
    rows = []
    for label, df in sets:
        rows.append(_row(label, df))
        if label.startswith("Excluding SSM > 75") and not have_repeat:
            rows.append([REPEAT_LABEL, "not available without repeated_examinations.csv",
                         "", "", "", "", "", ""])
    # additional adjustment (added to Models 1 and 2)
    rows.append(_row("Additional adjustment for LSM", pat, extra=("lsm",)))
    rows.append(_row("Additional adjustment for alcohol use (patients with recorded alcohol use)",
                     pat[pat["alcohol_any"].notna()], extra=("alcohol_any",)))
    # alternative definitions of exposure
    rows.append(_row("Azathioprine dose alone (mercaptopurine not counted)", pat,
                     cum="aza_cumulative_g", cat="aza_cumulative_cat4", user="aza_user"))
    rows.append(_row("Excluding the three mercaptopurine-exposed patients",
                     pat[pat["mercaptopurine"] != 1]))
    rows.append(_row("Mercaptopurine converted with the molar mass ratio only (x 1.82)",
                     prep.add_thiopurine_equivalents(pat, factor=prep.MP_TO_AZA_MOLAR)))
    # leave-one-out summary
    loo = leave_one_out(pat)
    rows.append([
        "Leave-one-out (omission of each of the "
        f"{len(loo)} patients with SSM >= 40 kPa in turn)", "-", "-", "-",
        f"OR range {loo['_or'].min():.2f}-{loo['_or'].max():.2f}; "
        f"p < 0.05 in {int((loo['_p'] < 0.05).sum())} of {len(loo)} fits "
        f"(trend p < 0.05 in {int((loo['_trend'] < 0.05).sum())} of {len(loo)})",
        "-", "-",
        f"OR range {loo['_or_u'].min():.2f}-{loo['_or_u'].max():.2f}; "
        f"p < 0.05 in {int((loo['_p_u'] < 0.05).sum())} of {len(loo)}"])
    return pd.DataFrame(rows, columns=[
        "Analysis set", "SSM >= 40 kPa, events/n by cumulative-dose category "
        "(none / <= 100 / 101-300 / > 300 g)", "Trend p (all / users)", "n / events",
        "OR per 100 g, Model 1 (95% CI); p", "OR per 100 g, Model 2 (95% CI); p",
        "n / events, users", "OR per 100 g, users only, Model 1 (95% CI); p"])


def leave_one_out(pat: pd.DataFrame) -> pd.DataFrame:
    """Model 1 refitted after omitting each patient with SSM >= 40 kPa in turn.

    Columns starting with an underscore hold the unrounded values used for the
    summary row of Supplementary Table S13 and for Supplementary Figure S4.
    """
    cases = pat[pat["ssm_ge_40"] == 1].sort_values("ssm", ascending=False)
    rows = []
    for idx, case in cases.iterrows():
        sub = pat.drop(index=idx)
        m1 = firth_or(sub, CUM, 100, ())
        u1 = firth_or(sub[sub[USER] == 1], CUM, 100, ())
        _, p_all, _ = _counts(sub)
        rows.append([
            round(case["ssm"], 1), round(case[CUM]) if pd.notna(case[CUM]) else np.nan,
            f"{m1['or']:.2f} ({m1['low']:.2f}-{m1['high']:.2f})", fmt_p(m1["p"]), fmt_p(p_all),
            f"{u1['or']:.2f} ({u1['low']:.2f}-{u1['high']:.2f})", fmt_p(u1["p"]),
            m1["or"], m1["low"], m1["high"], m1["p"], p_all, u1["or"], u1["low"], u1["high"],
            u1["p"]])
    return pd.DataFrame(rows, columns=[
        "Omitted patient: SSM (kPa)", "Omitted patient: cumulative thiopurine dose (g)",
        "OR per 100 g, all patients, Model 1 (95% CI)", "p", "Cochran-Armitage trend p (all)",
        "OR per 100 g, users only, Model 1 (95% CI)", "p (users)",
        "_or", "_low", "_high", "_p", "_trend", "_or_u", "_low_u", "_high_u", "_p_u"])


def main():
    OUT.mkdir(exist_ok=True)
    _, pat, _ref = prep.load_all()
    s13 = tableS13(pat)
    loo = leave_one_out(pat)
    s13.to_csv(OUT / "tableS13_robustness.csv", index=False)
    loo.loc[:, ~loo.columns.str.startswith("_")].to_csv(OUT / "tableS12_leave_one_out.csv",
                                                        index=False)
    print(f"wrote tableS13_robustness.csv              ({len(s13)} rows)")
    print(f"wrote tableS12_leave_one_out.csv           ({len(loo)} rows)")


if __name__ == "__main__":
    main()

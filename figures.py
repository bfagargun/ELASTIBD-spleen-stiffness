"""Figures 1-3 and Supplementary Figures S1-S4.

Each figure is written to ``outputs/figures/`` as PNG (300 dpi) and PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import prep
from analysis_supplementary import SSM_BINS, _quantreg, _ssm_category
from analysis_tables import firth_or, gmr
from firth import cochran_armitage, wilson_ci

_HERE = Path(__file__).resolve().parent
FIGDIR = (_HERE.parent if _HERE.name == "code" else _HERE) / "outputs" / "figures"
BLUE, ORANGE, GREY, GREEN = "#3b6ea5", "#d1701a", "#8a8a8a", "#2e7d5b"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.8,
                     "savefig.facecolor": "white"})


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote figures/{name}.png and .pdf")


def _despine(ax, keep=("left", "bottom")):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(side in keep)


# ---------------------------------------------------------------------------
def figure1(pat_att, pat, ref_att, ref):
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.axis("off")
    boxes = [
        (0.25, 0.88, f"Spleen examination attempted\nin {len(pat_att)} patients with IBD"),
        (0.25, 0.64, f"Valid spleen stiffness measurement\nn = {len(pat)} (primary analysis set)"),
        (0.25, 0.40, f"Reliability subset (IQR/median <= 30%)\nn = {int((pat['reliable'] == 1).sum())}"),
        (0.25, 0.16, f"Markedly elevated SSM (>= 40 kPa)\nn = {int(pat['ssm_ge_40'].sum())}"),
        (0.78, 0.88, f"Healthy volunteers\nexamined: n = {len(ref_att)}"),
        (0.78, 0.64, f"Reference sample with a valid\nmeasurement: n = {len(ref)}"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#333333"))
    for x, y0, y1 in [(0.25, 0.82, 0.71), (0.25, 0.58, 0.47), (0.25, 0.34, 0.23),
                      (0.78, 0.82, 0.71)]:
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9))
    n_fail = len(pat_att) - len(pat)
    ax.annotate("", xy=(0.50, 0.765), xytext=(0.27, 0.765),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9))
    ax.text(0.505, 0.765, f"Unsuccessful examination: n = {n_fail}", fontsize=7.5, va="center",
            color="#333333")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1)
    _save(fig, "Figure1_flow_chart")


# ---------------------------------------------------------------------------
def _violin(ax, groups, labels, color):
    data = [g.dropna().values for g in groups]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.35)
        pc.set_edgecolor("none")
    for i, d in enumerate(data, start=1):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color="#333333", lw=2.4)
        ax.plot(i, med, "o", color="white", mec="#333333", ms=4, zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7.5)
    _despine(ax)


def _proportion_panel(ax, pat, cat_col, cat_labels, title):
    d = pat.dropna(subset=[cat_col])
    cats = list(d[cat_col].cat.categories)
    width = 0.38
    for k, (col, colour, label) in enumerate(
            [("ssm_ge_30", BLUE, "SSM >= 30 kPa"), ("ssm_ge_40", ORANGE, "SSM >= 40 kPa")]):
        ev, tot = [], []
        for i, c in enumerate(cats):
            g = d[d[cat_col] == c]
            k_ev, n = int(g[col].sum()), len(g)
            ev.append(k_ev)
            tot.append(n)
            lo, hi = wilson_ci(k_ev, n)
            x = i + (k - 0.5) * width
            ax.bar(x, 100 * k_ev / n, width=width * 0.9, color=colour, alpha=0.85,
                   label=label if i == 0 else None)
            ax.errorbar(x, 100 * k_ev / n, yerr=[[100 * (k_ev / n - lo)], [100 * (hi - k_ev / n)]],
                        color="#333333", lw=0.8, capsize=2)
            ax.text(x, 100 * hi + 1.2, f"{k_ev}/{n}", ha="center", fontsize=6.2)
        p = cochran_armitage(ev, tot)[1]
        ax.text(0.02, 0.95 - 0.09 * k, f"trend p = {p:.3f}", transform=ax.transAxes,
                fontsize=7, color=colour)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([cat_labels.get(c, c) for c in cats], fontsize=7.5)
    ax.set_ylabel("Patients (%)")
    ax.set_ylim(0, max(45, ax.get_ylim()[1] * 1.25))
    ax.set_title(title, fontsize=8.5, loc="left")
    _despine(ax)


def figure2(pat):
    fig = plt.figure(figsize=(7.5, 8.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    _violin(ax, [pat.loc[pat["thiopurine"] == 0, "ssm"], pat.loc[pat["thiopurine"] == 1, "ssm"]],
            ["Not exposed", "Exposed"], BLUE)
    ax.set_yscale("log")
    ax.set_ylabel("Spleen stiffness (kPa, log scale)")
    ax.set_title("A  Thiopurine exposure", fontsize=8.5, loc="left")

    ax = fig.add_subplot(gs[0, 1])
    cats = list(pat["aza_cumulative_cat4"].cat.categories)
    _violin(ax, [pat.loc[pat["aza_cumulative_cat4"] == c, "ssm"] for c in cats],
            ["None", "<= 100 g", "101-300 g", "> 300 g"], BLUE)
    ax.set_yscale("log")
    ax.set_ylabel("Spleen stiffness (kPa, log scale)")
    ax.set_title("B  Cumulative azathioprine dose", fontsize=8.5, loc="left")

    ax = fig.add_subplot(gs[1, 0])
    _proportion_panel(ax, pat, "aza_cumulative_cat4",
                      {"none": "None", "<= 100 g": "<= 100 g", "101-300 g": "101-300 g",
                       "> 300 g": "> 300 g"}, "C  By cumulative dose")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.30, 1.0))

    ax = fig.add_subplot(gs[1, 1])
    _proportion_panel(ax, pat, "aza_duration_cat4",
                      {"none": "None", "<= 12 months": "<= 12 mo", "13-48 months": "13-48 mo",
                       "> 48 months": "> 48 mo"}, "D  By duration of exposure")

    # E: forest plot
    ax = fig.add_subplot(gs[2, :])
    users = pat[pat["aza_user"] == 1]
    rows = [
        ("AZA cumulative dose, per 100 g (Model 1)", pat, "aza_cumulative_g", 100, (), ORANGE),
        ("   + disease duration", pat, "aza_cumulative_g", 100, ("disease_duration_years",), ORANGE),
        ("   + steroids and anti-TNF", pat, "aza_cumulative_g", 100,
         ("disease_duration_years", "steroid_courses", "antitnf"), ORANGE),
        ("AZA duration, per 12 months (Model 1)", pat, "aza_months", 12, (), ORANGE),
        ("AZA daily dose, per 25 mg (Model 1)", pat, "aza_mg_day", 25, (), ORANGE),
        ("Anti-TNF duration, per 12 months", pat, "antitnf_months", 12, (), GREY),
        ("Disease duration, per year", pat, "disease_duration_years", 1, (), GREY),
        ("Corticosteroid courses, per course", pat, "steroid_courses", 1, (), GREY),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (label, df, col, unit, extra, colour) in zip(ys, rows):
        r = firth_or(df, col, unit, extra)
        ax.plot([r["low"], r["high"]], [y, y], color=colour, lw=1.2)
        ax.plot(r["or"], y, "o", color=colour, ms=4.5)
        ax.text(6.2, y, f"{r['or']:.2f} ({r['low']:.2f}-{r['high']:.2f})", fontsize=7,
                va="center")
    ax.axvline(1, color="#666666", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 6.0)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel("Adjusted odds ratio for SSM >= 40 kPa (log scale)")
    ax.set_title("E  Adjusted odds ratios (Firth penalised logistic regression)",
                 fontsize=8.5, loc="left")
    _despine(ax)
    _save(fig, "Figure2_thiopurine_exposure")


# ---------------------------------------------------------------------------
def figure3(pat, ref):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    ax = axes[0]
    ax.add_patch(Rectangle((2.5, prep.SSM_MARKED), prep.LSM_CACLD - 2.5, 120 - prep.SSM_MARKED,
                           facecolor=ORANGE, alpha=0.10, zorder=0))
    for mask, colour, label in [
        ((pat["thiopurine"] == 1), BLUE, "IBD, thiopurine-exposed"),
        ((pat["thiopurine"] == 0), GREEN, "IBD, not exposed"),
    ]:
        g = pat[mask]
        ax.scatter(g["lsm"], g["ssm"], s=12, alpha=0.65, color=colour, label=label,
                   edgecolors="none")
    ax.scatter(ref["lsm"], ref["ssm"], s=12, alpha=0.5, color=GREY, marker="^",
               label="Reference sample", edgecolors="none")
    low_plt = pat[pat["thrombocytopenia"] == 1]
    ax.scatter(low_plt["lsm"], low_plt["ssm"], s=55, facecolors="none", edgecolors="#333333",
               lw=0.8, label="Platelets < 150 x10$^3$/uL")
    idx = pat[(pat["ssm"] > 85) & (pat["ssm"] < 87)]
    if len(idx):
        ax.annotate("index case\n(grade II varices)", xy=(idx["lsm"].iloc[0], idx["ssm"].iloc[0]),
                    xytext=(11.0, 72.0), fontsize=6.5, ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#333333"))
    ax.axhline(prep.SSM_MARKED, color=ORANGE, lw=0.8, ls="--")
    ax.axvline(prep.LSM_CACLD, color="#666666", lw=0.8, ls="--")
    n_disc = int(((pat["ssm"] >= 40) & (pat["lsm"] < 10)).sum())
    n_both = int(((pat["ssm"] >= 40) & (pat["lsm"] >= 10)).sum())
    n_lsm = int(((pat["ssm"] < 40) & (pat["lsm"] >= 10)).sum())
    n_none = int(((pat["ssm"] < 40) & (pat["lsm"] < 10)).sum())
    ax.text(0.05, 0.93, f"n = {n_disc}", transform=ax.transAxes, fontsize=7.5, color=ORANGE)
    ax.text(0.88, 0.93, f"n = {n_both}", transform=ax.transAxes, fontsize=7.5)
    ax.text(0.88, 0.06, f"n = {n_lsm}", transform=ax.transAxes, fontsize=7.5)
    ax.text(0.05, 0.06, f"n = {n_none}", transform=ax.transAxes, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Liver stiffness (kPa, log scale)")
    ax.set_ylabel("Spleen stiffness (kPa, log scale)")
    ax.set_title("A  Spleen versus liver stiffness", fontsize=8.5, loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.45, -0.20),
              ncol=2)
    _despine(ax)

    ax = axes[1]
    d = pat.assign(cat=pat["ssm"].map(_ssm_category))
    labels = [lab for _, _, lab in SSM_BINS]
    _violin(ax, [d.loc[d["cat"] == lab, "plt"] for lab in labels], labels, BLUE)
    ax.axhline(150, color=ORANGE, lw=0.8, ls="--")
    ax.set_ylabel("Platelet count (x10$^3$/uL)")
    ax.set_xlabel("Spleen stiffness category (kPa)")
    ax.set_title("B  Platelet count across spleen-stiffness categories", fontsize=8.5, loc="left")
    fig.tight_layout()
    _save(fig, "Figure3_discordance")


# ---------------------------------------------------------------------------
def figure_s1(pat, ref):
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.2))
    ax = axes[0]
    _violin(ax, [ref["ssm"], pat["ssm"]], ["Reference", "IBD"], BLUE)
    for t, c in [(21, GREY), (30, BLUE), (40, ORANGE), (50, "#8e44ad")]:
        ax.axhline(t, color=c, lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("Spleen stiffness (kPa, log scale)")
    g = gmr(pd.concat([pat.assign(patient=1.0), ref.assign(patient=0.0)], ignore_index=True),
            "patient", covars=("age", "male", "bmi"))
    ax.set_title(f"A  Adjusted GMR {g['gmr']:.2f} ({g['low']:.2f}-{g['high']:.2f})",
                 fontsize=8.5, loc="left")

    ax = axes[1]
    thresholds = [21, 30, 40, 50]
    width = 0.38
    for k, (df, colour, label) in enumerate([(ref, GREY, "Reference"), (pat, BLUE, "IBD")]):
        for i, t in enumerate(thresholds):
            kk, n = int((df["ssm"] >= t).sum()), len(df)
            lo, hi = wilson_ci(kk, n)
            x = i + (k - 0.5) * width
            ax.bar(x, 100 * kk / n, width=width * 0.9, color=colour,
                   label=label if i == 0 else None)
            ax.errorbar(x, 100 * kk / n, yerr=[[100 * (kk / n - lo)], [100 * (hi - kk / n)]],
                        color="#333333", lw=0.8, capsize=2)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f">= {t}" for t in thresholds])
    ax.set_xlabel("Spleen stiffness threshold (kPa)")
    ax.set_ylabel("Participants (%)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("B  Proportions above thresholds", fontsize=8.5, loc="left")
    _despine(ax)

    ax = axes[2]
    for df, colour, label in [(ref, GREY, "Reference"), (pat, BLUE, "IBD")]:
        x = np.sort(df["ssm"].dropna().values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), color=colour, lw=1.2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Spleen stiffness (kPa, log scale)")
    ax.set_ylabel("Cumulative proportion")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("C  Cumulative distribution", fontsize=8.5, loc="left")
    _despine(ax)
    fig.tight_layout()
    _save(fig, "SupplFigureS1_IBD_vs_reference")


def figure_s2(pat):
    users = pat[pat["aza_user"] == 1]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9))
    pairs = [("aza_mg_day", "ssm", "Azathioprine dose (mg/day)", "SSM (kPa)", "A"),
             ("aza_mg_kg", "ssm", "Azathioprine dose (mg/kg/day)", "SSM (kPa)", "B"),
             ("aza_mg_day", "weight", "Azathioprine dose (mg/day)", "Body weight (kg)", "C")]
    from scipy import stats as st
    for ax, (x, y, xl, yl, panel) in zip(axes, pairs):
        d = users[[x, y]].dropna()
        ax.scatter(d[x], d[y], s=11, alpha=0.55, color=BLUE, edgecolors="none")
        rho, p = st.spearmanr(d[x], d[y])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{panel}  rho = {rho:+.2f} (p = {p:.3g})", fontsize=8.5, loc="left")
        _despine(ax)
    fig.tight_layout()
    _save(fig, "SupplFigureS2_dose_weight")


def figure_s3(pat):
    users = pat[pat["aza_user"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, (res, title) in zip(axes, [(_quantreg(pat, "thiopurine"), "A  Thiopurine exposure"),
                                       (_quantreg(users, "aza_mg_day", unit=25),
                                        "B  AZA daily dose (per 25 mg)")]):
        q = [r["quantile"] * 100 for r in res]
        est = [r["ratio"] for r in res]
        lo = [r["ratio"] - r["low"] for r in res]
        hi = [r["high"] - r["ratio"] for r in res]
        ax.errorbar(q, est, yerr=[lo, hi], fmt="o", color=BLUE, ms=4, lw=1.1, capsize=3)
        ax.axhline(1, color="#666666", lw=0.8, ls="--")
        ax.set_xticks(q)
        ax.set_xlabel("Percentile of the SSM distribution")
        ax.set_ylabel("Ratio (95% CI)")
        ax.set_title(title, fontsize=8.5, loc="left")
        _despine(ax)
    fig.tight_layout()
    _save(fig, "SupplFigureS3_quantile_regression")


def figure_s4(pat):
    from robustness import leave_one_out
    loo = leave_one_out(pat)
    est = loo["OR (all patients)"].str.extract(r"^([\d.]+) \(([\d.]+)-([\d.]+)\)").astype(float)
    full = firth_or(pat, "aza_cumulative_g", 100, ())
    ys = np.arange(len(loo))[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    for y, (_, r) in zip(ys, est.iterrows()):
        ax.plot([r[1], r[2]], [y, y], color=ORANGE, lw=1.1)
        ax.plot(r[0], y, "o", color=ORANGE, ms=4)
    ax.axvline(1, color="#666666", lw=0.8, ls="--")
    ax.axvline(full["or"], color="#333333", lw=0.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{s:.1f} kPa, {int(c)} g" for s, c in zip(
        loo["Omitted patient: SSM (kPa)"],
        loo["Omitted patient: cumulative azathioprine dose (g)"])], fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2])
    ax.set_xticklabels(["1", "1.5", "2"])
    ax.set_xlabel("Adjusted OR per 100 g of cumulative azathioprine (log scale)")
    ax.set_title(f"Leave-one-out analysis (full-data OR {full['or']:.2f})", fontsize=8.5,
                 loc="left")
    _despine(ax)
    fig.tight_layout()
    _save(fig, "SupplFigureS4_leave_one_out")


def main():
    pat_att, pat, ref = prep.load_all()
    ref_att = prep.load_reference()
    figure1(pat_att, pat, ref_att, ref)
    figure2(pat)
    figure3(pat, ref)
    figure_s1(pat, ref)
    figure_s2(pat)
    figure_s3(pat)
    figure_s4(pat)


if __name__ == "__main__":
    main()

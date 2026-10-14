"""Figures 1-3 and Supplementary Figures S1-S4.

Each figure is written to ``outputs/figures/`` as PNG (300 dpi) and PDF.
Figures 1-3 and Supplementary Figures S3-S4 are the published versions;
Supplementary Figures S1-S2 are simplified renderings of the published panels.
Figure 1 is set in Inter when that font is installed (as published) and in
DejaVu Sans otherwise; all other figures use DejaVu Sans.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy import stats

import prep
from analysis_supplementary import _quantreg
from analysis_tables import firth_or, fmt_p, gmr
from firth import cochran_armitage, logit_mle, wilson_ci

_HERE = Path(__file__).resolve().parent
FIGDIR = (_HERE.parent if _HERE.name == "code" else _HERE) / "outputs" / "figures"
BLUE, ORANGE, GREY = "#3b6ea5", "#d1701a", "#8a8a8a"
BASE_RC = {"font.family": "DejaVu Sans", "font.size": 10, "savefig.facecolor": "white"}

M1 = ()
M2 = ("disease_duration_years",)
M3 = ("disease_duration_years", "steroid_courses", "antitnf")


def _save(fig, name, facecolor="white", tight=False):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    kw = dict(facecolor=facecolor)
    if tight:
        kw["bbox_inches"] = "tight"
    fig.savefig(FIGDIR / f"{name}.png", dpi=300, **kw)
    fig.savefig(FIGDIR / f"{name}.pdf", metadata={"CreationDate": None}, **kw)
    plt.close(fig)
    print(f"wrote figures/{name}.png and .pdf")


def _despine(ax, keep=("left", "bottom")):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(side in keep)


def _pstr(p):
    s = fmt_p(p)
    return f"p {s}" if s.startswith("<") else f"p = {s}"


def _logit30(df, exposure, unit, extra=()):
    """Conventional logistic regression for SSM >= 30 kPa (Wald 95 % CI)."""
    cols = [exposure, "age", "male", "bmi"] + list(extra)
    d = df.dropna(subset=["ssm_ge_30"] + cols).copy()
    cols = [c for c in cols if c == exposure or d[c].nunique(dropna=True) > 1]
    d[exposure] = d[exposure] / unit
    m = logit_mle(d[cols].astype(float), d["ssm_ge_30"].values)
    t = m["table"].set_index("term").loc[exposure]
    return dict(or_=float(t.OR), low=float(t.ci_low), high=float(t.ci_high))


# ---------------------------------------------------------------------------
# Figure 1 - flow chart
# ---------------------------------------------------------------------------
def figure1(pat_att, pat, ref_att, ref):
    have_inter = "Inter" in {f.name for f in font_manager.fontManager.ttflist}
    rc = {"font.family": ["Inter", "DejaVu Sans"] if have_inter else "DejaVu Sans"}
    BG, FILL, GREYF, EDGE, TXT = "#FCFAF6", "#FAE7C3", "#D7D5C7", "#3B3428", "#2A2621"
    W, H, FS = 1600, 1291, 10.3
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(10.09, 8.14), facecolor=BG)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        ax.set_facecolor(BG)

        def box(x0, y0, x1, y1, text, grey=False):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=GREYF if grey else FILL,
                                   edgecolor=EDGE, lw=1.3, zorder=2))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=FS,
                    color=TXT, linespacing=1.42, zorder=3)

        def arrow(x0, y0, x1, y1):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                         lw=1.3, color=EDGE, zorder=1, shrinkA=0, shrinkB=0))

        def line(xs, ys):
            ax.plot(xs, ys, color=EDGE, lw=1.3, zorder=1, solid_capstyle="butt")

        n_all, n_att, n_an = prep.n_cohort(), len(pat_att), len(pat)
        n_doc = int(pat["thiopurine"].notna().sum())
        n_cum = int(pat["thio_cumulative_g"].notna().sum())
        n_rel = int((pat["reliable"] == 1).sum())
        n_rel_no = int((pat["reliable"] == 0).sum())
        n_rel_na = int(pat["reliable"].isna().sum())
        n_ev = int(pat["ssm_ge_40"].sum())
        n_ev_rel = int(((pat["ssm_ge_40"] == 1) & (pat["reliable"] == 1)).sum())
        no_value = ref_att["ssm"].isna()
        n_ref_failed = int((no_value & (ref_att["ssm_success"] != 1)).sum())
        n_ref_unrecorded = int((no_value & (ref_att["ssm_success"] == 1)).sum())

        box(78, 42, 800, 155, f"ELASTIBD baseline cohort\n{n_all} adults with inflammatory bowel disease")
        line([438, 438], [155, 211])
        arrow(438, 211, 438, 265)
        line([438, 860], [211, 211])
        arrow(860, 211, 891, 211)
        box(893, 160, 1553, 262, f"Excluded (n = {n_all - n_att})\n• No spleen examination recorded",
            grey=True)
        box(78, 267, 800, 393,
            f"Spleen-dedicated 100-Hz spleen stiffness\nmeasurement attempted\nn = {n_att}")
        line([438, 438], [393, 469])
        arrow(438, 469, 438, 543)
        line([438, 860], [469, 469])
        arrow(860, 469, 891, 469)
        box(893, 418, 1553, 520, f"Excluded (n = {n_att - n_an})\n• No valid measurement obtained",
            grey=True)
        box(78, 545, 800, 720,
            f"Successful measurement: primary analysis set\nn = {n_an}\n"
            f"thiopurine exposure documented: n = {n_doc}\n"
            f"cumulative thiopurine dose available: n = {n_cum}")
        line([438, 438], [720, 872])
        line([250, 627], [872, 872])
        arrow(250, 872, 250, 945)
        arrow(627, 872, 627, 945)
        box(78, 947, 423, 1198,
            f"Reliability subset\n(IQR/median ≤ 30%)\nn = {n_rel}\nIQR/median > 30%: n = {n_rel_no}\n"
            f"not recorded: n = {n_rel_na}")
        box(455, 947, 800, 1198,
            f"Markedly elevated spleen\nstiffness (≥ 40 kPa)\nn = {n_ev} ({100 * n_ev / n_an:.1f}%)\n"
            f"all with LSM < 10 kPa\n{n_ev_rel} of the {n_ev} are in the\nreliability subset")
        box(893, 683, 1553, 796,
            "Reference volunteers, ELASTIBD cohort\nnot pairwise matched to the spleen subset")
        arrow(1223, 796, 1223, 870)
        box(893, 872, 1553, 1022,
            f"Spleen examination attempted\nn = {len(ref_att)}\n"
            f"no valid value: n = {n_ref_failed}; value not recorded: n = {n_ref_unrecorded}")
        arrow(1223, 1022, 1223, 1083)
        box(893, 1085, 1553, 1198,
            f"Successful measurement: reference sample\nn = {len(ref)} (threshold derivation)")
        _save(fig, "Figure1_flow_chart", facecolor=BG)


# ---------------------------------------------------------------------------
# Figure 2 - thiopurine exposure and spleen stiffness
# ---------------------------------------------------------------------------
def figure2(pat):
    rc = dict(BASE_RC, **{"axes.linewidth": 1.0, "xtick.major.width": 1.0,
                          "ytick.major.width": 1.0})
    BLUEPT, REDPT = "#6693CA", "#CA584D"
    LBLUE, ORANGE2, EBAR, GREYC = "#A8C6E8", "#E08A1E", "#555555", "#7F7F7F"
    rng = np.random.default_rng(20260914)
    YT = [5, 10, 20, 30, 40, 50, 75, 100]
    thio_users = pat[pat["thio_user"] == 1]

    def strip(ax, groups, colors, alphas, labels):
        for i, (g, c, a) in enumerate(zip(groups, colors, alphas)):
            g = np.asarray(g)
            x = i + rng.uniform(-0.17, 0.17, len(g))
            ax.scatter(x, g, s=26, color=c, alpha=a, edgecolors=c, linewidths=0.8, zorder=2)
            q1, med, q3 = np.percentile(g, [25, 50, 75])
            ax.plot([i - 0.3, i + 0.3], [med, med], color="black", lw=4, zorder=3,
                    solid_capstyle="butt")
            ax.plot([i, i], [q1, q3], color="black", lw=2.2, zorder=3)
            for q in (q1, q3):
                ax.plot([i - 0.14, i + 0.14], [q, q], color="black", lw=2.2, zorder=3)
        ax.axhline(30, color="#999999", lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.axhline(40, color="#555555", lw=1.1, ls=(0, (6, 2, 1, 2)), zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(4.5, 135)
        ax.set_yticks(YT)
        ax.set_yticklabels([str(t) for t in YT])
        ax.minorticks_off()
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, fontsize=9.2)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylabel("Spleen stiffness (kPa, log scale)", fontsize=10.5)
        ax.tick_params(axis="y", labelsize=10.5)
        _despine(ax)

    def bars(ax, df, cat, ticklabels, xlabel):
        d = df.dropna(subset=[cat])
        cats = list(d[cat].cat.categories)
        w = 0.38
        ps = {}
        for k, (col, colour, lab) in enumerate([("ssm_ge_30", LBLUE, "SSM ≥ 30 kPa"),
                                                ("ssm_ge_40", ORANGE2, "SSM ≥ 40 kPa")]):
            ev, tot = [], []
            for i, c in enumerate(cats):
                g = d[d[cat] == c]
                e, n = int(g[col].sum()), len(g)
                ev.append(e)
                tot.append(n)
                lo, hi = wilson_ci(e, n)
                x = i + (k - 0.5) * w
                p = 100 * e / n
                ax.bar(x, p, width=w, color=colour, label=lab if i == 0 else None, zorder=2)
                ax.errorbar(x, p, yerr=[[p - 100 * lo], [100 * hi - p]], color=EBAR, lw=1.3,
                            capsize=3, capthick=1.3, zorder=3)
                ax.text(x, 100 * hi + 1.0, f"{e}/{n}", ha="center", va="bottom", fontsize=7.8)
            ps[col] = cochran_armitage(ev, tot)[1]
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(ticklabels, fontsize=9.2)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.set_ylim(0, 52)
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.set_ylabel("Patients above threshold (%)", fontsize=10.5)
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        _despine(ax)
        return ps

    def letter(ax, s, x=-0.25, y=1.06):
        ax.text(x, y, s, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(11.4, 8.9))
        pos = {"A": [0.065, 0.62, 0.235, 0.33], "B": [0.40, 0.62, 0.245, 0.33],
               "C": [0.745, 0.62, 0.235, 0.33], "D": [0.065, 0.09, 0.235, 0.355],
               "E": [0.585, 0.075, 0.405, 0.385]}

        # A - whole distribution by thiopurine exposure
        ax = fig.add_axes(pos["A"])
        u = pat.loc[pat["thiopurine"] == 0, "ssm"]
        e = pat.loc[pat["thiopurine"] == 1, "ssm"]
        strip(ax, [u, e], [BLUEPT, REDPT], [0.75, 0.75],
              [f"Not exposed\n(n = {len(u)})", f"Exposed\n(n = {len(e)})"])
        ax.set_xlabel("Thiopurine exposure", fontsize=10.5)
        ax.set_title(f"medians {u.median():.1f} vs {e.median():.1f} kPa; "
                     f"Mann–Whitney p = {stats.mannwhitneyu(u, e).pvalue:.2f}", fontsize=9)
        letter(ax, "A")

        # B - whole distribution by cumulative thiopurine dose
        ax = fig.add_axes(pos["B"])
        cat = "thio_cumulative_cat4"
        d = pat.dropna(subset=[cat])
        groups = [d.loc[d[cat] == c, "ssm"] for c in d[cat].cat.categories]
        kw = stats.kruskal(*groups).pvalue
        labs = ["None", "≤ 100 g", "101–300 g", "> 300 g"]
        strip(ax, groups, [BLUEPT, "#E8ACA8", "#D9887F", REDPT], [0.75, 0.75, 0.8, 0.85],
              [f"{lab}\n(n = {len(g)})" for lab, g in zip(labs, groups)])
        ax.set_xlabel("Cumulative thiopurine dose (AZA equivalent)", fontsize=10.5)
        ax.set_title(f"Kruskal–Wallis p = {kw:.2f} (whole distribution)", fontsize=9)
        letter(ax, "B")

        # C, D - proportions above 30 and 40 kPa
        ax = fig.add_axes(pos["C"])
        ps = bars(ax, pat, "thio_cumulative_cat4", labs, "Cumulative thiopurine dose")
        ax.set_title(f"Trend: ≥ 40 kPa p = {fmt_p(ps['ssm_ge_40'])}; "
                     f"≥ 30 kPa p = {fmt_p(ps['ssm_ge_30'])}", fontsize=9)
        letter(ax, "C")
        ax = fig.add_axes(pos["D"])
        ps = bars(ax, pat, "thio_months_cat4", ["None", "≤ 12", "13–48", "> 48"],
                  "Thiopurine duration (months)")
        ax.set_title(f"Trend: ≥ 40 kPa p = {fmt_p(ps['ssm_ge_40'])}; "
                     f"≥ 30 kPa p = {fmt_p(ps['ssm_ge_30'])}", fontsize=9)
        letter(ax, "D")

        # E - forest plot
        ax = fig.add_axes(pos["E"])
        rows = [
            ("Cumulative dose, per 100 g", pat, "thio_cumulative_g", 100, M1, "main"),
            ("Duration, per 12 months", pat, "thio_months", 12, M1, "main"),
            ("AZA daily dose, per 25 mg/day", pat, "aza_mg_day", 25, M1, "main"),
            ("Cumulative dose (users only)", thio_users, "thio_cumulative_g", 100, M1, "sens"),
            ("Cumulative dose + disease duration", pat, "thio_cumulative_g", 100, M2, "sens"),
            ("Cumulative dose + steroids + anti-TNF", pat, "thio_cumulative_g", 100, M3, "sens"),
            ("Anti-TNF duration (comparator)", pat, "antitnf_months", 12, M1, "comp"),
            ("Disease duration (comparator)", pat, "disease_duration_years", 1, M1, "comp"),
            ("Steroid courses (comparator)", pat, "steroid_courses", 1, M1, "comp"),
        ]
        n = len(rows)
        ys = np.arange(n)[::-1].astype(float)
        ax.axhspan(ys[5] - 0.5, ys[3] + 0.5, color="#F2F2F2", zorder=0)
        for y, (lab, df, col, unit, extra, kind) in zip(ys, rows):
            r = firth_or(df, col, unit, extra)
            colour = GREYC if kind == "comp" else ORANGE2
            ax.plot([r["low"], r["high"]], [y, y], color=colour, lw=2.2, zorder=3)
            for xx in (r["low"], r["high"]):
                ax.plot([xx, xx], [y - 0.09, y + 0.09], color=colour, lw=2.0, zorder=3)
            ax.plot(r["or"], y, "o", color=colour, ms=8, zorder=4)
            ax.text(1.0, y, f"{r['or']:.2f} ({r['low']:.2f}–{r['high']:.2f})",
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.6)
            if kind != "comp":
                q = _logit30(df, col, unit, extra)
                yy = y + 0.24
                ax.plot([q["low"], q["high"]], [yy, yy], color=LBLUE, lw=1.8, zorder=2)
                for xx in (q["low"], q["high"]):
                    ax.plot([xx, xx], [yy - 0.07, yy + 0.07], color=LBLUE, lw=1.6, zorder=2)
                ax.plot(q["or_"], yy, "s", color=LBLUE, ms=7, zorder=2)
        ax.axvline(1, color="black", lw=1.6, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 11.5)
        ax.set_xticks([0.5, 1, 2, 4])
        ax.set_xticklabels(["0.5", "1", "2", "4"], fontsize=11)
        ax.minorticks_off()
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
        ax.set_ylim(-1.75, n - 0.35)
        ax.text(1.0, n - 0.45, "OR (95% CI), SSM ≥ 40 kPa", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8.6, fontweight="bold")
        ax.set_xlabel("Adjusted odds ratio (log scale)\nFirth penalised logistic regression, "
                      "adjusted for age, sex and BMI", fontsize=9)
        _despine(ax)
        handles = [Line2D([], [], marker="o", color=ORANGE2, ls="", ms=9, label="SSM ≥ 40 kPa"),
                   Line2D([], [], marker="s", color=LBLUE, ls="", ms=9, label="SSM ≥ 30 kPa"),
                   Line2D([], [], marker="o", color=GREYC, ls="", ms=9,
                          label="Comparator (≥ 40 kPa)")]
        ax.legend(handles=handles, frameon=False, fontsize=8.2, loc="lower left",
                  bbox_to_anchor=(0.24, -0.012), ncol=2, handletextpad=0.25, columnspacing=1.0,
                  borderaxespad=0.2)
        letter(ax, "E", x=-0.53, y=1.03)
        _save(fig, "Figure2_thiopurine_exposure")


# ---------------------------------------------------------------------------
# Figure 3 - spleen versus liver stiffness and platelet counts
# ---------------------------------------------------------------------------
def figure3(pat, ref):
    rc = dict(BASE_RC, **{"axes.linewidth": 1.0})
    RED, BLUE3, GREY3, BROWN, SHADE = "#C0392B", "#4A7FC1", "#B8B8B8", "#8A4B08", "#FAEDDF"
    rng = np.random.default_rng(7)
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(11.6, 5.84))
        ax = fig.add_axes([0.057, 0.12, 0.457, 0.82])
        YMAX = 185
        ax.add_patch(plt.Rectangle((2, prep.SSM_MARKED), prep.LSM_CACLD - 2, YMAX - prep.SSM_MARKED,
                                   facecolor=SHADE, edgecolor="none", zorder=0))
        ax.scatter(ref["lsm"], ref["ssm"], s=42, color=GREY3, edgecolors="none", zorder=2,
                   label=f"Reference sample (n = {len(ref)})")
        un, ex = pat[pat["thiopurine"] == 0], pat[pat["thiopurine"] == 1]
        uk = pat[pat["thiopurine"].isna()]
        ax.scatter(un["lsm"], un["ssm"], s=42, color=BLUE3, edgecolors="none", zorder=3,
                   label=f"IBD, thiopurine-unexposed (n = {len(un)})")
        ax.scatter(ex["lsm"], ex["ssm"], s=42, color=RED, edgecolors="none", zorder=3,
                   label=f"IBD, thiopurine-exposed (n = {len(ex)})")
        ax.scatter(uk["lsm"], uk["ssm"], s=42, facecolors="white", edgecolors="#333333",
                   linewidths=1.1, zorder=3, label=f"IBD, exposure unknown (n = {len(uk)})")
        lp = pat[pat["thrombocytopenia"] == 1]
        ax.scatter(lp["lsm"], lp["ssm"], s=330, facecolors="none", edgecolors="black",
                   linewidths=1.9, zorder=4, label="Platelets < 150 ×10³/µL")
        ax.axhline(prep.SSM_MARKED, color="#555555", lw=1.4, ls=(0, (4, 2.5)), zorder=1)
        ax.axhline(prep.SSM_ELEVATED, color="#888888", lw=1.3, ls=(0, (1, 2)), zorder=1)
        ax.axvline(prep.LSM_CACLD, color="#555555", lw=1.4, ls=(0, (3, 2.5)), zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(2, 50)
        ax.set_ylim(4.4, YMAX)
        xt = [2, 3, 5, 7.2, 10, 20, 40]
        ax.set_xticks(xt)
        ax.set_xticklabels(["2", "3", "5", "7.2", "10", "20", "40"])
        yt = [5, 10, 20, 30, 40, 50, 75, 100]
        ax.set_yticks(yt)
        ax.set_yticklabels([str(t) for t in yt])
        ax.minorticks_off()
        ax.tick_params(labelsize=10.5)
        ax.set_xlabel("Liver stiffness (kPa, log scale)", fontsize=11)
        ax.set_ylabel("Spleen stiffness (kPa, log scale)", fontsize=11)
        _despine(ax)
        n_disc = int(((pat["ssm"] >= 40) & (pat["lsm"] < 10)).sum())
        n_both = int(((pat["ssm"] >= 40) & (pat["lsm"] >= 10)).sum())
        n_lsm = int(((pat["ssm"] < 40) & (pat["lsm"] >= 10)).sum())
        n_none = int(((pat["ssm"] < 40) & (pat["lsm"] < 10)).sum())
        ref_disc = int(((ref["ssm"] >= 40) & (ref["lsm"] < 10)).sum())
        ax.text(2.12, 178, f"Discordant pattern\nSSM ≥ 40 kPa with LSM < 10 kPa\nn = {n_disc} "
                f"({100 * n_disc / len(pat):.1f}% of patients;\n{ref_disc} of {len(ref)} "
                "reference participants)", fontsize=9.2, color=BROWN, fontweight="bold",
                va="top", ha="left", zorder=5)
        ax.text(10.6, 150, f"SSM ≥ 40 & LSM ≥ 10\nn = {n_both}", fontsize=9.5, color="#333333",
                va="center")
        ax.text(10.6, 5.0, f"LSM ≥ 10 kPa only\nn = {n_lsm}", fontsize=9.5, color="#333333",
                va="center")
        ax.text(2.12, 4.85, f"Neither: n = {n_none}", fontsize=9, color="#333333", va="center")
        rho, p = stats.spearmanr(pat["lsm"], pat["ssm"])
        ax.text(0.985, 0.352, f"IBD: Spearman ρ = {rho:.2f}, p = {p:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.6)
        idx = pat[(pat["ssm"] > 85) & (pat["ssm"] < 86)]
        if len(idx):
            ax.annotate("Index case: grade II oesophageal\nvarices at endoscopy",
                        xy=(idx["lsm"].iloc[0], idx["ssm"].iloc[0]), xytext=(11.5, 64),
                        fontsize=9.2, va="center",
                        arrowprops=dict(arrowstyle="->", lw=1.1, color="black", shrinkA=0,
                                        shrinkB=4),
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                  edgecolor="#888888", lw=0.9), zorder=6)
        ax.legend(loc="lower right", bbox_to_anchor=(1.075, 0.085), frameon=False, fontsize=9.2,
                  scatterpoints=1, handletextpad=0.4, borderaxespad=0.2)
        ax.text(-0.11, 1.0, "A", transform=ax.transAxes, fontsize=16, fontweight="bold",
                va="bottom")

        ax = fig.add_axes([0.623, 0.12, 0.367, 0.82])
        bins = [(-np.inf, 21, "< 21"), (21, 30, "21–29.9"), (30, 40, "30–39.9"),
                (40, 50, "40–49.9"), (50, np.inf, "≥ 50")]
        fills = ["#E2ECF6", "#E2ECF6", "#E2ECF6", "#F9DEB9", "#F6D1CD"]
        pts = ["#4A7FC1", "#4A7FC1", "#4A7FC1", "#E08A1E", "#C0392B"]
        labs = []
        for i, ((lo, hi, lab), fc, pc) in enumerate(zip(bins, fills, pts)):
            g = pat[(pat["ssm"] >= lo) & (pat["ssm"] < hi)]["plt"].dropna().values
            vp = ax.violinplot([g], positions=[i], widths=0.85, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(fc)
                b.set_alpha(1)
                b.set_edgecolor("none")
                b.set_zorder(1)
            x = i + rng.uniform(-0.16, 0.16, len(g))
            ax.scatter(x, g, s=16, color=pc, alpha=0.75, edgecolors=pc, linewidths=0.8, zorder=2)
            q1, med, q3 = np.percentile(g, [25, 50, 75])
            ax.plot([i - 0.28, i + 0.28], [med, med], color="black", lw=3, zorder=3,
                    solid_capstyle="butt")
            ax.plot([i, i], [q1, q3], color="black", lw=1.8, zorder=3)
            for q in (q1, q3):
                ax.plot([i - 0.12, i + 0.12], [q, q], color="black", lw=1.8, zorder=3)
            labs.append(f"{lab}\n(n = {len(g)})")
        ax.axhline(150, color="#666666", lw=1.3, ls=(0, (3, 2)), zorder=0)
        ax.text(4.45, 153, "150", color="#777777", fontsize=9, ha="right", va="bottom")
        ax.set_xticks(range(5))
        ax.set_xticklabels(labs, fontsize=10.5)
        ax.set_ylim(0, 800)
        ax.set_yticks(range(0, 801, 100))
        ax.tick_params(axis="y", labelsize=10.5)
        ax.set_xlim(-0.6, 4.6)
        ax.set_xlabel("Spleen stiffness category (kPa)", fontsize=11)
        ax.set_ylabel("Platelet count (×10³/µL)", fontsize=11)
        _despine(ax)
        d = pat.dropna(subset=["plt"])
        r2, p2 = stats.spearmanr(d["ssm"], d["plt"])
        tp = [100 * pat[(pat["ssm"] >= lo) & (pat["ssm"] < hi)]["thrombocytopenia"].mean()
              for lo, hi, _ in bins]
        ax.text(0.98, 0.975, f"Spearman ρ = {r2:.2f}, p = {p2:.3f}\nthrombocytopenia: "
                f"{tp[0]:.0f}% → {tp[-1]:.0f}% across categories", transform=ax.transAxes,
                ha="right", va="top", fontsize=9.6)
        ax.text(-0.17, 1.0, "B", transform=ax.transAxes, fontsize=16, fontweight="bold",
                va="bottom")
        _save(fig, "Figure3_discordance")


# ---------------------------------------------------------------------------
# Supplementary Figures S1 and S2 (simplified renderings)
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


def figure_s1(pat, ref):
    with plt.rc_context(dict(BASE_RC, **{"font.size": 8, "axes.linewidth": 0.8})):
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
        _save(fig, "SupplFigureS1_IBD_vs_reference", tight=True)


def figure_s2(pat):
    users = pat[pat["aza_user"] == 1]
    with plt.rc_context(dict(BASE_RC, **{"font.size": 8, "axes.linewidth": 0.8})):
        fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9))
        pairs = [("aza_mg_day", "ssm", "Azathioprine dose (mg/day)", "SSM (kPa)", "A"),
                 ("aza_mg_kg", "ssm", "Azathioprine dose (mg/kg/day)", "SSM (kPa)", "B"),
                 ("aza_mg_day", "weight", "Azathioprine dose (mg/day)", "Body weight (kg)", "C")]
        for ax, (x, y, xl, yl, panel) in zip(axes, pairs):
            d = users[[x, y]].dropna()
            ax.scatter(d[x], d[y], s=11, alpha=0.55, color=BLUE, edgecolors="none")
            rho, p = stats.spearmanr(d[x], d[y])
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{panel}  rho = {rho:+.2f} (p = {p:.3g})", fontsize=8.5, loc="left")
            _despine(ax)
        fig.tight_layout()
        _save(fig, "SupplFigureS2_dose_weight", tight=True)


# ---------------------------------------------------------------------------
# Supplementary Figures S3 and S4
# ---------------------------------------------------------------------------
def figure_s3(pat):
    users = pat[pat["aza_user"] == 1]
    panels = [(_quantreg(pat, "thiopurine"), "Thiopurine exposure vs none"),
              (_quantreg(users, "aza_mg_day", unit=25),
               "Azathioprine daily dose, per 25 mg (users)"),
              (_quantreg(pat, "thio_cumulative_g", unit=100),
               "Cumulative thiopurine dose, per 100 g (AZA equivalent)")]
    with plt.rc_context(BASE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
        for ax, (res, title), letter in zip(axes, panels, "ABC"):
            q = np.arange(len(res))
            est = np.array([r["ratio"] for r in res])
            lo = est - np.array([r["low"] for r in res])
            hi = np.array([r["high"] for r in res]) - est
            ax.errorbar(q, est, yerr=[lo, hi], fmt="o", color="#E08A1E", ms=7, lw=1.8, capsize=4,
                        capthick=1.6)
            ax.axhline(1, color="black", lw=1.1)
            ax.set_xticks(q)
            ax.set_xticklabels([f"{int(r['quantile'] * 100)}th" for r in res], fontsize=11)
            ax.tick_params(axis="y", labelsize=11)
            ax.set_xlabel("Percentile of the SSM distribution", fontsize=11)
            ax.set_ylabel("Ratio of quantile (95% CI)", fontsize=11)
            ax.set_title(title, fontsize=10.5)
            _despine(ax)
            ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=16, fontweight="bold",
                    va="bottom")
        fig.tight_layout(w_pad=3.0)
        _save(fig, "SupplFigureS3_quantile_regression")


def figure_s4(pat):
    from robustness import CUM, leave_one_out
    loo = leave_one_out(pat)
    full = firth_or(pat, CUM, 100, ())
    with plt.rc_context(BASE_RC):
        fig, ax = plt.subplots(figsize=(10.1, 6.2))
        ys = np.arange(len(loo))[::-1]
        for y, (est, lo, hi, p) in zip(ys, loo[["_or", "_low", "_high", "_p"]].to_numpy()):
            ax.plot([lo, hi], [y, y], color="#E08A1E", lw=1.4)
            for xx in (lo, hi):
                ax.plot([xx, xx], [y - 0.15, y + 0.15], color="#E08A1E", lw=1.2)
            ax.plot(est, y, "o", color="#E08A1E", ms=6.5)
            ax.text(1.0, y, f"{est:.2f} ({lo:.2f}–{hi:.2f}); {_pstr(p)}",
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.6)
        ax.axvline(1, color="black", lw=1.2)
        ax.axvline(full["or"], color="#888888", lw=1.2, ls=(0, (1.5, 2)))
        ax.set_xscale("log")
        ax.set_xlim(0.72, 2.9)
        ax.set_xticks([0.8, 1, 1.2, 1.5, 2])
        ax.set_xticklabels(["0.8", "1", "1.2", "1.5", "2"], fontsize=11)
        ax.minorticks_off()
        ax.set_yticks(ys)
        ax.set_yticklabels([f"omit patient with SSM {s:.1f} kPa (cumulative dose {int(c)} g)"
                            for s, c in zip(loo.iloc[:, 0], loo.iloc[:, 1])], fontsize=8.8)
        ax.set_ylim(-0.8, len(loo) - 0.3)
        ax.set_xlabel("Adjusted odds ratio for SSM ≥ 40 kPa per 100 g of cumulative thiopurine "
                      "dose (log scale)", fontsize=10.5)
        ax.set_title(f"Leave-one-out analysis over the {len(loo)} patients with SSM ≥ 40 kPa\n"
                     f"(dotted line: full-data estimate, OR {full['or']:.2f})", fontsize=10.5)
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

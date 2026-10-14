# ELASTIBD spleen stiffness — analysis code

Python code that reproduces every number, table and figure of

> Ağargün BF, Işık B, Şenkal İV, Rustamzada A, Nuriyev K, İstemihan Z, İmanov Z,
> Kızıltaş C, Çavuş B, Çifcibaşı Örmeci A, Demir K, Beşışık F, Kaymakoğlu S, Akyüz F.
> **Thiopurines and spleen stiffness in inflammatory bowel disease: no overall
> association, but markedly elevated values despite low liver stiffness with high
> cumulative exposure.** (submitted)

The study is a cross-sectional analysis of the prospective ELASTIBD cohort
(Istanbul University, Istanbul Faculty of Medicine): 271 adults with
inflammatory bowel disease examined with spleen-dedicated 100-Hz
vibration-controlled transient elastography, with 57 adults without
inflammatory bowel disease as a reference sample.

Release **v1.1.0** corresponds to version 8 of the manuscript. What changed
since v1.0.1 is listed in [CHANGELOG.md](CHANGELOG.md).

## Data availability

**This repository contains code only.** The individual participant data cannot
be shared publicly because of the privacy of the people who took part, in
keeping with the study's ethics approval (Istanbul Faculty of Medicine Clinical
Research Ethics Committee, no. 844800, 11 April 2022). De-identified data are
available on reasonable request to the corresponding author
(bfagargun@istanbul.edu.tr).

The scripts expect two SPSS files:

| file | content |
| --- | --- |
| `listev20-sadecehastalar.sav` | patients with IBD |
| `listev18_hastavekontrol.sav` | patients plus the reference sample |

`prep.py` drops the direct identifiers contained in those files (name,
national identity number, telephone number, date of birth, hospital file
number, record number, examination date, city) immediately after reading them; they are not
used in any analysis and never reach an output file. The variables the analysis
does use are listed in [data_dictionary.md](data_dictionary.md), and
the derived variables, thresholds and models in
[derived_variables.md](derived_variables.md).

## Usage

The published results were produced with Python 3.13 and the package versions
pinned in `requirements.txt`.

```bash
git clone https://github.com/bfagargun/ELASTIBD-spleen-stiffness.git
cd ELASTIBD-spleen-stiffness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir data                      # place the two .sav files here (not tracked)
export ELASTIBD_DATA=./data     # or set the path directly

python run_all.py               # tables, figures and the reproduction check
```

Everything is written to `outputs/` (comma-separated tables and, in
`outputs/figures/`, PNG at 300 dpi plus vector PDF). `outputs/` is not tracked
by git.

Individual steps can also be run on their own:

```bash
python analysis_tables.py         # Tables 1-4 (and Supplementary Table S14)
python analysis_supplementary.py  # Supplementary Tables S1-S11, S15 and S16
python robustness.py              # Supplementary Tables S12 and S13
python figures.py                 # Figures 1-3 and Supplementary Figures S1-S4
python verify.py                  # reproduction check only
```

| output | manuscript |
| --- | --- |
| `table1_characteristics.csv` | Table 1 and Supplementary Table S14 |
| `table2a_categorical_exposure.csv`, `table2b_continuous_exposure.csv` | Table 2 and the continuous exposure metrics in the Results |
| `table3a_thresholds_by_category.csv`, `table3b_firth_models.csv` | Table 3A and 3B (Table 3B also lists the Wald intervals for comparison) |
| `table4_marked_cases.csv` | Table 4 |
| `tableS1_...` to `tableS11_...`, `tableS15_...`, `tableS16_...` | Supplementary Tables S1–S11, S15 and S16 |
| `tableS12_leave_one_out.csv`, `tableS13_robustness.csv` | Supplementary Tables S12 and S13 |
| `figures/Figure1_flow_chart`, `Figure2_thiopurine_exposure`, `Figure3_discordance` | Figures 1–3 |
| `figures/SupplFigureS1_...` to `SupplFigureS4_...` | Supplementary Figures S1–S4 |

Figures 1–3 and Supplementary Figures S3–S4 are the published figures; with
the pinned package versions they are reproduced pixel for pixel, provided that
the Inter font is installed for Figure 1 (otherwise it falls back to DejaVu
Sans).
Supplementary Figures S1 and S2 are simplified renderings of the published
panels.

## Reproduction check

`verify.py` compares the results with the published manuscript (version 8) in
two ways. It first recomputes the numbers quoted in the text, the flow chart
and the tables one by one — the study population, the spleen-stiffness
distribution and reference limits, the factors associated with spleen
stiffness, the geometric mean ratios, the categories, trend tests and Firth
models of Table 3, the threshold scan, quantile regression and linearity
analyses, the joint dose–duration model, the robustness, leave-one-out and
specificity analyses, the discordance and haematological results and the
description of the patients with markedly elevated spleen stiffness — and
compares each, rounded as printed, with the published value. It then checks
that every published row of Tables 1–3 and Supplementary Tables S1–S16 is
reproduced exactly (published values in `published_tables.py`; Table 4 and the
omitted patients' values in Supplementary Table S12 describe individual
patients and are not stored). It exits with a non-zero status if any check
fails.

```
675 of 675 checks passed.
```

Without `repeated_examinations.csv` (see below) the two checks that need it are
reported as skipped (673 of 673 checks passed; 2 skipped).

## Files

| file | purpose |
| --- | --- |
| `prep.py` | reads the SPSS files, drops identifiers, derives the analysis variables, including thiopurine exposure in azathioprine equivalents |
| `firth.py` | Firth penalised logistic regression with profile penalised-likelihood inference, Cochran–Armitage trend test, Wilson intervals |
| `analysis_tables.py` | Tables 1, 2A, 2B, 3A, 3B and 4 |
| `analysis_supplementary.py` | Supplementary Tables S1–S11, S15 and S16 |
| `robustness.py` | robustness and leave-one-out analyses (Supplementary Tables S13 and S12) |
| `figures.py` | Figures 1–3 and Supplementary Figures S1–S4 |
| `verify.py` | reproduction check against the published numbers |
| `published_tables.py` | published values of Tables 1–3 and Supplementary Tables S1–S16, used by `verify.py` |
| `run_all.py` | runs all of the above |

## Statistical methods implemented here

* Spleen stiffness is right-skewed and is analysed on the natural-logarithmic
  scale; adjusted associations are reported as geometric mean ratios from
  linear regression with HC3 heteroskedasticity-robust standard errors.
* **Thiopurine exposure** is ever exposure to azathioprine or mercaptopurine.
  Cumulative dose is the maintenance daily dose × duration in months × 30 days,
  expressed in grams of azathioprine equivalent: mercaptopurine doses are
  multiplied by 2.08, the conventional factor that combines the molar mass
  ratio of the two drugs (277.3/152.2 = 1.82) with the approximately 88 %
  conversion of azathioprine to mercaptopurine. Treatment periods are summed in
  patients who received both drugs. Daily and weight-adjusted doses are
  analysed for azathioprine only.
* Markedly elevated spleen stiffness (≥ 40 kPa, 20 events) is modelled with
  Firth penalised logistic regression (Jeffreys-prior penalty; Firth,
  *Biometrika* 1993; Heinze & Schemper, *Stat Med* 2002). Confidence intervals
  are obtained from the profile penalised likelihood and p-values from
  penalised likelihood-ratio tests, with the penalty evaluated in the full
  model (as in the R package logistf); the Wald values are also returned.
  Conventional logistic regression is used for the ≥ 30 kPa threshold.
* Trends across ordered exposure categories use the Cochran–Armitage test with
  equally spaced scores and non-users as the lowest category.
* Quantile regression (25th, 50th, 75th and 90th percentiles; statsmodels
  `QuantReg`, t-based confidence intervals) is used for the distributional
  analyses.
* The shape of the association with cumulative dose is examined with a
  threshold scan from 21 to 50 kPa, a restricted cubic spline with knots at
  the 10th, 50th and 90th percentiles of exposed patients, a quadratic term,
  a > 300 g indicator and cumulative dose capped at 500 g.
* Categorical exposure levels (Table 2) are compared with their reference
  level in pairwise models fitted on the two levels being compared, so each
  contrast is unaffected by the remaining categories.
* Complete-case analysis throughout; the number of observations is reported for
  every model. Covariates with no variation inside a subset (for example sex in
  a sex-stratified analysis) are dropped automatically.

## Known limitation of this repository

One row of Supplementary Table S13 — "excluding examinations repeated within
the session (n = 6)" — cannot be derived from the two SPSS files, because those
files already contain the repeat measurements. `robustness.py` reproduces that
row only if an auxiliary file `repeated_examinations.csv` (a single column
`row` with the 0-based positions of the six patients among the 271 analysed
patients, in file order; held by the corresponding author) is placed in the
data directory; otherwise the row is reported as not available and every other
row is computed as published.

## Citing

See `CITATION.cff`. Please cite the manuscript as the primary reference; for
the code, cite this repository and the release used
(https://github.com/bfagargun/ELASTIBD-spleen-stiffness, release v1.1.0).

## Licence

MIT — see `LICENSE`.

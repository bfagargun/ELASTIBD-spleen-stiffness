# ELASTIBD spleen stiffness — analysis code

Python code that reproduces every number, table and figure of

> Ağargün BF, Işık B, Şenkal İV, Rustamzada A, Nuriyev K, İstemihan Z, İmanov Z,
> Kızıltaş C, Çavuş B, Çifcibaşı Örmeci A, Demir K, Beşışık F, Kaymakoğlu S, Akyüz F.
> **Cumulative azathioprine exposure is associated with markedly elevated spleen
> stiffness despite low liver stiffness in inflammatory bowel disease.** (submitted)

The study is a cross-sectional analysis of the prospective ELASTIBD cohort
(Istanbul University, Istanbul Faculty of Medicine): 271 adults with
inflammatory bowel disease examined with spleen-dedicated 100-Hz
vibration-controlled transient elastography, with 57 healthy adults as a
reference sample.

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
| `listev18_hastavekontrol.sav` | patients plus the healthy reference sample |

`prep.py` drops the direct identifiers contained in those files (name,
national identity number, telephone number, date of birth, hospital file
number, examination date, city) immediately after reading them; they are not
used in any analysis and never reach an output file. The variables the analysis
does use are listed in [data_dictionary.md](data_dictionary.md), and
the derived variables and thresholds in
[derived_variables.md](derived_variables.md).

## Usage

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
python analysis_tables.py         # Tables 1-4
python analysis_supplementary.py  # Supplementary Tables S1-S11
python robustness.py              # Table 3C and Supplementary Table S12
python figures.py                 # Figures 1-3 and S1-S4
python verify.py                  # reproduction check only
```

## Reproduction check

`verify.py` recomputes 96 numbers reported in the manuscript — the study
population, the spleen-stiffness distribution and reference limits, the
geometric mean ratios, the Cochran–Armitage trend tests, all Firth models of
Table 3B, and the discordance and haematological results — and compares each
with the published value. It exits with a non-zero status if any check fails.

```
96 of 96 checks passed.
```

## Files

| file | purpose |
| --- | --- |
| `prep.py` | reads the SPSS files, drops identifiers, derives the analysis variables |
| `firth.py` | Firth penalised logistic regression, Cochran–Armitage trend test, Wilson intervals |
| `analysis_tables.py` | Tables 1, 2A, 2B, 3A, 3B and 4 |
| `analysis_supplementary.py` | Supplementary Tables S1–S11 |
| `robustness.py` | Table 3C and the leave-one-out analysis (Supplementary Table S12) |
| `figures.py` | Figures 1–3 and Supplementary Figures S1–S4 |
| `verify.py` | reproduction check against the published numbers |
| `run_all.py` | runs all of the above |

## Statistical methods implemented here

* Spleen stiffness is right-skewed and is analysed on the natural-logarithmic
  scale; adjusted associations are reported as geometric mean ratios from
  linear regression with HC3 heteroskedasticity-robust standard errors.
* Markedly elevated spleen stiffness (≥ 40 kPa, 20 events) is modelled with
  Firth penalised logistic regression, implemented in `firth.py` as penalised
  score equations with the Jeffreys prior and Wald confidence intervals
  (Firth, *Biometrika* 1993; Heinze & Schemper, *Stat Med* 2002). The
  conventional maximum-likelihood fit is used for the ≥ 30 kPa threshold.
* Trends across ordered exposure categories use the Cochran–Armitage test with
  equally spaced scores and non-users as the lowest category.
* Quantile regression (25th, 50th, 75th and 90th percentiles) is used for the
  distributional analyses.
* Categorical exposure levels (Table 2A) are compared with their reference
  level in pairwise models fitted on the two levels being compared, so each
  contrast is unaffected by the remaining categories.
* Complete-case analysis throughout; the number of observations is reported for
  every model. Covariates with no variation inside a subset (for example sex in
  a sex-stratified analysis) are dropped automatically.

## Known limitation of this repository

One row of Table 3C — "excluding examinations repeated within the session
because the first acquisition had an absolute interquartile range above 25 kPa
(n = 6)" — cannot be derived from the two SPSS files, because those files
already contain the repeat measurements. `robustness.py` reproduces that row
only if an auxiliary file `repeated_examinations.csv` (a single column `row`
with 0-based row positions in the patient file, held by the corresponding
author) is placed in the data directory; otherwise the row is reported as not
available and every other row is computed as published.

## Citing

See `CITATION.cff`. Please cite the manuscript as the primary reference; for
the code, cite this repository and the release used
(https://github.com/bfagargun/ELASTIBD-spleen-stiffness, release v1.0.0).

## Licence

MIT — see `LICENSE`.

# Changelog

## v1.1.0 — 2026-10-14

Corresponds to version 8 of the manuscript ("Thiopurines and spleen stiffness
in inflammatory bowel disease: no overall association, but markedly elevated
values despite low liver stiffness with high cumulative exposure").

### Exposure definition
* Cumulative dose and duration now cover all thiopurines: mercaptopurine doses
  are converted to azathioprine equivalents (× 2.08) and added to the
  azathioprine dose, and mercaptopurine months are added to the duration
  (`thio_cumulative_g`, `thio_months` and their categories in `prep.py`). The
  "users only" analyses of cumulative dose and duration are restricted to
  thiopurine users. Daily and weight-adjusted doses remain
  azathioprine-specific.
* The cumulative-dose categories change from 72 / 73 / 76 / 42 to
  71 / 73 / 75 / 44 patients (events unchanged, 4 / 3 / 5 / 8); the trend p for
  SSM ≥ 40 kPa changes from 0.022 to 0.027 (all patients) and from 0.009 to
  0.011 (users), and the Model 1 odds ratio per 100 g from 1.41 to 1.39. The
  azathioprine-only results are kept as a sensitivity analysis.
* New variables `mp_mg_day` and `mp_months`; new categorical variable
  `aza_mg_kg_cat` (weight-adjusted dose).

### Statistical methods
* Firth models now report profile penalised-likelihood 95 % confidence
  intervals and penalised likelihood-ratio p-values (`firth.firth_profile`);
  the Wald values are still returned and are listed in Table 3B for
  comparison. A failed line search, a constrained fit that does not converge
  or a confidence limit that cannot be bracketed now raise an error instead of
  passing silently.
* Quantile regression is fitted with up to 5,000 iterations; its confidence
  intervals are t-based (statsmodels default), as in the published tables.

### Tables
* Table 2 lists thiopurine and azathioprine exposure, treatment regimen,
  azathioprine daily and weight-adjusted dose, thiopurine duration and
  cumulative thiopurine dose, in the order of the manuscript.
* Table 3A covers cumulative thiopurine dose, thiopurine duration and
  azathioprine daily dose; Table 3B covers all patients and thiopurine users.
* The robustness table is now Supplementary Table S13
  (`tableS13_robustness.csv`, previously `table3c_robustness.csv`), with five
  new rows: additional adjustment for LSM and for alcohol use, azathioprine dose
  alone, exclusion of the three mercaptopurine-exposed patients, and conversion
  with the molar mass ratio alone (× 1.82).
* Supplementary Table S2 adds the Crohn's disease phenotype (Montreal B);
  Supplementary Table S5 adds the primary model without alcohol use and uses
  thiopurine users for cumulative dose; Supplementary Table S8 shows
  thiopurine duration; Supplementary Tables S4, S9 and S10 use profile
  penalised-likelihood inference; Supplementary Table S11 states the pairwise
  sample sizes and adds the all-patient correlation of cumulative dose with
  disease duration.
* New Supplementary Table S15 (azathioprine daily dose and duration in the same
  model) and Supplementary Table S16 (threshold scan, quantile regression of
  cumulative dose and linearity analyses).
* p-values in all output tables are printed in the journal format (a value
  that rounds to 0.100 is printed as 0.10).

### Figures
* `figures.py` now produces the published Figures 1–3 and Supplementary
  Figures S3–S4 (pixel-identical with the pinned package versions and, for
  Figure 1, the Inter font). Figure 1 takes its counts from the data, including
  the 358 patients of the cohort file (`prep.n_cohort`). PDF files carry no
  creation date, so repeated runs give identical files.

### Reproduction check and documentation
* `verify.py` checks the published numbers of manuscript version 8 (previously
  version 7), compared as printed, and every published row of Tables 1–3 and
  Supplementary Tables S1–S16 (new file `published_tables.py`): 675 checks.
  The two checks that need `repeated_examinations.csv` are reported as skipped
  when the file is not available.
* The reference sample is described as adults without IBD throughout.
* Package versions are pinned for Python 3.13, including formulaic, patsy and
  pillow.

## v1.0.1 — 2026-09-14

* Table 2A compares each exposure level with its reference level in a pairwise
  model; the reproduction check was extended accordingly.

## v1.0.0

* First release, accompanying version 7 of the manuscript.

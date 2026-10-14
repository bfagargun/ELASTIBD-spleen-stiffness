# Derived variables, thresholds and analysis sets

All derivations are in `prep.py`.

## Analysis sets

| set | definition | n |
| --- | --- | --- |
| cohort | all patients in the ELASTIBD baseline file | 358 |
| attempted | a spleen stiffness value was recorded **or** the examination was coded as unsuccessful | 310 patients, 64 reference participants |
| analysis set | a valid spleen stiffness value was recorded | 271 patients |
| reference sample | adults without IBD or known liver disease with a valid value | 57 |
| reliability subset | IQR/median ≤ 30 % (sensitivity analyses only, never an exclusion) | 186 |
| strict reliability subset | IQR/median < 25 % | 146 |

No participant is excluded on the basis of the interquartile range: no
reliability criterion has been validated for spleen stiffness, so IQR/median is
used only to define subsets for sensitivity analyses. Six analysed patients
were re-examined within the same session, and the repeat measurement is the one
held in the data. Of the seven reference participants without a value, six had
no valid measurement and in one the value was not recorded.

## Exposure variables

| variable | definition |
| --- | --- |
| `thiopurine` | ever exposure to azathioprine **or** mercaptopurine; missing only when neither drug has a recorded history |
| `aza_mg_day` | maintenance daily dose of azathioprine (mg/day); set to 0 in non-users |
| `aza_months` | duration of azathioprine exposure (months); set to 0 in non-users |
| `aza_mg_kg` | `aza_mg_day / weight` (mg/kg/day) |
| `aza_cumulative_g` | **`aza_mg_day × aza_months × 30 / 1000`** (grams); sensitivity analysis only |
| `mp_cumulative_g` | `mp_mg_day × mp_months × 30 / 1000` (grams of mercaptopurine); 0 in non-users |
| `thio_cumulative_g` | **`aza_cumulative_g + 2.08 × mp_cumulative_g`**, the cumulative thiopurine dose in grams of azathioprine equivalent |
| `thio_months` | `aza_months + mp_months`, the duration of thiopurine exposure |
| `thio_user` | 1 = thiopurine-exposed, 0 = unexposed (missing without a medication history) |
| `aza_combo` | Neither / Anti-TNF without AZA / AZA monotherapy / AZA + anti-TNF |

The conversion factor 2.08 is the conventional factor that combines the molar
mass ratio of azathioprine to mercaptopurine (277.3/152.2 = 1.82) with the
approximately 88 % conversion of azathioprine to mercaptopurine. Treatment
periods are summed in the two patients who received both drugs; the conversion
changes the exposure values of the three mercaptopurine-exposed patients only.
Supplementary Table S13 repeats the analysis with azathioprine alone, without
the three mercaptopurine-exposed patients and with the molar mass ratio alone
(× 1.82). Missing information propagates: when a user's dose or duration is
missing, the cumulative dose is missing too.

In the all-patient models, non-users contribute zero exposure. The "users only"
analyses of cumulative dose and duration are restricted to thiopurine users
(`thio_user == 1`); those of daily and weight-adjusted dose to azathioprine
users (`aza == 1`).

### Exposure categories

| metric | categories |
| --- | --- |
| cumulative thiopurine dose | none · ≤ 100 g · 101–300 g · > 300 g (`thio_cumulative_cat4`) |
| thiopurine duration | none · ≤ 12 months · 13–48 months · > 48 months (`thio_months_cat4`) |
| azathioprine daily dose | none · ≤ 75 mg · 100 mg · > 100 mg (`aza_dose_cat4`) |
| azathioprine weight-adjusted dose | < 1.5 · 1.5–1.99 · ≥ 2.0 mg/kg/day (users only, `aza_mg_kg_cat`) |

The duration categories reflect the median latency of nodular regenerative
hyperplasia reported after azathioprine. Trend tests use equally spaced scores
over the four ordered categories with non-users as the lowest category, and are
repeated among users only. The azathioprine-only categories
(`aza_cumulative_cat4`, `aza_duration_cat4`) are kept for the sensitivity
analysis.

## Outcome thresholds

| threshold | value | rationale |
| --- | --- | --- |
| elevated SSM | ≥ 30 kPa | 95th percentile of the reference sample (29.6 kPa) |
| markedly elevated SSM | ≥ 40 kPa | above the reference maximum (35.5 kPa); Baveno VII threshold below which high-risk varices can be excluded |
| Baveno VII rule-out | ≥ 21 kPa | tabulated for comparison |
| Baveno VII rule-in | ≥ 50 kPa | tabulated for comparison |
| discordant pattern | SSM ≥ 40 kPa with LSM < 10 kPa | elastographic pattern of non-cirrhotic portal hypertension |
| significant fibrosis | LSM ≥ 7.2 kPa | |
| compensated advanced chronic liver disease | LSM ≥ 10 kPa | |
| steatosis | CAP ≥ 248 dB/m | |
| device ceiling | 100 kPa | upper limit of the spleen module; one measurement reached it |

## Models

* **Distribution (geometric mean ratios).** Linear regression of `log(ssm)`
  adjusted for age, sex, BMI, any alcohol use and LSM, with HC3 robust standard
  errors. This model and the primary hypothesis were specified before the data
  were analysed. Because alcohol use is unrecorded in 42 of the 266 patients
  with a medication history, the model is also refitted without it
  (Supplementary Table S5).
* **Upper tail (odds ratios).** Firth penalised logistic regression for
  SSM ≥ 40 kPa with profile penalised-likelihood 95 % confidence intervals and
  penalised likelihood-ratio p-values. Model 1: age, sex, BMI. Model 2:
  additionally disease duration. Model 3: additionally number of corticosteroid
  courses and anti-TNF exposure. Conventional logistic regression is used for
  SSM ≥ 30 kPa.
* **Shape of the association (Supplementary Table S16).** Model 1 refitted at
  thresholds from 21 to 50 kPa; quantile regression of `log(ssm)` on
  cumulative dose; a restricted cubic spline with knots at the 10th, 50th and
  90th percentiles of cumulative dose among exposed patients (9, 129 and
  420 g), a quadratic term, a > 300 g indicator and cumulative dose capped at
  500 g.
* **Dose and duration together (Supplementary Table S15).** Azathioprine daily
  dose (per 25 mg/day) and duration (per 12 months) entered in the same Firth
  model.
* **Multivariable models (Supplementary Table S4).** Model A is clinical
  (age, sex, BMI, alcohol, disease type, disease duration, thiopurine and
  anti-TNF exposure, corticosteroid courses); Model B adds LSM, platelet count
  and steatosis. Platelet count is kept in the second model because
  thrombocytopenia may lie downstream of portal hypertension rather than being
  a cause of spleen stiffness.

The threshold-based tail analyses and the robustness analyses were developed
after the primary analysis, are reported as exploratory, and were not corrected
for multiplicity.

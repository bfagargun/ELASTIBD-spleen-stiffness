# Derived variables, thresholds and analysis sets

All derivations are in `code/prep.py`.

## Analysis sets

| set | definition | n |
| --- | --- | --- |
| attempted | a spleen stiffness value was recorded **or** the examination was coded as unsuccessful | 310 patients, 64 reference participants |
| analysis set | a valid spleen stiffness value was recorded | 271 patients |
| reference sample | healthy participants with a valid value | 57 |
| reliability subset | IQR/median ≤ 30 % (sensitivity analyses only, never an exclusion) | 186 |
| strict reliability subset | IQR/median < 25 % | 143 |

No participant is excluded on the basis of the interquartile range: no
reliability criterion has been validated for spleen stiffness, so IQR/median is
used only to define subsets for sensitivity analyses. Examinations whose first
acquisition had an absolute IQR above 25 kPa were repeated within the same
session and the repeat measurement is the one held in the data.

## Exposure variables

| variable | definition |
| --- | --- |
| `thiopurine` | ever exposure to azathioprine **or** mercaptopurine; missing only when neither drug has a recorded history |
| `aza_mg_day` | maintenance daily dose of azathioprine (mg/day); set to 0 in non-users |
| `aza_months` | duration of azathioprine exposure (months); set to 0 in non-users |
| `aza_mg_kg` | `aza_mg_day / weight` (mg/kg/day) |
| `aza_cumulative_g` | **`aza_mg_day × aza_months × 30 / 1000`** (grams) |
| `aza_combo` | Neither / Anti-TNF without AZA / AZA monotherapy / AZA + anti-TNF |

In the all-patient models, non-users contribute zero exposure; the
"users only" models are restricted to `aza == 1`.

### Exposure categories

| metric | categories |
| --- | --- |
| cumulative dose | none · ≤ 100 g · 101–300 g · > 300 g |
| duration | none · ≤ 12 months · 13–48 months · > 48 months |
| daily dose | none · ≤ 75 mg · 100 mg · > 100 mg |

The duration categories reflect the median latency of nodular regenerative
hyperplasia reported after azathioprine. Trend tests use equally spaced scores
over the four ordered categories with non-users as the lowest category, and are
repeated among azathioprine users only.

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
  were analysed.
* **Upper tail (odds ratios).** Firth penalised logistic regression for
  SSM ≥ 40 kPa. Model 1: age, sex, BMI. Model 2: additionally disease duration.
  Model 3: additionally number of corticosteroid courses and anti-TNF exposure.
  Conventional logistic regression is used for SSM ≥ 30 kPa.
* **Multivariable models (Supplementary Table S4).** Model A is clinical
  (age, sex, BMI, alcohol, disease type, disease duration, thiopurine and
  anti-TNF exposure, corticosteroid courses); Model B adds LSM, platelet count
  and steatosis. Platelet count is kept in the second model because
  thrombocytopenia may lie downstream of portal hypertension rather than being
  a cause of spleen stiffness.

The threshold-based tail analyses and the robustness analyses were developed
after the primary analysis, are reported as exploratory, and were not corrected
for multiplicity.

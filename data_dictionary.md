# Data dictionary

The analysis reads two SPSS files that are not distributed with this repository
(see the README). The tables below list every source variable the code uses,
the analysis name it is renamed to in `code/prep.py`, and its coding. Variables
present in the source files but not listed here are not used.

Both files carry the same variable names; the combined file additionally
contains `hastasağlıklı` (1 = patient with IBD, 2 = healthy participant), which
is used to select the reference sample.

## Dropped on load

`adsoyad`, `tcno`, `telefonno`, `doğumtarihi`, `poldosyası`, `sırano`,
`fs_tarihi`, `yasadıgısehir` — direct identifiers. They are removed
immediately after the file is read and are used in no analysis and in no
output.

## Elastography

| source variable | analysis name | coding / unit |
| --- | --- | --- |
| `DalakmedkPa` | `ssm` | spleen stiffness, median of the valid acquisitions, kPa (100-Hz spleen module) |
| `DalakIQR` | `ssm_iqr` | absolute interquartile range of the spleen acquisitions, kPa |
| `DalakIQRmed` | `ssm_iqr_median` | IQR/median of the spleen examination, per cent |
| `Dalakolcumbasarisi` | `ssm_success` | 1 = examination successful, 0 = unsuccessful |
| `KCmedkPa` | `lsm` | liver stiffness, kPa |
| `KCIQRmed` | `lsm_iqr_median` | IQR/median of the liver examination, per cent |
| `CAP` | `cap` | controlled attenuation parameter, dB/m |
| `Prob` | `probe` | 1 = M probe, 2 = XL probe (liver examination) |

## Demographics and lifestyle

| source variable | analysis name | coding / unit |
| --- | --- | --- |
| `yaş` | `age` | years |
| `cinsiyet` | `male` | 1 = male, 0 = female |
| `BMI` | `bmi` | kg/m² |
| `kilo` | `weight` | kg |
| `Belçevresi` | `waist` | cm |
| `Alkol0yok1var` | `alcohol_any` | 1 = any alcohol use, 0 = none |
| `Alkolgrhafta` | `alcohol_g_week` | g/week |
| `Sigaranominal` | `smoking` | 0 = never, 1 = current, 2 = former |
| `Sigarapaketyıl` | `pack_years` | pack-years |
| `HT` / `DM` | `hypertension` / `diabetes` | 1 = yes, 0 = no |

## Disease characteristics

| source variable | analysis name | coding / unit |
| --- | --- | --- |
| `Hastalık0Crohn1ÜK` | `uc` | 1 = ulcerative colitis, 0 = Crohn's disease |
| `hastalikyasiay` | `disease_duration_months` | months since diagnosis |
| `taniyasi` | `age_at_diagnosis` | years |
| `remisyon` | `remission` | 1 = clinical remission (CDAI < 150, SCCAI ≤ 2 or Mayo ≤ 2) |
| `CDAI`, `ÜKSCCAI`, `ÜKMayoskor` | `cdai`, `sccai`, `mayo` | activity indices |
| `Chrohnisefenotip...` | `cd_phenotype` | 0 = inflammatory, 1 = penetrating, 2 = stricturing |
| `Barsakrezeksiyonu` | `resection` | 1 = prior bowel resection |
| `Esktraintestinaltutulum0yok1var` | `eim` | 1 = extraintestinal manifestations |
| `Hastsalıkaktivasyonuvaryok` | `flare_any` | 1 = documented flare in the past 12 months |
| `Hastalıkaktivasyonuson1yıldakaçdefa` | `flare_n` | number of flares in the past 12 months |
| `Son1yılCRPortalaması` | `crp_mean` | mean C-reactive protein over 12 months, mg/L |
| `Splenomegali`, `Hepatomegali` | `splenomegaly`, `hepatomegaly` | 1 = present on abdominal imaging (available in a subset) |

## Medication

| source variable | analysis name | coding / unit |
| --- | --- | --- |
| `AZA` | `aza` | 1 = ever exposed to azathioprine |
| `AZAdozu` | `aza_mg_day` | maintenance daily dose, mg/day (0 in non-users) |
| `AZAsüresiay` | `aza_months` | duration of exposure, months (0 in non-users) |
| `Merkaptopürin` | `mercaptopurine` | 1 = ever exposed |
| `AntiTNF` | `antitnf` | 1 = ever exposed |
| `AntiTNFsüresiay` | `antitnf_months` | months |
| `Methotrexate` | `methotrexate` | 1 = ever exposed |
| `vedo_uste_yes_no` | `vedo_uste` | 1 = vedolizumab or ustekinumab |
| `Steroidkürsayısı` | `steroid_courses` | number of systemic corticosteroid courses |
| `Budesonid` | `budesonide` | 1 = ever exposed |
| `OralMesalazindozu` | `mesalazine_dose` | mg/day; exposure is defined as a dose above zero |
| `INH` | `isoniazid` | 1 = isoniazid prophylaxis |

## Laboratory

| source variable | analysis name | unit |
| --- | --- | --- |
| `AST`, `ALT`, `ALP`, `GGT` | `ast`, `alt`, `alp`, `ggt` | U/L |
| `TotalBil` | `bilirubin` | mg/dL |
| `Albumin` | `albumin` | g/dL |
| `HGB` | `hgb` | g/dL |
| `WBC` | `wbc` | ×10³/µL |
| `PLT` | `plt` | ×10³/µL |
| `FIB4skor`, `APRIskor` | `fib4`, `apri` | index |
| `steatoz_248` | `steatosis` | 1 = CAP ≥ 248 dB/m |

## Missing data

Complete-case analysis is used throughout and the number of observations is
reported for every model. Medication history is unavailable for 5 of the 271
analysed patients; the IQR/median of the spleen examination is unrecorded for
15; alcohol use, waist circumference, imaging findings and flare data are
available in subsets, which is why model denominators differ between analyses.

"""Data preparation for the ELASTIBD spleen-stiffness analysis.

The analysis reads two SPSS files that are *not* part of this repository
(see README and docs/data_dictionary.md):

* ``listev20-sadecehastalar.sav``  - patients with inflammatory bowel disease
* ``listev18_hastavekontrol.sav``  - patients plus the healthy reference sample

Set the directory that holds them with the environment variable
``ELASTIBD_DATA`` (default: ``./data``).

Direct identifiers present in the source files (name, national identity number,
telephone number, date of birth, hospital file number) are dropped immediately
after reading and never enter any derived object or output file.

Analysis sets
-------------
attempted  : spleen examination attempted (a value was recorded or the
             examination was coded as unsuccessful)
analysis   : valid spleen stiffness measurement (the primary analysis set)
reference  : healthy participants with a valid measurement
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("ELASTIBD_DATA", "data"))
PATIENT_FILE = "listev20-sadecehastalar.sav"
REFERENCE_FILE = "listev18_hastavekontrol.sav"

#: Direct identifiers that are dropped on load and never used.
IDENTIFIERS = [
    "adsoyad",       # name
    "tcno",          # national identity number
    "telefonno",     # telephone number
    "doğumtarihi",   # date of birth
    "poldosyası",    # hospital file number
    "sırano",        # record number
    "fs_tarihi",     # examination date
    "yasadıgısehir",  # city of residence
]

#: Source variable -> analysis variable. Documented in docs/data_dictionary.md.
RENAME = {
    # elastography
    "DalakmedkPa": "ssm",
    "DalakIQR": "ssm_iqr",
    "DalakIQRmed": "ssm_iqr_median",
    "Dalakolcumbasarisi": "ssm_success",
    "KCmedkPa": "lsm",
    "KCIQRmed": "lsm_iqr_median",
    "CAP": "cap",
    "Prob": "probe",
    # demographics and lifestyle
    "yaş": "age",
    "cinsiyet": "male",
    "BMI": "bmi",
    "kilo": "weight",
    "Belçevresi": "waist",
    "Alkol0yok1var": "alcohol_any",
    "Alkolgrhafta": "alcohol_g_week",
    "Sigaranominal": "smoking",
    "Sigarapaketyıl": "pack_years",
    "HT": "hypertension",
    "DM": "diabetes",
    # disease characteristics
    "Hastalık0Crohn1ÜK": "uc",
    "hastalikyasiay": "disease_duration_months",
    "taniyasi": "age_at_diagnosis",
    "remisyon": "remission",
    "CDAI": "cdai",
    "ÜKSCCAI": "sccai",
    "ÜKMayoskor": "mayo",
    "Chrohnisefenotip0inflamatuvar1fistülizan2stenozan": "cd_phenotype",
    "Barsakrezeksiyonu": "resection",
    "Esktraintestinaltutulum0yok1var": "eim",
    "Hastsalıkaktivasyonuvaryok": "flare_any",
    "Hastalıkaktivasyonuson1yıldakaçdefa": "flare_n",
    "Son1yılCRPortalaması": "crp_mean",
    "Splenomegali": "splenomegaly",
    "Hepatomegali": "hepatomegaly",
    # medication
    "AZA": "aza",
    "AZAdozu": "aza_mg_day",
    "AZAsüresiay": "aza_months",
    "Merkaptopürin": "mercaptopurine",
    "AntiTNF": "antitnf",
    "AntiTNFsüresiay": "antitnf_months",
    "Methotrexate": "methotrexate",
    "vedo_uste_yes_no": "vedo_uste",
    "Steroidkürsayısı": "steroid_courses",
    "Budesonid": "budesonide",
    "OralMesalazindozu": "mesalazine_dose",
    "INH": "isoniazid",
    # laboratory
    "AST": "ast",
    "ALT": "alt",
    "ALP": "alp",
    "GGT": "ggt",
    "TotalBil": "bilirubin",
    "Albumin": "albumin",
    "HGB": "hgb",
    "WBC": "wbc",
    "PLT": "plt",
    "FIB4skor": "fib4",
    "APRIskor": "apri",
    "steatoz_248": "steatosis",
}

# ---------------------------------------------------------------------------
# thresholds and cut-offs used throughout the manuscript
# ---------------------------------------------------------------------------
SSM_ELEVATED = 30.0        # 95th percentile of the reference sample (29.6 kPa)
SSM_MARKED = 40.0          # above the reference maximum (35.5 kPa); Baveno VII
SSM_BAVENO_RULE_OUT = 21.0
SSM_BAVENO_RULE_IN = 50.0
LSM_CACLD = 10.0           # compensated advanced chronic liver disease
LSM_SIGNIFICANT_FIBROSIS = 7.2
CAP_STEATOSIS = 248.0
RELIABILITY_IQR_MEDIAN = 30.0   # sensitivity subset only, never an exclusion
DEVICE_CEILING = 100.0


def _read(path: Path) -> pd.DataFrame:
    import pyreadstat

    df, _meta = pyreadstat.read_sav(str(path))
    return df.drop(columns=[c for c in IDENTIFIERS if c in df.columns])


def _common_derivations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns}).copy()
    df["log_ssm"] = np.log(df["ssm"])
    df["xl_probe"] = (df["probe"] == 2).astype(float).where(df["probe"].notna())
    df["ssm_ge_21"] = (df["ssm"] >= SSM_BAVENO_RULE_OUT).astype(float)
    df["ssm_ge_30"] = (df["ssm"] >= SSM_ELEVATED).astype(float)
    df["ssm_ge_40"] = (df["ssm"] >= SSM_MARKED).astype(float)
    df["ssm_ge_50"] = (df["ssm"] >= SSM_BAVENO_RULE_IN).astype(float)
    df["ssm_lsm_ratio"] = df["ssm"] / df["lsm"]
    df["reliable"] = (df["ssm_iqr_median"] <= RELIABILITY_IQR_MEDIAN).astype(float).where(
        df["ssm_iqr_median"].notna()
    )
    df["reliable_strict"] = (df["ssm_iqr_median"] < 25).astype(float).where(
        df["ssm_iqr_median"].notna()
    )
    df["lsm_ge_10"] = (df["lsm"] >= LSM_CACLD).astype(float)
    df["lsm_ge_72"] = (df["lsm"] >= LSM_SIGNIFICANT_FIBROSIS).astype(float)
    df["discordant"] = ((df["ssm"] >= SSM_MARKED) & (df["lsm"] < LSM_CACLD)).astype(float)
    df["thrombocytopenia"] = (df["plt"] < 150).astype(float).where(df["plt"].notna())
    return df


def _patient_derivations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # thiopurine exposure: ever azathioprine or mercaptopurine; missing only when
    # no medication history at all is available for either drug
    thio = ((df["aza"] == 1) | (df["mercaptopurine"] == 1)).astype(float)
    no_history = df["aza"].isna() & df["mercaptopurine"].isna()
    df["thiopurine"] = thio.where(~no_history)

    # azathioprine exposure metrics
    df["aza_mg_kg"] = df["aza_mg_day"] / df["weight"]
    # cumulative dose in grams: maintenance daily dose (mg) x months x 30 days / 1000
    df["aza_cumulative_g"] = df["aza_mg_day"] * df["aza_months"] * 30.0 / 1000.0
    df["aza_years"] = df["aza_months"] / 12.0
    df["disease_duration_years"] = df["disease_duration_months"] / 12.0
    df["antitnf_years"] = df["antitnf_months"] / 12.0

    users = df["aza"] == 1
    df["aza_user"] = users.astype(float).where(df["aza"].notna())
    # non-users contribute zero exposure in the all-patient analyses
    for col in ["aza_mg_day", "aza_months", "aza_cumulative_g", "aza_mg_kg", "aza_years"]:
        df.loc[df["aza"] == 0, col] = 0.0

    df["aza_dose_cat"] = pd.cut(
        df["aza_mg_day"].where(users), bins=[-0.1, 75, 100, 1e6],
        labels=["<= 75 mg", "100 mg", "> 100 mg"],
    )
    df["aza_duration_cat"] = pd.cut(
        df["aza_months"].where(users), bins=[-0.1, 12, 48, 1e6],
        labels=["<= 12 months", "13-48 months", "> 48 months"],
    )
    df["aza_cumulative_cat"] = pd.cut(
        df["aza_cumulative_g"].where(users), bins=[-0.1, 100, 300, 1e6],
        labels=["<= 100 g", "101-300 g", "> 300 g"],
    )
    # four ordered categories with non-users as the lowest level (trend tests)
    def _with_nonusers(cat_col, labels):
        out = pd.Series(pd.NA, index=df.index, dtype="object")
        out[df["aza"] == 0] = "none"
        m = df[cat_col].notna()
        out[m] = df.loc[m, cat_col].astype(str)
        return pd.Categorical(out, categories=["none"] + labels, ordered=True)

    df["aza_cumulative_cat4"] = _with_nonusers(
        "aza_cumulative_cat", ["<= 100 g", "101-300 g", "> 300 g"])
    df["aza_duration_cat4"] = _with_nonusers(
        "aza_duration_cat", ["<= 12 months", "13-48 months", "> 48 months"])
    df["aza_dose_cat4"] = _with_nonusers(
        "aza_dose_cat", ["<= 75 mg", "100 mg", "> 100 mg"])

    df["aza_combo"] = np.select(
        [
            (df["aza"] == 1) & (df["antitnf"] == 1),
            (df["aza"] == 1) & (df["antitnf"] == 0),
            (df["aza"] == 0) & (df["antitnf"] == 1),
            (df["aza"] == 0) & (df["antitnf"] == 0),
        ],
        ["AZA + anti-TNF", "AZA monotherapy", "Anti-TNF without AZA", "Neither"],
        default=None,
    )
    df["mesalazine"] = (df["mesalazine_dose"] > 0).astype(float).where(
        df["mesalazine_dose"].notna())
    return df


def load_patients(data_dir: Path | str | None = None) -> pd.DataFrame:
    """All patients in whom a spleen examination was attempted."""
    d = Path(data_dir) if data_dir is not None else DATA_DIR
    df = _read(d / PATIENT_FILE)
    attempted = df["ssm_notna"] if "ssm_notna" in df else (
        df["DalakmedkPa"].notna() | df["Dalakolcumbasarisi"].notna()
    )
    df = df.loc[attempted].copy()
    return _patient_derivations(_common_derivations(df))


def load_reference(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Healthy reference participants in whom an examination was attempted."""
    d = Path(data_dir) if data_dir is not None else DATA_DIR
    df = _read(d / REFERENCE_FILE)
    df = df.loc[df["hastasağlıklı"] == 2].copy()
    attempted = df["DalakmedkPa"].notna() | df["Dalakolcumbasarisi"].notna()
    return _common_derivations(df.loc[attempted].copy())


def analysis_set(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to participants with a valid spleen stiffness measurement."""
    return df.loc[df["ssm"].notna()].copy()


def load_all(data_dir: Path | str | None = None):
    """Return (patients_attempted, patients_analysed, reference_analysed)."""
    pat = load_patients(data_dir)
    ref = load_reference(data_dir)
    return pat, analysis_set(pat), analysis_set(ref)


def describe_continuous(x: pd.Series, digits: int = 1) -> str:
    x = x.dropna()
    return f"{x.median():.{digits}f} ({x.quantile(0.25):.{digits}f}-{x.quantile(0.75):.{digits}f})"


def describe_binary(x: pd.Series) -> str:
    x = x.dropna()
    return f"{int(x.sum())} ({100 * x.mean():.1f}%)"

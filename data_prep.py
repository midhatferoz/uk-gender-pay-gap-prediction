"""Load, clean and reshape the UK Gender Pay Gap files into a forecasting panel.

Pipeline:  raw yearly CSVs  ->  load_all  ->  clean  ->  assign_sector  ->  build_panel
Each row of the final panel is one employer in one origin year t, with features
measured at t and a target describing the change in its median hourly pay gap
between t and t+1.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Raw header (lower-cased, punctuation removed)  ->  our column name.
# Normalising headers makes "EmployerId" and "Employer ID" equivalent.
COLUMN_MAP = {
    "employerid": "employer_id",
    "employername": "employer_name",
    "companynumber": "company_number",
    "siccodes": "sic_codes",
    "diffmeanhourlypercent": "gap_mean",
    "diffmedianhourlypercent": "gap_median",
    "diffmeanbonuspercent": "bonus_gap_mean",
    "diffmedianbonuspercent": "bonus_gap_median",
    "malebonuspercent": "male_bonus_pct",
    "femalebonuspercent": "female_bonus_pct",
    "malelowerquartile": "male_lower_q",
    "femalelowerquartile": "female_lower_q",
    "malelowermiddlequartile": "male_lowmid_q",
    "femalelowermiddlequartile": "female_lowmid_q",
    "maleuppermiddlequartile": "male_upmid_q",
    "femaleuppermiddlequartile": "female_upmid_q",
    "maletopquartile": "male_top_q",
    "femaletopquartile": "female_top_q",
    "employersize": "employer_size",
    "submittedafterthedeadline": "submitted_late",
    "duedate": "due_date",
    "datesubmitted": "date_submitted",
}

NUMERIC_COLS = [
    "gap_mean", "gap_median", "bonus_gap_mean", "bonus_gap_median",
    "male_bonus_pct", "female_bonus_pct",
    "male_lower_q", "female_lower_q", "male_lowmid_q", "female_lowmid_q",
    "male_upmid_q", "female_upmid_q", "male_top_q", "female_top_q",
]
WINSOR_COLS = ["gap_mean", "gap_median", "bonus_gap_mean", "bonus_gap_median"]

SIZE_MAP = {
    "lessthan250": 0, "250to499": 1, "500to999": 2,
    "1000to4999": 3, "5000to19999": 4, "20000ormore": 5,
}

# Feature groups (used for the RQ3 ablation).
BASE_FEATURES = [
    "gap_median", "gap_mean", "gap_prev", "change_prev",
    "employer_size_ord", "submitted_late", "days_late",
    "is_fashion", "is_retail_other","is_unknown",
]
BONUS_FEATURES = ["bonus_gap_mean", "bonus_gap_median", "male_bonus_pct", "female_bonus_pct"]
QUARTILE_FEATURES = [
    "female_lower_q", "female_lowmid_q", "female_upmid_q", "female_top_q", "fem_gradient",
]
FEATURE_SETS = {
    "BASE": BASE_FEATURES,
    "BASE+BONUS": BASE_FEATURES + BONUS_FEATURES,
    "BASE+QUARTILE": BASE_FEATURES + QUARTILE_FEATURES,
    "FULL": BASE_FEATURES + BONUS_FEATURES + QUARTILE_FEATURES,
}
FULL_FEATURES = FEATURE_SETS["FULL"]


def _norm(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


# --------------------------------------------------------------------------- load
def load_year(path, year):
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="latin-1")
    rename = {c: COLUMN_MAP[_norm(c)] for c in df.columns if _norm(c) in COLUMN_MAP}
    df = df.rename(columns=rename)
    df["report_year"] = year
    return df


def load_all(raw_dir=config.RAW_DIR, first=config.FIRST_YEAR, last=config.LAST_YEAR):
    """Read gpg_<year>.csv for every year found and stack them."""
    frames = []
    for year in range(first, last + 1):
        path = Path(raw_dir) / f"gpg_{year}.csv"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        frames.append(load_year(path, year))
    if not frames:
        raise FileNotFoundError(
            f"No gpg_<year>.csv files in {raw_dir}. Run download_data.py (or put the "
            "CSVs there manually). For a smoke test use make_synthetic_data.py."
        )
    df = pd.concat(frames, ignore_index=True)
    for col in COLUMN_MAP.values():           # make sure every column exists
        if col not in df.columns:
            df[col] = np.nan
    if "employer_id" not in df.columns or df["employer_id"].isna().all():
        df["employer_id"] = df["company_number"]
    return df


# -------------------------------------------------------------------------- clean
def size_to_ord(value):
    return SIZE_MAP.get(re.sub(r"[^a-z0-9]", "", str(value).lower()), np.nan)


def _parse_dates(series):
    parsed = pd.to_datetime(
        series,
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce"
    )

    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(
            series,
            dayfirst=True,
            errors="coerce"
        )

    return parsed


def clean(df, winsorize=config.WINSORIZE, tol=config.QUARTILE_SUM_TOLERANCE):
    """Type conversion, de-duplication, sanity checks, winsorising, derived columns."""
    log = {"rows_raw": int(len(df))}
    df = df.copy()

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
    df["employer_id"] = df["employer_id"].replace(r"^\s*$", np.nan, regex=True)
    df["company_number"] = df["company_number"].replace(r"^\s*$", np.nan, regex=True)

    df["employer_id"] = df["employer_id"].fillna(df["company_number"])
    df["employer_id"] = df["employer_id"].astype("string").str.strip()

    valid_id = (
    df["employer_id"].notna()
    & ~df["employer_id"].str.lower().isin(["", "nan", "none", "<na>"])
)

    df = df[valid_id & df["gap_median"].notna()]
    
    log["rows_after_missing_id_or_gap"] = int(len(df))

    df["submitted_late"] = (
        df["submitted_late"].astype(str).str.strip().str.upper().eq("TRUE").astype(int)
    )
    df["due_date"] = _parse_dates(df["due_date"])
    df["date_submitted"] = _parse_dates(df["date_submitted"])
    df["days_late"] = (df["date_submitted"] - df["due_date"]).dt.days.clip(lower=0, upper=365)

    # Employers can edit and resubmit: keep the latest submission per employer-year.
    n0 = len(df)
    df = (
        df.sort_values(["employer_id", "report_year", "date_submitted"], na_position="first")
        .drop_duplicates(["employer_id", "report_year"], keep="last")
    )
    log["duplicates_dropped"] = int(n0 - len(df))

    # Male% + female% in each pay quartile should be ~100; drop clear data-entry errors.
    pairs = [("male_lower_q", "female_lower_q"), ("male_lowmid_q", "female_lowmid_q"),
             ("male_upmid_q", "female_upmid_q"), ("male_top_q", "female_top_q")]
    sums = pd.concat([df[m] + df[f] for m, f in pairs], axis=1)
    bad = ((sums - 100).abs() > tol).any(axis=1)
    log["quartile_sum_errors_dropped"] = int(bad.sum())
    df = df[~bad]

    if winsorize:
        lo, hi = winsorize
        for c in WINSOR_COLS:
            ql, qh = df[c].quantile([lo, hi])
            df[c] = df[c].clip(ql, qh)
    log["winsorized"] = bool(winsorize)

    df["fem_gradient"] = df["female_top_q"] - df["female_lower_q"]
    df["employer_size_ord"] = df["employer_size"].map(size_to_ord)
    log["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), log


# ------------------------------------------------------------------------- sector
def classify_sector(sic):
    if pd.isna(sic) or not str(sic).strip():
        return "unknown"

    codes = re.findall(r"\d{4,5}", str(sic))

    if not codes:
        return "unknown"

    if any(c.startswith(config.FASHION_PREFIXES) for c in codes):
        return "fashion"

    if any(c.startswith(config.RETAIL_PREFIX) for c in codes):
        return "retail_other"

    return "other"


def assign_sector(df):
    df = df.copy()
    df["sector"] = df["sic_codes"].map(classify_sector)
    df["is_fashion"] = (df["sector"] == "fashion").astype(int)
    df["is_retail_other"] = (df["sector"] == "retail_other").astype(int)
    df["is_unknown"] = (df["sector"] == "unknown").astype(int)
    return df


# -------------------------------------------------------------------------- panel
def build_panel(df, exclude_pandemic=False, target_mode=config.TARGET_MODE):
    """One row per employer-origin year t that also reported in t+1.

    Features use information up to year t only. Lag features are NaN unless the
    employer also reported in t-1.
    """
    df = df.sort_values(["employer_id", "report_year"]).copy()
    g = df.groupby("employer_id")
    df["next_year"] = g["report_year"].shift(-1)
    df["gap_next"] = g["gap_median"].shift(-1)
    df["prev_year"] = g["report_year"].shift(1)

    prev_ok = df["prev_year"] == df["report_year"] - 1
    if exclude_pandemic:
        prev_ok &= ~df["prev_year"].isin(config.PANDEMIC_YEARS)
    df["gap_prev"] = g["gap_median"].shift(1).where(prev_ok)

    if target_mode == "absolute":
        df["change_prev"] = df["gap_median"].abs() - df["gap_prev"].abs()
        df["target_change"] = df["gap_next"].abs() - df["gap_median"].abs()
    else:
        df["change_prev"] = df["gap_median"] - df["gap_prev"]
        df["target_change"] = df["gap_next"] - df["gap_median"]

    panel = df[df["next_year"] == df["report_year"] + 1].copy()
    if exclude_pandemic:
        bad = panel["report_year"].isin(config.PANDEMIC_YEARS) | panel["next_year"].isin(
            config.PANDEMIC_YEARS
        )
        panel = panel[~bad]
    return panel.reset_index(drop=True)


# ------------------------------------------------------------------------ summary
def data_summaries(df, panel):
    """Employer counts by year and sector (cleaned data) and usable pairs (panel)."""
    reports = df.pivot_table(index="report_year", columns="sector", values="employer_id",
                             aggfunc="nunique", fill_value=0)
    reports["total"] = reports.sum(axis=1)
    pairs = panel.pivot_table(index="report_year", columns="sector", values="employer_id",
                              aggfunc="nunique", fill_value=0)
    pairs["total"] = pairs.sum(axis=1)
    return reports, pairs

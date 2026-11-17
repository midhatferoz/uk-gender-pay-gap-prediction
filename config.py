"""Central settings for the UK Gender Pay Gap forecasting project.

Everything you might want to change for the paper (years, sector definitions,
validation folds, stable band, ...) lives here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"

# ---- data -------------------------------------------------------------------
FIRST_YEAR = 2017          # reporting year 2017-18
LAST_YEAR = 2025           # reporting year 2025-26 (the 2026-27 file is incomplete)
RANDOM_STATE = 42

# ---- sector definitions (UK SIC 2007 codes; verify against your data) --------
# An employer is "fashion" if ANY of its SIC codes starts with one of these.
#   14      manufacture of wearing apparel
#   4771    retail sale of clothing in specialised stores
#   4772    retail sale of footwear and leather goods
#   46420   wholesale of clothing and footwear
FASHION_PREFIXES = ("14", "4771", "4772", "46420")
# Any other employer with a code starting with "47" is "retail_other"
# (supermarkets, pharmacies, DIY, ...). Everyone else is "other".
RETAIL_PREFIX = "47"

# ---- cleaning ----------------------------------------------------------------
WINSORIZE = None       # clip gap columns at these quantiles (None = off)
QUARTILE_SUM_TOLERANCE = 2.0    # male% + female% in a pay quartile should be ~100
# Reporting years affected by COVID-19 (2019-20 reporting was suspended and
# coverage dropped; 2020-21 snapshot dates fall in the furlough period).
# Only used when you run with --exclude-pandemic.
PANDEMIC_YEARS = (2019, 2020)

# ---- target ------------------------------------------------------------------
# "signed"   : next year's median gap minus this year's (positive = men's pay gap grows)
# "absolute" : |next gap| - |this gap| (positive = gap moves further from parity,
#              whichever gender is paid more)
TARGET_MODE = "signed"
STABLE_BAND = 1.0               # +/- percentage points counted as "stable"

# ---- validation (origin year = year of the features; target is origin + 1) ---
TUNE_ORIGINS = (2020, 2021, 2022)   # expanding-window folds for hyper-parameter tuning
TEST_ORIGINS = (2023, 2024)         # backtest folds; 2024 -> 2025 is the final hold-out
SHAP_ORIGINS = (2024,)              # rows explained with SHAP (model trained on earlier origins)
SHAP_MIN_GROUP = 30                 # warn if a sector group in the SHAP sample is smaller
N_BOOT = 500                        # employer-level bootstrap resamples

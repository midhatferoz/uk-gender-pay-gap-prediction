# Predicting UK Gender Pay Gap Changes: Fashion & Retail Focus

End-to-end predictive analytics pipeline. For every UK employer that reports in two consecutive
years, it predicts the **change in the median hourly gender pay gap** from year *t* to *t+1*,
compares Ridge, Random Forest and XGBoost against naive baselines, evaluates by sector
(fashion / other retail / everything else), tests whether bonus and pay-quartile features add
information, and explains the model with SHAP.

> **Status of testing.** The code was run end to end on *synthetic* data with the same schema as
> the real files (`make_synthetic_data.py`). It has **not** been run on the real data, and
> `download_data.py` plus the real XGBoost/SHAP libraries were not testable in the build
> environment (no internet). Run the smoke test below first, then check each output on the real
> data before trusting it. Never report numbers from the synthetic data.

## Quick start

```bash
pip install -r requirements.txt

# 1) smoke test on fake data (about a minute)
python make_synthetic_data.py
python run_pipeline.py --raw-dir data/synthetic --tag smoke_test

# 2) real data
python download_data.py            # or download manually, see below
python run_pipeline.py             # main analysis -> outputs/main/
bash run_all.sh                    # main + robustness checks
```

Manual download: https://gender-pay-gap.service.gov.uk/viewing/download. Save the yearly files as
`data/raw/gpg_2017.csv` ... `data/raw/gpg_2025.csv` (the number is the *start* of the reporting
year, so `2024` is reporting year 2024-25). Skip the partial 2026-27 file.

## Project layout

| File | Purpose |
|---|---|
| `config.py` | All settings: years, sector SIC prefixes, validation folds, stable band, bootstrap size |
| `data_prep.py` | Load, clean, sector assignment, panel construction, feature groups |
| `modelling.py` | Baselines, models, time-based tuning, backtest, bootstrap CIs, classification, ablation |
| `analysis.py` | Figures and SHAP |
| `run_pipeline.py` | Runs everything (`--help` for options) |
| `make_synthetic_data.py` | Fake data for smoke testing only |
| `download_data.py`, `run_all.sh` | Data download; main + robustness runs |

## What the pipeline does

1. **Load and clean.** Normalises headers, converts types, keeps the latest submission per
   employer-year, drops rows where male% + female% in any pay quartile is off from 100 by more than
   2 points, winsorises gap columns at the 1st/99th percentile.
2. **Sectors.** From the SIC codes: `fashion` (any code starting 14, 4771, 4772 or 46420),
   `retail_other` (other codes starting 47), `other`. **Verify these prefixes against your data**
   and edit `config.py`; report the group sizes in the paper.
3. **Panel.** One row per employer-origin year *t* with a *t+1* record. Features use year *t*
   (and *t-1* where available). Target `target_change = gap(t+1) - gap(t)`.
   `--target absolute` uses |gap| instead (see "widen" caveat below).
4. **Validation.** Expanding window by origin year. Tuning folds: origins 2020, 2021, 2022.
   Backtest folds: 2023 and 2024 (2024 to 2025 is the final hold-out). Training rows always have an
   earlier origin than the test rows. Backtest folds are never used for tuning.
5. **Models.** `NoChange` and `MeanReversion` baselines; Ridge, Random Forest, XGBoost
   (falls back to scikit-learn HistGradientBoosting if XGBoost is missing, and says so in the model name).
6. **Metrics.** MAE, RMSE, R², and *skill* = 1 - MAE/MAE(no change), with 95% bootstrap intervals
   that resample whole employers. Widen / stable / narrow classes are derived from the predicted
   and true change (default stable band +/-1 pp): balanced accuracy, macro-F1, per-class recall.
7. **Research questions.** RQ1: model vs baselines (`test_metrics_overall.csv`). RQ2: skill and SHAP
   by sector (`test_metrics_by_sector.csv`, `shap_*`). RQ3: ablation BASE vs +BONUS vs +QUARTILE vs
   FULL with paired bootstrap (`ablation.csv`).

## Outputs (`outputs/<tag>/`)

Tables: `employers_per_year_by_sector`, `usable_pairs_by_origin_by_sector`, `tuning_results`,
`backtest_predictions`, `test_metrics_overall / by_sector / by_origin`, `classification_metrics`,
`class_shares`, `confusion_matrix_<model>`, `ablation`, `shap_mean_abs_by_sector`,
`shap_rank_agreement`. Figures: gap trend by sector, target distribution, model comparison, skill by
sector, ablation, SHAP summary, SHAP by sector. `run_summary.json` records settings and versions.

## Things to know before writing the paper

* **Beat the baseline or say so.** Persistence is strong. If skill is near zero, that is a valid
  finding; the CIs tell you how much to claim.
* **Fashion sample size.** Check `usable_pairs_by_origin_by_sector.csv` and the `n_rows` column in
  `test_metrics_by_sector.csv`. Small groups have wide intervals; SHAP for a small group is
  unstable (the code warns when it is under 30 rows).
* **"Widen" with negative gaps.** With the signed target, a rise in the gap for an employer
  where women are paid more moves it *toward* parity. Run `--target absolute` as a robustness check.
* **Pandemic years.** `--exclude-pandemic` drops transitions touching reporting years 2019 and
  2020 (see `config.PANDEMIC_YEARS`). I believe 2019-20 reporting was suspended (the file is much
  smaller) and 2020-21 snapshots fall in the furlough period; confirm this from the official guidance and cite it.
* **Selection.** Only employers with 250+ staff report, and only employers present in consecutive
  years enter the panel. State this as a limitation.
* **Interpretation.** SHAP shows what the model uses, not causes. A pay gap is not unequal pay for
  equal work.
* **Ablation uses the tuned full-feature hyper-parameters** for every subset (a simplification).
* **No sector-specific models** (one pooled model, evaluated by sector). A natural extension.
* **Hyper-parameter grids are small** to keep run time low; widen them in `modelling.get_grids()`.

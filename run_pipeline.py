"""End-to-end pipeline: load -> clean -> panel -> tune -> backtest -> evaluate -> explain.

Examples
--------
python run_pipeline.py                              # main analysis on data/raw
python run_pipeline.py --raw-dir data/synthetic --tag smoke_test
python run_pipeline.py --exclude-pandemic --tag robust_no_pandemic
python run_pipeline.py --no-winsorize --tag robust_no_winsor
python run_pipeline.py --target absolute --tag robust_abs_target
python run_pipeline.py --stable-band 2 --tag robust_band2
"""


import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import analysis
import config
import data_prep as dp
import modelling as md


# 2023 is used for model selection.
# 2024 is reserved for final evaluation.
DEV_ORIGINS = (2023,)
FINAL_ORIGINS = (2024,)


def parse_args():
    parser = argparse.ArgumentParser(
        description="UK Gender Pay Gap prediction pipeline"
    )

    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--no-winsorize", action="store_true")
    parser.add_argument("--exclude-pandemic", action="store_true")
    parser.add_argument(
        "--target",
        choices=["signed", "absolute"],
        default=config.TARGET_MODE
    )
    parser.add_argument(
        "--stable-band",
        type=float,
        default=config.STABLE_BAND
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=config.N_BOOT
    )
    parser.add_argument("--skip-tuning", action="store_true")

    return parser.parse_args()


def calculate_metrics(preds, model_cols):
    """Calculate MAE, RMSE and R2 for each model."""

    rows = []

    for model in model_cols:
        valid = preds[["y_true", model]].dropna()

        if len(valid) == 0:
            continue

        y_true = valid["y_true"]
        y_pred = valid[model]

        rows.append({
            "model": model,
            "n": len(valid),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred) if len(valid) > 1 else np.nan
        })

    return pd.DataFrame(rows)


def evaluate_groups(preds, model_cols, n_boot):
    """Evaluate model performance across the main sector groups."""

    groups = {
        "all": preds,
        "fashion": preds[preds["sector"] == "fashion"],
        "retail_other": preds[preds["sector"] == "retail_other"],
        "all_retail": preds[
            preds["sector"].isin(["fashion", "retail_other"])
        ],
        "other": preds[preds["sector"] == "other"]
    }

    point_results = []
    bootstrap_results = []

    for group_name, group_data in groups.items():

        if len(group_data) < 30:
            print(f"Skipping {group_name}: fewer than 30 observations.")
            continue

        point = calculate_metrics(group_data, model_cols)
        point.insert(0, "group", group_name)
        point_results.append(point)

        boot = md.bootstrap_summary(
            group_data,
            model_cols,
            baseline="NoChange",
            n_boot=n_boot
        )

        boot.insert(0, "group", group_name)
        bootstrap_results.append(boot)

    point_table = (
        pd.concat(point_results, ignore_index=True)
        if point_results else pd.DataFrame()
    )

    bootstrap_table = (
        pd.concat(bootstrap_results, ignore_index=True)
        if bootstrap_results else pd.DataFrame()
    )

    return point_table, bootstrap_table


def main():

    args = parse_args()

    output_dir = config.OUT_DIR / args.tag
    fig_dir = output_dir / "figures"
    tab_dir = output_dir / "tables"

    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== GENDER PAY GAP PIPELINE ==========\n")

    # --------------------------------------------------
    # 1. LOAD AND CLEAN DATA
    # --------------------------------------------------

    print("Step 1: Loading annual datasets...")

    raw = dp.load_all(args.raw_dir)

    winsorize = None if args.no_winsorize else config.WINSORIZE

    data, cleaning_log = dp.clean(
        raw,
        winsorize=winsorize
    )

    data = dp.assign_sector(data)

    print("Cleaned data shape:", data.shape)
    print("\nSector distribution:")
    print(data["sector"].value_counts(dropna=False))

    # --------------------------------------------------
    # 2. BUILD EMPLOYER-YEAR PANEL
    # --------------------------------------------------

    print("\nStep 2: Building employer-year panel...")

    panel = dp.build_panel(
        data,
        exclude_pandemic=args.exclude_pandemic,
        target_mode=args.target
    )

    if panel.empty:
        raise ValueError(
            "The employer-year panel is empty. "
            "Check the annual datasets and reporting years."
        )

    print("Panel shape:", panel.shape)
    print("\nPanel sector distribution:")
    print(panel["sector"].value_counts(dropna=False))

    # --------------------------------------------------
    # 3. SAVE DATA SUMMARIES
    # --------------------------------------------------

    print("\nStep 3: Saving data summaries...")

    reports, pairs = dp.data_summaries(data, panel)

    reports.to_csv(
        tab_dir / "employers_per_year_by_sector.csv"
    )

    pairs.to_csv(
        tab_dir / "consecutive_employer_pairs_by_sector.csv"
    )

    data.to_csv(
        tab_dir / "cleaned_data.csv",
        index=False
    )

    panel.to_csv(
        tab_dir / "employer_year_panel.csv",
        index=False
    )

    # --------------------------------------------------
    # 4. EXPLORATORY PLOTS
    # --------------------------------------------------

    print("\nStep 4: Creating exploratory plots...")

    analysis.plot_gap_trend(
        data,
        fig_dir / "gender_pay_gap_trend.png"
    )

    analysis.plot_change_distribution(
        panel,
        fig_dir / "target_change_distribution.png"
    )

    # --------------------------------------------------
    # 5. MODEL FEATURES AND HYPERPARAMETER TUNING
    # --------------------------------------------------

    print("\nStep 5: Preparing model features...")

    features = dp.FULL_FEATURES
    grids = md.get_grids()
    defaults = md.default_params()

    if args.skip_tuning:
        print("Hyperparameter tuning skipped.")
        best_params = defaults.copy()
        tuning_results = pd.DataFrame()

    else:
        print("Running hyperparameter tuning...")

        best_params, tuning_results = md.tune(
            panel,
            features,
            grids
        )

        # Ensure every model has a parameter configuration.
        best_params = {
            **defaults,
            **best_params
        }

    tuning_results.to_csv(
        tab_dir / "tuning_results.csv",
        index=False
    )

    with open(tab_dir / "best_parameters.json", "w") as f:
        json.dump(best_params, f, indent=4, default=str)

    model_cols = md.BASELINES + list(best_params.keys())

    print("\nSelected model configurations:")
    print(best_params)

    # --------------------------------------------------
    # 6. DEVELOPMENT BACKTEST: 2023
    # --------------------------------------------------

    print("\nStep 6: Running development backtest (2023)...")

    dev_preds = md.backtest(
        panel,
        features,
        best_params,
        test_origins=DEV_ORIGINS,
        mode=args.target
    )

    if dev_preds.empty:
        raise ValueError(
            "No development predictions were generated for 2023."
        )

    dev_preds.to_csv(
        tab_dir / "development_predictions_2023.csv",
        index=False
    )

    dev_metrics = calculate_metrics(
        dev_preds,
        model_cols
    )

    dev_metrics.to_csv(
        tab_dir / "development_metrics_2023.csv",
        index=False
    )

    # Select the best trained ML model using 2023 MAE only.
    candidate_metrics = dev_metrics[
        dev_metrics["model"].isin(best_params.keys())
    ]

    if candidate_metrics.empty:
        raise ValueError(
            "No ML model produced valid development metrics."
        )

    best_model = candidate_metrics.sort_values("MAE").iloc[0]["model"]

    print("\nBest model selected using 2023 MAE:", best_model)

    # --------------------------------------------------
    # 7. FINAL HOLDOUT: 2024
    # --------------------------------------------------

    print("\nStep 7: Running final holdout evaluation (2024)...")

    final_preds = md.backtest(
        panel,
        features,
        best_params,
        test_origins=FINAL_ORIGINS,
        mode=args.target
    )

    if final_preds.empty:
        raise ValueError(
            "No final predictions were generated for 2024."
        )

    final_preds.to_csv(
        tab_dir / "final_predictions_2024.csv",
        index=False
    )

    # --------------------------------------------------
    # 8. FINAL PERFORMANCE METRICS
    # --------------------------------------------------

    print("\nStep 8: Calculating final performance metrics...")

    final_metrics = calculate_metrics(
        final_preds,
        model_cols
    )

    final_metrics.to_csv(
        tab_dir / "final_metrics_2024.csv",
        index=False
    )

    print("\nFinal model metrics:")
    print(final_metrics.to_string(index=False))

    # --------------------------------------------------
    # 9. EMPLOYER-LEVEL BOOTSTRAP
    # --------------------------------------------------

    print("\nStep 9: Calculating bootstrap confidence intervals...")

    bootstrap = md.bootstrap_summary(
        final_preds,
        model_cols,
        baseline="NoChange",
        n_boot=args.n_boot
    )

    bootstrap.to_csv(
        tab_dir / "bootstrap_results_2024.csv",
        index=False
    )

    # --------------------------------------------------
    # 10. SECTOR-LEVEL EVALUATION
    # --------------------------------------------------

    print("\nStep 10: Evaluating sector performance...")

    sector_metrics, sector_bootstrap = evaluate_groups(
        final_preds,
        model_cols,
        args.n_boot
    )

    sector_metrics.to_csv(
        tab_dir / "sector_metrics_2024.csv",
        index=False
    )

    sector_bootstrap.to_csv(
        tab_dir / "sector_bootstrap_2024.csv",
        index=False
    )

    # --------------------------------------------------
    # 11. CLASSIFICATION EVALUATION
    # --------------------------------------------------

    print("\nStep 11: Classification evaluation...")

    classification, class_shares = md.classification_table(
    final_preds,
    model_cols,
    band=args.stable_band
    )

    classification.to_csv(
    tab_dir / "classification_results_2024.csv",
    index=False
    )

# Save actual class distribution
    pd.DataFrame(
        list(class_shares.items()),
        columns=["class", "proportion"]
    ).to_csv(
        tab_dir / "classification_class_shares_2024.csv",
        index=False
    )



    confusion_tables = []

    for model in model_cols:
        cm = md.confusion(
            final_preds,
            model,
            band=args.stable_band
        )

        cm.to_csv(
            tab_dir / f"confusion_matrix_{model}.csv"
        )

    # --------------------------------------------------
    # 12. FEATURE ABLATION
    # --------------------------------------------------

    print("\nStep 12: Running feature ablation...")

    ablation_results = md.ablation(
        panel,
        dp.FEATURE_SETS,
        best_params[md.GB_NAME],
        test_origins=DEV_ORIGINS
    )

    ablation_results.to_csv(
        tab_dir / "ablation_results_2023.csv",
        index=False
    )

    # --------------------------------------------------
    # 13. SHAP EXPLANATION
    # --------------------------------------------------

    print("\nStep 13: Running SHAP analysis...")

    shap_result = analysis.run_shap(
        panel,
        features,
        best_params[md.GB_NAME],
        fig_dir,
        tab_dir
    )

    # --------------------------------------------------
    # 14. SAVE RUN SUMMARY
    # --------------------------------------------------

    print("\nStep 14: Saving run summary...")

    summary = {
        "project": "UK Gender Pay Gap Prediction",
        "target_mode": args.target,
        "stable_band": args.stable_band,
        "first_year": config.FIRST_YEAR,
        "last_year": config.LAST_YEAR,
        "development_origins": list(DEV_ORIGINS),
        "final_holdout_origins": list(FINAL_ORIGINS),
        "best_model": str(best_model),
        "best_parameters": best_params,
        "cleaning_log": cleaning_log,
        "raw_rows": int(len(raw)),
        "cleaned_rows": int(len(data)),
        "panel_rows": int(len(panel)),
        "feature_count": len(features),
        "features": features,
        "exclude_pandemic": args.exclude_pandemic,
        "winsorization": str(winsorize),
        "python_version": platform.python_version(),
        "bootstrap_iterations": args.n_boot
    }

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)

    print("\n========== PIPELINE COMPLETED ==========")
    print("Best model:", best_model)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()

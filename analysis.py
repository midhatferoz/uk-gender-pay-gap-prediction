"""Figures and SHAP explanations."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
import modelling

SECTOR_ORDER = ["fashion", "retail_other", "other"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gap_trend(df, path):
    """Median of employers' median hourly gap by reporting year and sector."""
    t = df.groupby(["report_year", "sector"])["gap_median"].median().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SECTOR_ORDER:
        if s in t:
            ax.plot(t.index, t[s], marker="o", label=s)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Reporting year (start of reporting year)")
    ax.set_ylabel("Median of employers' median hourly gap (%)")
    ax.set_title("Gender pay gap by sector (positive = men paid more)")
    ax.legend()
    _save(fig, path)


def plot_change_distribution(panel, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SECTOR_ORDER:
        v = panel.loc[panel.sector == s, "target_change"].clip(-15, 15)
        if len(v) > 10:
            ax.hist(v, bins=60, alpha=0.5, density=True, label=f"{s} (n={len(v)})")
    ax.set_xlabel("Next-year change in median hourly gap (pp, clipped at +/-15)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of the prediction target")
    ax.legend()
    _save(fig, path)


def plot_model_comparison(summary, baseline, path):
    s = summary.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(7, 4))
    yerr = np.vstack([s["MAE"] - s["MAE_lo"], s["MAE_hi"] - s["MAE"]])
    ax.bar(s["model"], s["MAE"], yerr=yerr, capsize=4, color="#4C72B0")
    ax.set_ylabel("MAE of predicted change (pp)")
    ax.set_title("Backtest error by model (95% employer-level bootstrap CI)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, path)


def plot_sector_skill(by_sector, path):
    d = by_sector[~by_sector["model"].isin(modelling.BASELINES)]
    groups = [g for g in ["fashion", "retail_other", "all_retail", "other", "all"]
              if g in set(d["group"])]
    models = list(d["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        sub = d[d.model == m].set_index("group").reindex(groups)
        yerr = np.vstack([(sub["skill_vs_NoChange"] - sub["skill_lo"]).fillna(0),
                          (sub["skill_hi"] - sub["skill_vs_NoChange"]).fillna(0)])
        ax.bar(np.arange(len(groups)) + i * width, sub["skill_vs_NoChange"], width,
               yerr=yerr, capsize=3, label=m)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(groups)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Skill vs no-change baseline (1 - MAE ratio)")
    ax.set_title("Predictive skill by sector group (95% CI)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_ablation(abl_summary, path):
    d = abl_summary[abl_summary["model"] != "BASE"].copy()
    diff_col = [c for c in d.columns if c.startswith("MAE_diff_vs_")][0]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    yerr = np.vstack([d[diff_col] - d["diff_lo"], d["diff_hi"] - d[diff_col]])
    ax.barh(d["model"], d[diff_col], xerr=yerr, capsize=4, color="#55A868")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Change in MAE vs BASE features (pp); negative = better")
    ax.set_title("Do bonus and pay-quartile features add information?")
    _save(fig, path)


# -------------------------------------------------------------------------- SHAP


def run_shap(panel, features, gb_params, fig_dir, tab_dir,
             shap_origins=config.SHAP_ORIGINS):
    """
    SHAP analysis for the HistGradientBoosting model.

    Trains on years before the SHAP holdout years and explains
    a stratified sample of out-of-sample predictions.
    """

    try:
        import shap
    except ImportError:
        print("  [skip] Install SHAP using: pip install shap")
        return None

    # --------------------------------------------------
    # 1. Prepare training and test data
    # --------------------------------------------------
    train = panel[panel.report_year < min(shap_origins)].copy()
    test = panel[panel.report_year.isin(shap_origins)].copy()

    if train.empty or test.empty:
        print("  [skip] SHAP: no train/test rows for SHAP_ORIGINS.")
        return None

    if "sector" not in test.columns:
        print("  [skip] SHAP: sector column missing from test data.")
        return None

    missing_features = [f for f in features if f not in panel.columns]

    if missing_features:
        print(f"  [skip] SHAP: missing features: {missing_features}")
        return None

    print("  Training HistGradientBoosting for SHAP...")

    model = modelling.build_model("HistGradientBoosting", gb_params)
    model.fit(train[features], train["target_change"])

    # --------------------------------------------------
    # 2. Create sector-stratified test sample
    # --------------------------------------------------
    # Explicit sampling avoids groupby.apply column handling issues.
    sampled_groups = []

    for sector, group in test.groupby("sector", dropna=False, sort=False):
        sampled_groups.append(
            group.sample(
                n=min(len(group), 250),
                random_state=config.RANDOM_STATE
            )
        )

    test_sample = pd.concat(
        sampled_groups,
        axis=0
    ).reset_index(drop=True)

    # Preserve sector labels separately from model features.
    sample_sectors = test_sample["sector"].to_numpy()

    X_test_sample = test_sample[features].copy()

    # --------------------------------------------------
    # 3. Prepare background data
    # --------------------------------------------------
    background = train[features].sample(
        n=min(len(train), 50),
        random_state=config.RANDOM_STATE
    ).copy()

    # --------------------------------------------------
    # 4. Define prediction function
    # --------------------------------------------------
    def predict_fn(x):
        if isinstance(x, pd.DataFrame):
            x_df = x[features]
        else:
            x_df = pd.DataFrame(x, columns=features)

        return model.predict(x_df)

    # --------------------------------------------------
    # 5. Calculate permutation SHAP values
    # --------------------------------------------------
    print("  Calculating permutation SHAP values...")

    explainer = shap.PermutationExplainer(
        predict_fn,
        background
    )

    explanation = explainer(
        X_test_sample,
        max_evals=2 * len(features) + 1
    )

    values = np.asarray(explanation.values)

    # --------------------------------------------------
    # 6. Global SHAP summary plot
    # --------------------------------------------------
    shap.summary_plot(
        values,
        X_test_sample,
        show=False,
        max_display=15
    )

    plt.title("SHAP summary: drivers of next-year change in the gap")

    _save(
        plt.gcf(),
        fig_dir / "fig_shap_summary.png"
    )

    # --------------------------------------------------
    # 7. Mean absolute SHAP importance by sector
    # --------------------------------------------------
    imp = pd.DataFrame(
        np.abs(values),
        columns=features
    )

    # Use the separately preserved sector labels.
    imp["sector"] = sample_sectors

    by_sector = imp.groupby("sector")[features].mean()
    by_sector.loc["all"] = imp[features].mean()

    counts = imp["sector"].value_counts()

    by_sector.to_csv(
        tab_dir / "shap_mean_abs_by_sector.csv"
    )

    for sector in SECTOR_ORDER:
        if counts.get(sector, 0) < config.SHAP_MIN_GROUP:
            print(
                f"  [warn] only {counts.get(sector, 0)} "
                f"{sector} rows in the SHAP sample; "
                "interpret its SHAP profile with care."
            )

    # --------------------------------------------------
    # 8. Plot top 10 features by sector
    # --------------------------------------------------
    top = (
        by_sector.loc["all"]
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    groups = [
        s for s in SECTOR_ORDER
        if s in by_sector.index
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    width = 0.8 / max(len(groups), 1)

    for i, sector in enumerate(groups):
        ax.barh(
            np.arange(len(top)) + i * width,
            by_sector.loc[sector, top].to_numpy(),
            width,
            label=f"{sector} (n={counts.get(sector, 0)})"
        )

    ax.set_yticks(
        np.arange(len(top)) + width * (len(groups) - 1) / 2
    )

    ax.set_yticklabels(top)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (percentage points)")
    ax.set_title("Feature importance by sector (hold-out sample)")
    ax.legend(fontsize=8)

    _save(
        fig,
        fig_dir / "fig_shap_by_sector.png"
    )

    # --------------------------------------------------
    # 9. Compare fashion and other feature importance
    # --------------------------------------------------
    agree = {}

    if "fashion" in by_sector.index and "other" in by_sector.index:
        rho, p = spearmanr(
            by_sector.loc["fashion"],
            by_sector.loc["other"]
        )

        agree = {
            "spearman_fashion_vs_other": float(rho),
            "p_value": float(p)
        }

        pd.Series(agree).to_csv(
            tab_dir / "shap_rank_agreement.csv"
        )

    print("  SHAP analysis completed.")

    return {
        "by_sector": by_sector,
        "agreement": agree,
        "n_rows": int(len(test_sample))
    }



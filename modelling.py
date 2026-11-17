"""Models, time-based tuning, backtesting, metrics and employer-level bootstrap.

Validation logic (important for the paper):
  * An "origin year" T means features measured in year T, target = change to T+1.
  * For every fold, training rows have origin < T and the test rows have origin == T,
    so a model never sees information from the future of the rows it predicts.
  * Hyper-parameters are tuned on TUNE_ORIGINS only; the backtest folds
    (TEST_ORIGINS) are never used for tuning.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:                      # pragma: no cover
    HAS_XGB = False

GB_NAME = "XGBoost" if HAS_XGB else "HistGradientBoosting"
BASELINES = ["NoChange", "MeanReversion"]
LABELS = ["decrease", "stable", "increase"]


# --------------------------------------------------------------------- baselines
class NoChange:
    """Predicts that next year's gap equals this year's (change = 0)."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MeanReversion:
    """Linear regression of the change on the current gap level only."""
    def __init__(self, mode):
        self.mode = mode

    def _x(self, X):
        v = X["gap_median"].to_numpy(dtype=float)
        return (np.abs(v) if self.mode == "absolute" else v)[:, None]

    def fit(self, X, y):
        self.lr = LinearRegression().fit(self._x(X), y)
        return self

    def predict(self, X):
        return self.lr.predict(self._x(X))


# ------------------------------------------------------------------------ models
def get_grids():
    grids = {
        "Ridge": {"alpha": [0.1, 1, 10, 100]},
        "RandomForest": {"min_samples_leaf": [10, 30], "max_features": [0.5, 1.0]},
    }
    if HAS_XGB:
        grids[GB_NAME] = {"max_depth": [3, 5], "min_child_weight": [5, 20]}
    else:
        grids[GB_NAME] = {"max_depth": [3, 5], "min_samples_leaf": [20, 100]}
    return grids


def default_params():
    return {
        "Ridge": {"alpha": 10},
        "RandomForest": {"min_samples_leaf": 20, "max_features": 0.5},
        GB_NAME: ({"max_depth": 4, "min_child_weight": 10} if HAS_XGB
                  else {"max_depth": 4, "min_samples_leaf": 50}),
    }


def build_model(name, params):
    rs = config.RANDOM_STATE
    if name == "Ridge":
        return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("sc", StandardScaler()),
                         ("m", Ridge(**params))])
    if name == "RandomForest":
        return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("m", RandomForestRegressor(n_estimators=300, random_state=rs,
                                                     n_jobs=-1, **params))])
    if name == "XGBoost":
        return XGBRegressor(n_estimators=400, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=rs, n_jobs=-1,
                            tree_method="hist", **params)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                             random_state=rs, **params)
    raise ValueError(name)


# ------------------------------------------------------------------------ tuning
def tune(panel, features, grids, tune_origins=config.TUNE_ORIGINS):
    """Expanding-window grid search; returns (best_params_by_model, results_table)."""
    rows = []
    for name, grid in grids.items():
        for params in ParameterGrid(grid):
            maes = []
            for T in tune_origins:
                train, val = panel[panel.report_year < T], panel[panel.report_year == T]
                if train.empty or val.empty:
                    continue
                model = build_model(name, params).fit(train[features], train["target_change"])
                maes.append(mean_absolute_error(val["target_change"], model.predict(val[features])))
            if maes:
                rows.append({"model": name, "params": str(params), "mean_val_MAE": np.mean(maes),
                             "n_folds": len(maes), "_params": params})
    res = pd.DataFrame(rows)
    best = {}
    for name in grids:
        sub = res[res.model == name]
        if len(sub):
            best[name] = sub.loc[sub["mean_val_MAE"].idxmin(), "_params"]
    return best, res.drop(columns="_params")


# ---------------------------------------------------------------------- backtest
def _fold_frames(panel, T):
    return panel[panel.report_year < T], panel[panel.report_year == T]


def backtest(panel, features, best_params, test_origins=config.TEST_ORIGINS,
             mode=config.TARGET_MODE):
    """Predictions of every model on every backtest fold (one row per employer-origin)."""
    frames = []
    for T in test_origins:
        train, test = _fold_frames(panel, T)
        if train.empty or test.empty:
            print(f"  [warn] origin {T}: no train or test rows, skipped")
            continue
        out = test[["employer_id", "report_year", "sector"]].copy()
        out["y_true"] = test["target_change"].to_numpy()
        out["NoChange"] = NoChange().fit(train, None).predict(test)
        out["MeanReversion"] = MeanReversion(mode).fit(train, train["target_change"]).predict(test)
        for name, params in best_params.items():
            model = build_model(name, params).fit(train[features], train["target_change"])
            out[name] = model.predict(test[features])
        frames.append(out)
    if not frames:
        raise RuntimeError("No backtest fold had data. Check TEST_ORIGINS in config.py.")
    return pd.concat(frames, ignore_index=True)


def ablation(panel, feature_sets, gb_params, test_origins=config.TEST_ORIGINS):
    """Retrain the boosted model on each feature subset (same hyper-parameters)."""
    frames = []
    for T in test_origins:
        train, test = _fold_frames(panel, T)
        if train.empty or test.empty:
            continue
        out = test[["employer_id", "report_year", "sector"]].copy()
        out["y_true"] = test["target_change"].to_numpy()
        for label, feats in feature_sets.items():
            model = build_model(GB_NAME, gb_params).fit(train[feats], train["target_change"])
            out[label] = model.predict(test[feats])
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------- metrics
def point_metrics(preds, model_cols):
    rows = []
    for m in model_cols:
        y, p = preds["y_true"], preds[m]
        rows.append({"model": m, "MAE": mean_absolute_error(y, p),
                     "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": r2_score(y, p)})
    return pd.DataFrame(rows)


def bootstrap_summary(preds, model_cols, baseline, n_boot=config.N_BOOT,
                      seed=config.RANDOM_STATE):
    """MAE, skill and paired MAE difference vs `baseline`, with employer-level bootstrap CIs.

    Resampling whole employers (not rows) respects the fact that one employer
    contributes several correlated rows across origin years.
    """
    if len(preds) == 0:
        return pd.DataFrame()
    codes, uniques = pd.factorize(preds["employer_id"])
    n_emp = len(uniques)
    cnt = np.bincount(codes, minlength=n_emp).astype(float)
    err = {m: np.bincount(codes, weights=(preds["y_true"] - preds[m]).abs().to_numpy(),
                          minlength=n_emp) for m in model_cols}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_emp, size=(n_boot, n_emp))
    denom = cnt[idx].sum(axis=1)
    boot = {m: err[m][idx].sum(axis=1) / denom for m in model_cols}
    point = {m: err[m].sum() / cnt.sum() for m in model_cols}

    def ci(a):
        return np.percentile(a, 2.5), np.percentile(a, 97.5)

    rows = []
    for m in model_cols:
        mae_lo, mae_hi = ci(boot[m])
        d_lo, d_hi = ci(boot[m] - boot[baseline])
        s_lo, s_hi = ci(1 - boot[m] / boot[baseline])
        rows.append({
            "model": m, "n_rows": len(preds), "n_employers": n_emp,
            "MAE": point[m], "MAE_lo": mae_lo, "MAE_hi": mae_hi,
            "MAE_diff_vs_" + baseline: point[m] - point[baseline],
            "diff_lo": d_lo, "diff_hi": d_hi,
            "skill_vs_" + baseline: 1 - point[m] / point[baseline],
            "skill_lo": s_lo, "skill_hi": s_hi,
        })
    return pd.DataFrame(rows)


def to_class(values, band=config.STABLE_BAND):
    return np.where(
        values > band, "increase",
        np.where(values < -band, "decrease", "stable")
    )

def classification_table(preds, model_cols, band=config.STABLE_BAND):
    """Derive widen / stable / narrow classes from true and predicted changes."""
    y = to_class(preds["y_true"], band)
    rows = []
    for m in model_cols:
        p = to_class(preds[m], band)
        rec = recall_score(y, p, labels=LABELS, average=None, zero_division=0)
        row = {"model": m,
               "balanced_accuracy": balanced_accuracy_score(y, p),
               "macro_F1": f1_score(y, p, labels=LABELS, average="macro", zero_division=0)}
        row.update({f"recall_{lab}": r for lab, r in zip(LABELS, rec)})
        rows.append(row)
    shares = {lab: float((y == lab).mean()) for lab in LABELS}
    return pd.DataFrame(rows), shares


def confusion(preds, model, band=config.STABLE_BAND):
    y = pd.Series(to_class(preds["y_true"], band), name="true")
    p = pd.Series(to_class(preds[model], band), name="predicted")
    return pd.crosstab(y, p).reindex(index=LABELS, columns=LABELS, fill_value=0)

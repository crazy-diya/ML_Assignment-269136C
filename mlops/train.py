"""Reproducible (re)training with a champion/challenger quality gate.

Uses the hyper-parameters found by the notebook's randomised search, so a full retrain takes
minutes instead of the hours a new search would need.

    python mlops/train.py                              # initial training on ikman_cars.csv
    python mlops/train.py --fresh data/fresh_batch.csv # retrain on old + fresh data, gate vs champion

Gate: with fresh data, 40 % of the fresh ads are held out. Neither the champion (current model.pkl)
nor the challenger saw them, so both are scored fairly. The challenger is promoted only if its
log-RMSE is no worse than the champion's.
"""
import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor  # noqa: F401  (import check for a clear error message)
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from carprep import CAT_COLS, NUM_COLS, RAW_COLS, CarFeatures, CatBoostSK, LogPriceModel, clean_rows  # noqa: E402

SEED = 42
MIN_EVAL_ROWS = 300   # the app's error range only follows evaluations on at least this many ads
# Hyper-parameters selected by 5-fold randomised search in Vehicle_Price_Prediction.ipynb
XGB_PARAMS = dict(colsample_bytree=0.6062, gamma=0.0545, learning_rate=0.0289, max_depth=6, min_child_weight=2,
                  n_estimators=321, reg_lambda=0.5107, subsample=0.6092)
LGBM_PARAMS = dict(colsample_bytree=0.6725, learning_rate=0.0718, max_depth=8, min_child_samples=6,
                   n_estimators=352, num_leaves=58, reg_lambda=1.9123, subsample=0.8212, subsample_freq=1)
CAT_PARAMS = dict(depth=5, iterations=887, l2_leaf_reg=2.0498, learning_rate=0.0693, random_strength=1.5934)


def tree_pipe(est):
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS)]).set_output(transform="pandas")
    return Pipeline([("feat", CarFeatures()), ("prep", prep), ("model", est)])


def build_stack():
    members = [
        ("xgboost", tree_pipe(XGBRegressor(random_state=SEED, n_jobs=1, tree_method="hist", **XGB_PARAMS))),
        ("lightgbm", tree_pipe(LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1, **LGBM_PARAMS))),
        ("catboost", Pipeline([("feat", CarFeatures()), ("model", CatBoostSK(random_seed=SEED, **CAT_PARAMS))])),
    ]
    return StackingRegressor(members, final_estimator=RidgeCV(alphas=np.logspace(-2, 2, 15)), cv=5, n_jobs=-1)


def score(model, X, y):
    """Metrics for a model that predicts rupees (LogPriceModel)."""
    pred = model.predict(X)
    return {"rmse_log": float(np.sqrt(mean_squared_error(np.log1p(y), np.log1p(pred)))),
            "r2": float(r2_score(y, pred)),
            "mape_pct": float(100 * mean_absolute_percentage_error(y, pred)),
            "mae_lkr": float(mean_absolute_error(y, pred)),
            "n": int(len(y))}


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def ui_metadata(df):
    return {
        "brands": sorted(df.brand.dropna().unique()),
        "models_by_brand": {b: sorted(g.model.dropna().unique()) for b, g in df.groupby("brand")},
        "conditions": sorted(df.condition.dropna().unique()),
        "transmissions": sorted(df.transmission.dropna().unique()),
        "bodies": sorted(df.body.dropna().unique()),
        "fuels": sorted(df.fuel_type.dropna().unique()),
        "locations": sorted(df.location.dropna().unique()),
        "year_range": [1950, 2026],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "ikman_cars.csv"))
    ap.add_argument("--fresh", default=None, help="CSV of fresh ads (unseen by the champion)")
    ap.add_argument("--out-dir", default=str(ROOT))
    ap.add_argument("--report", default=str(ROOT / "monitoring" / "retrain_report.json"))
    ap.add_argument("--tolerance", type=float, default=0.0, help="allowed relative log-RMSE worsening")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    base = clean_rows(pd.read_csv(args.data))
    champion = None
    champ_path = ROOT / "model.pkl"
    if args.fresh and champ_path.exists():
        with open(champ_path, "rb") as f:
            champion = pickle.load(f)

    if args.fresh:
        fresh = clean_rows(pd.read_csv(args.fresh))
        fresh = fresh[~fresh.slug.isin(base.slug)]
        f_train, f_hold = train_test_split(fresh, test_size=0.4, random_state=SEED)
        train_df = pd.concat([base, f_train], ignore_index=True)
        hold = f_hold
        print(f"base {len(base)} + fresh-train {len(f_train)} -> train {len(train_df)}; holdout {len(hold)} fresh ads")
    else:
        train_df, hold = train_test_split(base, test_size=0.2, random_state=SEED)
        print(f"initial training: train {len(train_df)}, holdout {len(hold)}")

    challenger = LogPriceModel(build_stack()).fit(train_df[RAW_COLS], train_df["price_lkr"])
    ch = score(challenger, hold[RAW_COLS], hold["price_lkr"])
    print("challenger:", ch)

    report = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_sha": git_sha(),
              "train_rows": int(len(train_df)), "challenger": ch, "champion": None, "promoted": True,
              "reason": "initial training"}
    if champion is not None:
        cp = score(champion, hold[RAW_COLS], hold["price_lkr"])
        report["champion"] = cp
        ok = ch["rmse_log"] <= cp["rmse_log"] * (1 + args.tolerance)
        report["promoted"] = bool(ok)
        report["reason"] = (f"challenger log-RMSE {ch['rmse_log']:.4f} vs champion {cp['rmse_log']:.4f}: "
                            + ("promoted" if ok else "rejected, champion kept"))
        print("champion:", cp)
    print("decision:", report["reason"])

    if report["promoted"]:
        # refit on everything we have so the deployed model uses all data (incl. the holdout)
        all_df = pd.concat([train_df, hold], ignore_index=True)
        final = LogPriceModel(build_stack()).fit(all_df[RAW_COLS], all_df["price_lkr"])
        with open(out / "model.pkl", "wb") as f:
            pickle.dump(final, f)
        version = datetime.now(timezone.utc).strftime("%Y.%m.%d")
        meta_path = out / "model_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.update(ui_metadata(all_df))
        meta["best_model"] = "Stacking (XGB + LGBM + CatBoost -> Ridge)"
        meta["version"] = version
        meta["trained_at_utc"] = report["time_utc"]
        meta["git_sha"] = report["git_sha"]
        meta["train_rows"] = int(len(all_df))
        scores = {"RMSE (log)": round(ch["rmse_log"], 4), "R²": round(ch["r2"], 4),
                  "MAPE %": round(ch["mape_pct"], 3), "MAE (LKR)": round(ch["mae_lkr"], 1)}
        if ch["n"] >= MIN_EVAL_ROWS or "test_metrics" not in meta:
            meta["test_metrics"], meta["test_metrics_n"] = scores, ch["n"]   # drives the app's +/- range
        meta["latest_eval"] = {**scores, "n": ch["n"], "on": "holdout of this retrain"}
        meta_path.write_text(json.dumps(meta, indent=1))
        if args.fresh:   # grow the versioned dataset with the new raw ads (cleaning stays reproducible)
            merged = pd.concat([pd.read_csv(args.data), pd.read_csv(args.fresh)], ignore_index=True)
            merged.drop_duplicates("slug").to_csv(out / "ikman_cars.csv", index=False)
        report["version"] = version
        print(f"saved model.pkl and model_meta.json (version {version}, {len(all_df)} rows)")

    rp = Path(args.report); rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=1))
    hist = rp.parent / "retrain_history.jsonl"
    with hist.open("a") as f:
        f.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()

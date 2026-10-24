"""Data-drift and performance monitoring for the deployed car-price model.

    python mlops/monitor.py --fresh data/fresh_batch.csv

Compares a batch of fresh ikman.lk ads (a stand-in for production traffic: Streamlit Cloud has no
persistent storage for prediction logs) against the training data:
  * PSI (population stability index) per feature   -> > 0.20 means significant drift
  * Kolmogorov-Smirnov test on numeric features    -> p < 0.01 flags a shifted distribution
  * live accuracy of model.pkl on the fresh ads    -> MAPE > 1.3x the model's test MAPE means degradation
Writes monitoring/report.json + report.md and appends monitoring/history.csv.
"""
import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from carprep import RAW_COLS, clean_rows  # noqa: E402

NUMERIC = ["price_lkr", "model_year", "engine_cc", "mileage_km"]
CATEGORICAL = ["brand", "condition", "fuel_type", "body", "transmission"]
PSI_ALERT, KS_P_ALERT, MAPE_RATIO_ALERT, MIN_ROWS = 0.20, 0.01, 1.30, 30
EPS = 1e-4


def psi_numeric(ref, new, bins=10):
    ref, new = ref.dropna(), new.dropna()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + EPS
    n = np.histogram(new, edges)[0] / len(new) + EPS
    return float(np.sum((n - r) * np.log(n / r)))


def psi_categorical(ref, new, min_share=0.02):
    # pool rare categories: with ~200 fresh ads, tiny brands would otherwise create spurious PSI
    share = ref.value_counts(normalize=True)
    keep = set(share[share >= min_share].index)
    ref = ref.where(ref.isin(keep), "Other")
    new = new.where(new.isin(keep), "Other")
    cats = set(ref.dropna().unique()) | set(new.dropna().unique())
    r = ref.value_counts(normalize=True).reindex(list(cats)).fillna(0) + EPS
    n = new.value_counts(normalize=True).reindex(list(cats)).fillna(0) + EPS
    return float(np.sum((n - r) * np.log(n / r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(ROOT / "ikman_cars.csv"))
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "monitoring"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ref = clean_rows(pd.read_csv(args.reference))
    fresh = clean_rows(pd.read_csv(args.fresh))
    fresh = fresh[~fresh.slug.isin(ref.slug)]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if len(fresh) < MIN_ROWS:
        rep = {"time_utc": now, "fresh_rows": int(len(fresh)), "drift_detected": False,
               "reason": f"only {len(fresh)} fresh ads (< {MIN_ROWS}); not enough data to judge"}
        (out / "report.json").write_text(json.dumps(rep, indent=1))
        (out / "report.md").write_text(f"## Model monitoring\n\n{rep['reason']}\n")
        _emit(False)
        print(rep["reason"]); return

    features = []
    for c in NUMERIC:
        ks = ks_2samp(ref[c].dropna(), fresh[c].dropna())
        features.append({"feature": c, "type": "numeric", "psi": round(psi_numeric(ref[c], fresh[c]), 4),
                         "ks_p": round(float(ks.pvalue), 5), "ref_median": float(ref[c].median()),
                         "fresh_median": float(fresh[c].median())})
    for c in CATEGORICAL:
        features.append({"feature": c, "type": "categorical", "psi": round(psi_categorical(ref[c], fresh[c]), 4),
                         "ks_p": None, "ref_median": None, "fresh_median": None})
    for f in features:
        f["drifted"] = bool(f["psi"] > PSI_ALERT or (f["ks_p"] is not None and f["ks_p"] < KS_P_ALERT and f["psi"] > 0.1))

    with open(ROOT / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    meta = json.loads((ROOT / "model_meta.json").read_text())
    pred = model.predict(fresh[RAW_COLS])
    actual = fresh["price_lkr"].to_numpy()
    mape = float(100 * np.mean(np.abs(pred / actual - 1)))
    rmse_log = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2)))
    base_mape = float(meta["test_metrics"]["MAPE %"])
    ratio = mape / base_mape
    perf_degraded = bool(ratio > MAPE_RATIO_ALERT)
    price_shift = float(fresh.price_lkr.median() / ref.price_lkr.median() - 1)

    drifted = [f["feature"] for f in features if f["drifted"]]
    drift = bool(drifted or perf_degraded)
    rep = {"time_utc": now, "model_version": meta.get("version", "n/a"), "fresh_rows": int(len(fresh)),
           "drifted_features": drifted, "performance": {"fresh_mape_pct": round(mape, 2), "baseline_mape_pct": base_mape,
           "ratio": round(ratio, 2), "fresh_rmse_log": round(rmse_log, 4), "degraded": perf_degraded},
           "median_price_shift_pct": round(100 * price_shift, 1), "drift_detected": drift, "features": features}
    (out / "report.json").write_text(json.dumps(rep, indent=1))

    md = ["## Model monitoring report", "",
          f"* Model version **{rep['model_version']}** · fresh ads checked: **{len(fresh)}** · time (UTC): {now}",
          f"* Live accuracy on unseen ads: **MAPE {mape:.1f}%** (test MAPE {base_mape:.1f}%, ratio {ratio:.2f}, "
          f"alert > {MAPE_RATIO_ALERT}) -> {'**DEGRADED**' if perf_degraded else 'OK'}",
          f"* Median price vs training data: {100 * price_shift:+.1f}%",
          f"* **Drift detected: {'YES' if drift else 'no'}**" + (f" ({', '.join(drifted)})" if drifted else ""), "",
          "| Feature | Type | PSI | KS p-value | Drift |", "|---|---|---|---|---|"]
    for f in features:
        md.append(f"| {f['feature']} | {f['type']} | {f['psi']:.3f} | {'' if f['ks_p'] is None else f['ks_p']} | "
                  f"{'YES' if f['drifted'] else 'no'} |")
    md += ["", f"_PSI > {PSI_ALERT} = significant shift. Fresh ads are a stand-in for production traffic._"]
    (out / "report.md").write_text("\n".join(md) + "\n")

    hist = out / "history.csv"
    row = pd.DataFrame([{"time_utc": now, "model_version": rep["model_version"], "fresh_rows": len(fresh),
                         "fresh_mape_pct": round(mape, 2), "mape_ratio": round(ratio, 2),
                         "drifted_features": ";".join(drifted), "drift_detected": drift}])
    row.to_csv(hist, mode="a", header=not hist.exists(), index=False)
    print("\n".join(md))
    _emit(drift)


def _emit(drift):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"drift_detected={'true' if drift else 'false'}\n")


if __name__ == "__main__":
    main()

## Model monitoring report

* Model version **2026.10.24** · fresh ads checked: **150** · time (UTC): 2026-11-02T03:16:31+00:00
* Live accuracy on unseen ads: **MAPE 15.9%** (test MAPE 12.6%, ratio 1.25, alert > 1.3) -> OK
* Median price vs training data: +19.4%
* **Drift detected: YES** (price_lkr, mileage_km, condition)

| Feature | Type | PSI | KS p-value | Drift |
|---|---|---|---|---|
| price_lkr | numeric | 0.134 | 0.00672 | YES |
| model_year | numeric | 0.049 | 0.11252 | no |
| engine_cc | numeric | 0.045 | 0.21269 | no |
| mileage_km | numeric | 0.171 | 0.0032 | YES |
| brand | categorical | 0.070 |  | no |
| condition | categorical | 0.237 |  | YES |
| fuel_type | categorical | 0.002 |  | no |
| body | categorical | 0.128 |  | no |
| transmission | categorical | 0.004 |  | no |

_PSI > 0.2 = significant shift. Fresh ads are a stand-in for production traffic._

## Model monitoring report

* Model version **2026.11.02** · fresh ads checked: **150** · time (UTC): 2026-11-09T03:15:59+00:00
* Live accuracy on unseen ads: **MAPE 12.3%** (test MAPE 12.6%, ratio 0.97, alert > 1.3) -> OK
* Median price vs training data: +1.3%
* **Drift detected: YES** (brand)

| Feature | Type | PSI | KS p-value | Drift |
|---|---|---|---|---|
| price_lkr | numeric | 0.114 | 0.12719 | no |
| model_year | numeric | 0.055 | 0.07605 | no |
| engine_cc | numeric | 0.135 | 0.37572 | no |
| mileage_km | numeric | 0.113 | 0.07224 | no |
| brand | categorical | 0.264 |  | YES |
| condition | categorical | 0.054 |  | no |
| fuel_type | categorical | 0.006 |  | no |
| body | categorical | 0.106 |  | no |
| transmission | categorical | 0.067 |  | no |

_PSI > 0.2 = significant shift. Fresh ads are a stand-in for production traffic._

## Model monitoring report

* Model version **2026.10.23** · fresh ads checked: **200** · time (UTC): 2026-10-24T20:30:47+00:00
* Live accuracy on unseen ads: **MAPE 9.4%** (test MAPE 12.6%, ratio 0.74, alert > 1.3) -> OK
* Median price vs training data: -4.4%
* **Drift detected: no**

| Feature | Type | PSI | KS p-value | Drift |
|---|---|---|---|---|
| price_lkr | numeric | 0.049 | 0.66681 | no |
| model_year | numeric | 0.098 | 0.0128 | no |
| engine_cc | numeric | 0.045 | 0.26362 | no |
| mileage_km | numeric | 0.078 | 0.03067 | no |
| brand | categorical | 0.052 |  | no |
| condition | categorical | 0.093 |  | no |
| fuel_type | categorical | 0.041 |  | no |
| body | categorical | 0.161 |  | no |
| transmission | categorical | 0.061 |  | no |

_PSI > 0.2 = significant shift. Fresh ads are a stand-in for production traffic._

## Model monitoring report

* Model version **2026.10.23** · fresh ads checked: **100** · time (UTC): 2026-10-24T20:54:42+00:00
* Live accuracy on unseen ads: **MAPE 8.8%** (test MAPE 12.6%, ratio 0.69, alert > 1.3) -> OK
* Median price vs training data: -18.5%
* **Drift detected: YES** (price_lkr, mileage_km, brand)

| Feature | Type | PSI | KS p-value | Drift |
|---|---|---|---|---|
| price_lkr | numeric | 0.255 | 0.00011 | YES |
| model_year | numeric | 0.038 | 0.22619 | no |
| engine_cc | numeric | 0.187 | 0.01265 | no |
| mileage_km | numeric | 0.238 | 0.08812 | YES |
| brand | categorical | 0.529 |  | YES |
| condition | categorical | 0.003 |  | no |
| fuel_type | categorical | 0.069 |  | no |
| body | categorical | 0.179 |  | no |
| transmission | categorical | 0.038 |  | no |

_PSI > 0.2 = significant shift. Fresh ads are a stand-in for production traffic._

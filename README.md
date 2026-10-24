# Sri Lankan Used-Car Price Prediction (Applied Machine Learning)

**Live demo (Azure Container Apps):** https://car-price-app.bluemeadow-a6e43a1f.malaysiawest.azurecontainerapps.io  
**Live demo (Streamlit Cloud):** https://crazy-diya-ml-assignment-269136c-app-8yzt7l.streamlit.app/

Predict the **asking price (LKR)** of a used/new car advertised in Sri Lanka from its specification — a tabular **regression** problem on ~3,000 live listings scraped from [ikman.lk](https://ikman.lk).

## Files
| File | Purpose |
|---|---|
| `Vehicle_Price_Prediction.ipynb` | **Main deliverable** — EDA, cleaning, tuned models, ensembles, SHAP, LIME, inference, export |
| `scrape_ikman_cars.py` | Polite scraper that built the dataset (respects `robots.txt`, 1.5 s delay, resumable) |
| `ikman_cars.csv` | Scraped dataset (3,000 ads) |
| `carprep.py` | Shared cleaning / imputation / feature engineering (used by notebook, pickle and app) |
| `model.pkl`, `model_meta.json` | Exported model (predicts LKR) and UI option lists |
| `app.py` | Streamlit front end |
| `model_comparison.csv` | Test-set metrics for every model |

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-notebook.txt   # full set for the notebook
jupyter lab                      # open Vehicle_Price_Prediction.ipynb (kernel: "ML Assignment (.venv)")
streamlit run app.py             # front end (needs only requirements.txt)
```

## Results (20 % hold-out test set)
| Model | R² | MAPE | RMSE (log price) |
|---|---|---|---|
| Stacking (XGB + LGBM + CatBoost → Ridge) | 0.914 | 12.6 % | 0.225 |
| CatBoost (best single model) | 0.905 | 13.1 % | 0.225 |
| Voting (XGB + LGBM + CatBoost) | 0.914 | 12.8 % | 0.231 |
| XGBoost / LightGBM / sklearn GBM | 0.89–0.91 | 14.0–14.7 % | 0.247–0.255 |
| Random Forest | 0.858 | 15.9 % | 0.273 |
| Ridge baseline / Decision-tree baseline | 0.761 / 0.735 | 16.7 % / 23.5 % | 0.272 / 0.335 |

Hyper-parameters were designed per algorithm and tuned with 5-fold randomised search (no defaults). Explanations: SHAP (global) and LIME (local).

## MLOps (bonus)
CI tests, weekly drift monitoring, gated retraining, Docker and Azure Container Apps deployment: see [MLOPS.md](MLOPS.md).

## Notes and limitations
* Prices are **asking prices** from one website at one point in time, not final sale prices.
* Data was collected for coursework; cite ikman.lk as the source.
* Rare luxury/vintage cars have few comparables and are predicted less reliably.

## Deploy the app (Streamlit Community Cloud)
Deployed at https://crazy-diya-ml-assignment-269136c-app-8yzt7l.streamlit.app/

1. Sign in at https://share.streamlit.io with GitHub and click **Create app**.
2. Repository `crazy-diya/ML_Assignment-269136C`, branch `main`, main file `app.py`.
3. Under **Advanced settings** choose Python 3.12 or 3.13, then **Deploy**.

`requirements.txt` contains only what the app needs; `requirements-notebook.txt` is the full list for the notebook.

# Slide outline — Sri Lankan Used-Car Price Prediction
Target: 10 slides, about 5 minutes for the video. Figures are in `report_figs/`.

| Slide | Title | On the slide | Say (speaker notes) | Figure |
|---|---|---|---|---|
| 1 | **Predicting used-car prices in Sri Lanka** | Your name, student ID, module | "I built a model that estimates the asking price of a car in Sri Lanka." | — |
| 2 | **The local problem** | Import restrictions since 2020 → prices high and opaque · reconditioned imports · hybrid/EV boom · buyers need a fair price | "This is a local problem: prices rose sharply after 2020 and are hard to judge. It's a regression problem on tabular data." | — |
| 3 | **Data: 3,000 ikman.lk ads** | Scraped politely (robots.txt, 1.5 s delay) · 10 features · asking price is the target · limitation: asking ≠ sale price | "I collected my own data from Sri Lanka's biggest classifieds site." | — |
| 4 | **What the data looked like** | Price very skewed → train on log(price) · condition and age drive price · 351 car models | "A few luxury cars skew the price, so I model log-price. Condition is a uniquely Sri Lankan feature." | `cell8_0.png` or `cell10_0.png` |
| 5 | **Cleaning real, messy data** | Invalid values found: price 0/106/123456, a 595M BMW with a 1 cc "electric" engine, year 2027, mileage 1/123456 · dropped 14 rows, imputed the rest · kept vintage and luxury cars | "Not all errors are nulls. I used domain knowledge to decide what to drop, impute or keep." | table from notebook cell 7 |
| 6 | **Models and tuning (no defaults)** | Baselines: Ridge, Decision Tree · RF, XGBoost, LightGBM, CatBoost, GBM · Voting and Stacking · 5-fold randomised search with designed ranges (depth, learning rate, subsampling, regularisation) | "I designed the parameter ranges from how each algorithm works and from the small data size, instead of using defaults." | — |
| 7 | **Results** | Stacking R² 0.914, MAPE 12.6% · CatBoost best single model · baselines R² 0.76 / 0.74 · RF weakest ensemble (0.858) | "Ensembles beat the baselines clearly. Stacking and CatBoost are almost tied; I don't oversell the stack." | `cell27_0.png` |
| 8 | **Strengths and limitations** | One line per model (see report §5) · RF can't extrapolate to luxury cars · CatBoost suits many categories · stacking is complex for a small gain | "Each model has trade-offs; the table shows why CatBoost won here." | — |
| 9 | **Explainability: SHAP and LIME** | SHAP: engine size, age, mileage matter most · LIME: Alto's small engine and manual gearbox lower its price · LIME on the Santa Fe exposed a suspicious mileage | "SHAP explains the whole model, LIME explains one car. LIME even revealed a likely data-entry error." | `cell32_0.png`, `cell35_0.png` |
| 10 | **Demo, limits, next steps** | Streamlit app (live demo) · held-out car: predicted 7.43M vs actual 7.48M (−0.6%) · limits: asking prices, one site, 3,000 ads · next: add time/ARIMA for market drift, Riyasewana data, MLOps deployment (bonus) | "Here's the app. Enter a car and it returns a price and a ±13% range." | screenshot of the app |

## Demo script (about 60 seconds)
1. Run `streamlit run app.py`.
2. Select Toyota → Aqua, 2016, Reconditioned, Hybrid, Hatchback, Automatic, 1,500 cc, 70,000 km, Colombo, then press **Predict**. The notebook gives about LKR 10.0M for this car.
3. Change the fuel to Electric or the brand to Honda → Vezel and show how the estimate changes.

## Before you record
- Fill in your name, ID and GitHub link in the report.
- Check each figure on its slide.
- Re-state the limitation that these are asking prices. Examiners like honest limits.

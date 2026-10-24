"""Streamlit front end for the Sri Lankan used-car price model.

Run:  streamlit run app.py
Needs model.pkl and model_meta.json (created by Vehicle_Price_Prediction.ipynb) next to this file.
"""
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

import carprep  # noqa: F401  (needed so pickle can find CarFeatures / LogPriceModel)

HERE = Path(__file__).parent

st.set_page_config(page_title="Sri Lanka Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load():
    with open(HERE / "model.pkl", "rb") as f:
        model = pickle.load(f)
    meta = json.load(open(HERE / "model_meta.json"))
    return model, meta


model, meta = load()

st.title("🚗 Sri Lanka Used-Car Price Predictor")
st.caption(f"Model: **{meta['best_model']}** trained on ~3,000 ikman.lk listings. "
           "Predicts the likely *asking price* in LKR.")

col1, col2 = st.columns(2)
brand = col1.selectbox("Brand", meta["brands"], index=meta["brands"].index("Toyota") if "Toyota" in meta["brands"] else 0)
models = meta["models_by_brand"].get(brand, [])
model_name = col2.selectbox("Model", models) if models else col2.text_input("Model")

year = col1.number_input("Year of manufacture", min_value=meta["year_range"][0], max_value=meta["year_range"][1], value=2016, step=1)
condition = col2.selectbox("Condition", meta["conditions"], index=meta["conditions"].index("Used") if "Used" in meta["conditions"] else 0)

def default(options, wanted):
    return options.index(wanted) if wanted in options else 0


fuel = col1.selectbox("Fuel type", meta["fuels"], index=default(meta["fuels"], "Petrol"))
transmission = col2.selectbox("Transmission", meta["transmissions"], index=default(meta["transmissions"], "Automatic"))

body = col1.selectbox("Body type", meta["bodies"], index=default(meta["bodies"], "Hatchback"))
location = col2.selectbox("Location", meta["locations"], index=default(meta["locations"], "Nugegoda"))

is_ev = fuel == "Electric"
engine = col1.number_input("Engine capacity (cc)" + (" — ignored for EVs" if is_ev else ""), min_value=0, max_value=8000,
                           value=1500, step=50, disabled=is_ev)
mileage = col2.number_input("Mileage (km)", min_value=0, max_value=1_000_000, value=60_000, step=1000)

if st.button("Predict price", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "brand": brand, "model": model_name, "model_year": year, "condition": condition,
        "transmission": transmission, "body": body, "fuel_type": fuel,
        "engine_cc": 0 if is_ev else engine, "mileage_km": mileage, "location": location,
    }])
    price = float(model.predict(row)[0])
    # approx. range from the model's typical relative error on unseen cars (MAPE)
    pct = meta["test_metrics"]["MAPE %"] / 100
    st.success(f"### Estimated price: LKR {price:,.0f}")
    st.write(f"Likely range: **LKR {price * (1 - pct):,.0f} – {price * (1 + pct):,.0f}** "
             f"(± {pct:.0%}, the model's average error on unseen cars)")
    if condition == "Used" and mileage < 500:
        st.warning("A 'Used' car with under 500 km is unusual — the model treats the mileage as missing.")

with st.expander("About this model"):
    m = meta["test_metrics"]
    st.write(f"Test R²: **{m['R²']:.3f}** · MAPE: **{m['MAPE %']:.1f}%** · MAE: **LKR {m['MAE (LKR)']:,.0f}**")
    st.write("Data: asking prices scraped from ikman.lk. Asking prices are usually higher than final sale prices, "
             "and rare luxury or vintage cars have few comparable listings, so treat the estimate as a guide.")

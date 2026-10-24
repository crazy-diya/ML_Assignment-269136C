"""CI tests: feature engineering, the shipped model, its metadata and the Streamlit app."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from carprep import CAT_COLS, NUM_COLS, RAW_COLS, CarFeatures, clean_rows  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(ROOT / "ikman_cars.csv")


@pytest.fixture(scope="module")
def model():
    with open(ROOT / "model.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def meta():
    return json.loads((ROOT / "model_meta.json").read_text())


def car(**kw):
    base = dict(brand="Toyota", model="Aqua", model_year=2016, condition="Reconditioned", transmission="Automatic",
                body="Hatchback", fuel_type="Hybrid", engine_cc=1500, mileage_km=70000, location="Nugegoda")
    base.update(kw)
    return pd.DataFrame([base])


# ---------------------------------------------------------------- data and features
def test_dataset_schema(data):
    assert {"slug", "price_lkr", *RAW_COLS} <= set(data.columns)
    assert len(data) >= 2500
    assert data.slug.is_unique


def test_clean_rows_drops_impossible_prices(data):
    cleaned = clean_rows(data)
    assert cleaned.price_lkr.between(200_000, 300_000_000).all()
    assert len(cleaned) > 0.95 * len(data)


def test_features_have_no_nulls_and_fix_invalid_values(data):
    feat = CarFeatures().fit(clean_rows(data))
    messy = pd.concat([
        car(fuel_type="Electric", engine_cc=60),                      # EV: kW, not cc
        car(engine_cc=1, mileage_km=123456),                          # impossible engine, placeholder km
        car(condition="Used", mileage_km=1),                          # "used" car with 1 km
        car(brand="NewBrand", model="Unseen", location="Nowhere"),    # unseen categories
        car(model_year=2099, mileage_km=np.nan, engine_cc=np.nan),    # future year, missing values
    ], ignore_index=True)
    out = feat.transform(messy)
    assert list(out.columns) == CAT_COLS + NUM_COLS
    assert out.isna().sum().sum() == 0
    assert out.loc[0, "engine_cc"] == 0
    assert out.loc[1, "engine_cc"] >= 300 and out.loc[1, "mileage_km"] != 123456
    assert out["age"].min() >= 0


# ---------------------------------------------------------------- model
def test_model_predicts_sane_prices(model):
    p = model.predict(car())[0]
    assert np.isfinite(p) and 3e5 < p < 5e8


def test_model_handles_unseen_categories(model):
    p = model.predict(car(brand="NewBrand", model="Unseen", location="Nowhere", body="Flying car"))[0]
    assert np.isfinite(p) and p > 0


def test_new_car_costs_more_than_old_one(model):
    new = model.predict(car(brand="Toyota", model="Corolla", model_year=2025, condition="Brand New", mileage_km=0,
                            fuel_type="Petrol", engine_cc=1500))[0]
    old = model.predict(car(brand="Toyota", model="Corolla", model_year=2005, condition="Used", mileage_km=180000,
                            fuel_type="Petrol", engine_cc=1500))[0]
    assert new > old


def test_batch_prediction_shape(model, data):
    sample = data.sample(50, random_state=0)
    pred = model.predict(sample[RAW_COLS])
    assert pred.shape == (50,) and np.isfinite(pred).all()


# ---------------------------------------------------------------- quality gate on shipped metadata
def test_metadata_and_quality_gate(meta):
    for key in ("brands", "models_by_brand", "conditions", "fuels", "locations", "test_metrics", "best_model"):
        assert key in meta
    m = meta["test_metrics"]
    assert m["R²"] >= 0.85, "model R² fell below the 0.85 release threshold"
    assert m["MAPE %"] <= 20, "model MAPE rose above the 20 % release threshold"


# ---------------------------------------------------------------- app
def test_app_runs_and_predicts():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Estimated price" in s.value for s in at.success)

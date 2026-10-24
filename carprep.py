"""Shared feature engineering for the Sri Lankan used-car price model.

Lives in its own module so the notebook, the pickled model and the Streamlit app
all use exactly the same cleaning/imputation logic (pickle stores a reference to
these classes, so this file must sit next to model.pkl).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

REF_YEAR = 2026
CAT_COLS = ["brand", "model_grp", "condition", "transmission", "body", "fuel_type", "location_grp"]
NUM_COLS = ["age", "engine_cc", "mileage_km", "km_per_year"]
RAW_COLS = ["brand", "model", "model_year", "condition", "transmission", "body",
            "fuel_type", "engine_cc", "mileage_km", "location"]

# Mileage values people type as placeholders on classified sites
PLACEHOLDER_KM = {1, 10, 100, 1111, 1234, 11111, 12345, 55555, 123456, 1000000}


class CarFeatures(TransformerMixin, BaseEstimator):
    """Cleans invalid values, imputes nulls and engineers features.

    fit() learns (a) which models/locations are frequent enough to keep as their own
    category and (b) domain-aware medians for imputation:
      engine_cc  -> median of the same car model
      mileage_km -> median of the same condition (Brand New / Used / ...)
    """

    def __init__(self, min_model_count=15, min_loc_count=30):
        self.min_model_count = min_model_count
        self.min_loc_count = min_loc_count

    # ---- value-level cleaning (works for one row or many) -------------------
    @staticmethod
    def _clean(df):
        df = df.copy()
        eng = pd.to_numeric(df["engine_cc"], errors="coerce")
        km = pd.to_numeric(df["mileage_km"], errors="coerce")
        is_ev = df["fuel_type"].eq("Electric")
        # EV "engine" is a kW rating on ikman, not cc -> neutral constant, fuel_type carries the info
        eng = eng.where(~is_ev, 0)
        eng = eng.where(is_ev | (eng >= 300))          # < 300 cc is impossible for a petrol/diesel/hybrid car
        km_bad = km.isin(PLACEHOLDER_KM) | (km > 600_000) | (km < 0)
        km = km.where(~km_bad | df["condition"].isin(["Brand New"]) & km.eq(0))
        used_like = df["condition"].eq("Used")
        km = km.where(~(used_like & (km < 500)))       # a "used" car with <500 km is a typo
        df["engine_cc"] = eng
        df["mileage_km"] = km
        df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce").clip(1950, REF_YEAR)
        return df

    def fit(self, X, y=None):
        df = self._clean(X[RAW_COLS])
        mc = df["model"].value_counts()
        self.keep_models_ = set(mc[mc >= self.min_model_count].index)
        lc = df["location"].value_counts()
        self.keep_locs_ = set(lc[lc >= self.min_loc_count].index)
        self.engine_by_model_ = df.groupby("model")["engine_cc"].median()
        self.engine_global_ = df["engine_cc"].median()
        self.km_by_cond_ = df.groupby("condition")["mileage_km"].median()
        self.km_global_ = df["mileage_km"].median()
        self.year_global_ = df["model_year"].median()
        return self

    def transform(self, X):
        df = self._clean(X[RAW_COLS])
        out = pd.DataFrame(index=df.index)
        out["brand"] = df["brand"].fillna("Unknown").astype(str)
        out["model_grp"] = np.where(df["model"].isin(self.keep_models_), df["model"], "Other")
        out["condition"] = df["condition"].fillna("Used").astype(str)
        out["transmission"] = df["transmission"].fillna("Automatic").astype(str)
        out["body"] = df["body"].fillna("Unknown").astype(str)
        out["fuel_type"] = df["fuel_type"].fillna("Petrol").astype(str)
        out["location_grp"] = np.where(df["location"].isin(self.keep_locs_), df["location"], "Other")

        year = df["model_year"].fillna(self.year_global_)
        out["age"] = REF_YEAR - year
        eng_fill = df["model"].map(self.engine_by_model_).fillna(self.engine_global_)
        out["engine_cc"] = df["engine_cc"].fillna(eng_fill)
        km_fill = df["condition"].map(self.km_by_cond_).fillna(self.km_global_)
        out["mileage_km"] = df["mileage_km"].fillna(km_fill)
        out["km_per_year"] = out["mileage_km"] / np.maximum(out["age"], 1)
        return out[CAT_COLS + NUM_COLS]

    def get_feature_names_out(self, input_features=None):
        return np.array(CAT_COLS + NUM_COLS)


class CatBoostSK(RegressorMixin, BaseEstimator):
    """Thin scikit-learn-clonable wrapper: CatBoost with native categorical handling."""

    def __init__(self, iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
                 random_strength=1.0, random_seed=42, thread_count=1):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg, random_strength=self.random_strength,
            random_seed=self.random_seed, thread_count=self.thread_count,
            cat_features=CAT_COLS, loss_function="RMSE", verbose=0)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class LogPriceModel(RegressorMixin, BaseEstimator):
    """Wraps a model trained on log1p(price) so predict() returns Rupees."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self.pipeline.predict(X))

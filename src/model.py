from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.features import FEATURE_COLUMNS


@dataclass
class ModelResult:
    model: ClassifierMixin
    model_name: str
    auc: float
    accuracy: float
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    feature_importance: pd.DataFrame


def _build_model(random_state: int = 42) -> tuple[ClassifierMixin, str]:
    try:
        from lightgbm import LGBMClassifier

        return (
            LGBMClassifier(
                n_estimators=220,
                learning_rate=0.035,
                max_depth=5,
                num_leaves=24,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                verbosity=-1,
            ),
            "LightGBM",
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier

        return (
            XGBClassifier(
                n_estimators=220,
                learning_rate=0.035,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=random_state,
            ),
            "XGBoost",
        )
    except ImportError:
        return (
            RandomForestClassifier(
                n_estimators=260,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
            "RandomForest (fallback)",
        )


def train_direction_model(featured: pd.DataFrame, validation_days: int = 90) -> ModelResult:
    data = featured.dropna(subset=FEATURE_COLUMNS + ["target"]).sort_values("date").copy()
    unique_dates = np.sort(data["date"].unique())
    if len(unique_dates) <= validation_days + 80:
        raise ValueError("모델 학습에 필요한 기간이 부족합니다.")
    split_date = pd.Timestamp(unique_dates[-validation_days])
    train = data[data["date"] < split_date]
    valid = data[data["date"] >= split_date]
    model, model_name = _build_model()
    model.fit(train[FEATURE_COLUMNS], train["target"].astype(int))
    probability = model.predict_proba(valid[FEATURE_COLUMNS])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    auc = float(roc_auc_score(valid["target"], probability)) if valid["target"].nunique() > 1 else 0.5
    accuracy = float(accuracy_score(valid["target"], prediction))
    importance = getattr(model, "feature_importances_", np.zeros(len(FEATURE_COLUMNS)))
    feature_importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importance}).sort_values(
        "importance", ascending=False
    )
    return ModelResult(
        model=model,
        model_name=model_name,
        auc=auc,
        accuracy=accuracy,
        train_end=pd.Timestamp(train["date"].max()),
        validation_start=split_date,
        feature_importance=feature_importance,
    )


def predict_upside(model: ClassifierMixin, rows: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(rows[FEATURE_COLUMNS])[:, 1]


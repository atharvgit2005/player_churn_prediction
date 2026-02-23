import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def _find_column_case_insensitive(df: pd.DataFrame, target_name: str) -> Optional[str]:
    lookup = {col.lower(): col for col in df.columns}
    return lookup.get(target_name.lower())


def _find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        matched = _find_column_case_insensitive(df, name)
        if matched:
            return matched
    return None


def _default_positive_classes(target_col: str, options: List[str]) -> List[str]:
    lowered = [opt.lower() for opt in options]
    if "engagement" in target_col.lower() and "low" in lowered:
        return [options[lowered.index("low")]]

    tokens = ["churn", "yes", "true", "1", "left", "at_risk", "low"]
    matches = [opt for opt in options if any(token in opt.lower() for token in tokens)]
    if matches:
        return [matches[0]]

    return [options[0]] if options else []


def _risk_bucket(probability: float) -> str:
    if probability <= 0.30:
        return "Low"
    if probability <= 0.70:
        return "Medium"
    return "High"


def _risk_color(risk_level: str) -> str:
    return {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}.get(risk_level, "#60a5fa")


def load_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    positive_classes: Optional[List[str]] = None,
) -> Dict[str, object]:
    data = df.copy()

    freq_col = _find_first_existing_column(
        data,
        [
            "session_frequency",
            "sessions_per_week",
            "sessionsperweek",
            "SessionsPerWeek",
        ],
    )
    duration_col = _find_first_existing_column(
        data,
        [
            "avg_session_duration",
            "avg_session_duration_minutes",
            "avgsessiondurationminutes",
            "AvgSessionDurationMinutes",
        ],
    )

    if freq_col and duration_col:
        data["engagement_score"] = pd.to_numeric(data[freq_col], errors="coerce") * pd.to_numeric(
            data[duration_col], errors="coerce"
        )

    data = data.dropna(subset=[target_col]).copy()

    target_series = data[target_col]
    unique_count = target_series.nunique(dropna=True)

    if unique_count == 2:
        y_str = target_series.astype(str).str.strip().str.lower()
        counts = y_str.value_counts()
        positive_candidates = [
            value
            for value in y_str.unique()
            if any(token in value for token in ["churn", "yes", "true", "1", "left"])
        ]
        positive_label = positive_candidates[0] if positive_candidates else counts.idxmin()
        y = (y_str == positive_label).astype(int)
        class_labels = {0: "non_churn", 1: f"churn ({positive_label})"}
    else:
        if not positive_classes:
            raise ValueError("Select at least one positive class value for churn.")

        positive_set = {value.strip().lower() for value in positive_classes}
        y = target_series.astype(str).str.strip().str.lower().isin(positive_set).astype(int)

        if y.nunique() != 2:
            raise ValueError("Positive class selection produced a single class. Choose different class values.")

        class_labels = {0: "non_churn", 1: f"churn ({', '.join(positive_classes)})"}

    x_display = data.drop(columns=[target_col]).copy()

    id_col = _find_first_existing_column(
        x_display,
        ["PlayerID", "player_id", "playerid", "user_id", "userid", "id"],
    )

    x_model = x_display.copy()
    id_dropped = False
    if id_col and id_col in x_model.columns:
        unique_ratio = x_model[id_col].nunique(dropna=False) / max(len(x_model), 1)
        if unique_ratio >= 0.80:
            x_model = x_model.drop(columns=[id_col])
            id_dropped = True

    numeric_features = x_model.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = x_model.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    class_distribution = y.value_counts(normalize=True)
    is_imbalanced = class_distribution.min() < 0.40

    categorical_modes = {}
    for feature in categorical_features:
        mode = x_model[feature].mode(dropna=True)
        categorical_modes[feature] = mode.iloc[0] if not mode.empty else "Unknown"

    return {
        "X_model": x_model,
        "X_display": x_display,
        "y": y,
        "preprocessor": preprocessor,
        "class_labels": class_labels,
        "is_imbalanced": is_imbalanced,
        "id_col": id_col,
        "id_dropped": id_dropped,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_modes": categorical_modes,
    }

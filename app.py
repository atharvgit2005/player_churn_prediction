import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
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


def _reset_training_state() -> None:
    keys = [
        "trained_models",
        "model_metrics",
        "confusion_matrices",
        "data_bundle",
        "target_configured",
    ]
    for key in keys:
        st.session_state.pop(key, None)


def _initialize_target_state(df: pd.DataFrame) -> None:
    target_options = df.columns.tolist()
    detected_target = _find_column_case_insensitive(df, "Churn")
    fallback_target = _find_first_existing_column(df, ["EngagementLevel", "RetentionStatus"])

    if "target_col" not in st.session_state or st.session_state["target_col"] not in target_options:
        if detected_target:
            st.session_state["target_col"] = detected_target
        elif fallback_target and fallback_target in target_options:
            st.session_state["target_col"] = fallback_target
        else:
            st.session_state["target_col"] = target_options[0]

    selected_target = st.session_state["target_col"]
    non_null = df[selected_target].dropna()

    if non_null.nunique() == 2:
        st.session_state["positive_classes"] = []
        return

    values = sorted(non_null.astype(str).str.strip().unique().tolist())
    default_positive = _default_positive_classes(selected_target, values)

    current = st.session_state.get("positive_classes", default_positive)
    if not set(current).issubset(set(values)):
        st.session_state["positive_classes"] = default_positive
    elif "positive_classes" not in st.session_state:
        st.session_state["positive_classes"] = default_positive

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    is_imbalanced: bool = False,
) -> Dict[str, Pipeline]:
    class_weight = "balanced" if is_imbalanced else None

    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight=class_weight,
            min_samples_leaf=5,
        ),
    }

    trained = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline

    return trained


def _plot_confusion_matrix(cm: np.ndarray, class_labels: Dict[int, str], title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    figure.colorbar(image, ax=axis)

    labels = [class_labels.get(0, "Class 0"), class_labels.get(1, "Class 1")]
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axis.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    figure.tight_layout()
    return figure


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: Dict[int, str],
) -> Dict[str, object]:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        y_proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
    else:
        y_proba = y_pred.astype(float)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else np.nan,
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_figure = _plot_confusion_matrix(cm, class_labels, f"{model_name} Confusion Matrix")

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "confusion_matrix_figure": cm_figure,
        "predictions": y_pred,
        "probabilities": y_proba,
    }


def _rf_feature_importance_figure(model: Pipeline, top_n: int = 12) -> Tuple[Optional[plt.Figure], pd.DataFrame]:
    estimator = model.named_steps.get("model")
    if not isinstance(estimator, RandomForestClassifier) or not hasattr(estimator, "feature_importances_"):
        return None, pd.DataFrame(columns=["feature", "importance"])

    preprocessor = model.named_steps.get("preprocessor")
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    else:
        feature_names = [f"feature_{i}" for i in range(len(estimator.feature_importances_))]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    top_df = importance_df.head(top_n).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(top_df["feature"], top_df["importance"], color="#2563eb")
    axis.set_title("Top Feature Importances (Random Forest)")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    figure.tight_layout()

    return figure, importance_df


def _top_driver_notes(importance_df: pd.DataFrame, top_n: int = 3) -> List[str]:
    if importance_df.empty:
        return ["Feature importance is unavailable for the selected model."]

    total = float(importance_df["importance"].sum()) or 1.0
    notes = []
    for _, row in importance_df.head(top_n).iterrows():
        share = (float(row["importance"]) / total) * 100
        notes.append(f"{row['feature']} contributes approximately {share:.1f}% of model importance.")
    return notes


def _build_probability_frame(
    model: Pipeline,
    X_model: pd.DataFrame,
    X_display: pd.DataFrame,
    y_true: pd.Series,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    if hasattr(model, "predict_proba"):
        churn_probability = model.predict_proba(X_model)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_model)
        churn_probability = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
    else:
        churn_probability = model.predict(X_model).astype(float)

    probability_frame = pd.DataFrame(
        {
            "row_id": np.arange(len(X_model)),
            "churn_probability": churn_probability,
            "predicted_label": (churn_probability >= 0.50).astype(int),
            "actual_label": y_true.values,
        }
    )
    probability_frame["risk_level"] = probability_frame["churn_probability"].apply(_risk_bucket)

    if id_col and id_col in X_display.columns:
        probability_frame[id_col] = X_display[id_col].values

    return probability_frame


def _risk_distribution_plots(probability_frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(probability_frame["churn_probability"], bins=20, color="#0ea5e9", alpha=0.9)
    axes[0].set_title("Churn Probability Distribution")
    axes[0].set_xlabel("Predicted churn probability")
    axes[0].set_ylabel("Players")
    axes[0].axvline(0.30, color="#22c55e", linestyle="--", linewidth=1)
    axes[0].axvline(0.70, color="#ef4444", linestyle="--", linewidth=1)

    risk_counts = probability_frame["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    axes[1].bar(risk_counts.index, risk_counts.values, color=["#22c55e", "#f59e0b", "#ef4444"])
    axes[1].set_title("Risk Bucket Distribution")
    axes[1].set_xlabel("Risk level")
    axes[1].set_ylabel("Players")

    figure.tight_layout()
    return figure


def predict_single_player(
    model: Pipeline,
    player_input: Dict[str, object],
    numeric_features: List[str],
    categorical_features: List[str],
    categorical_modes: Dict[str, object],
) -> Dict[str, object]:
    row = {}
    for feature in numeric_features:
        value = player_input.get(feature, np.nan)
        row[feature] = pd.to_numeric(value, errors="coerce")

    for feature in categorical_features:
        value = player_input.get(feature, categorical_modes.get(feature, "Unknown"))
        row[feature] = value if pd.notna(value) and str(value).strip() else categorical_modes.get(feature, "Unknown")

    player_frame = pd.DataFrame([row])

    if hasattr(model, "predict_proba"):
        churn_probability = float(model.predict_proba(player_frame)[:, 1][0])
    elif hasattr(model, "decision_function"):
        decision_value = float(model.decision_function(player_frame)[0])
        churn_probability = 1.0 / (1.0 + np.exp(-decision_value))
    else:
        churn_probability = float(model.predict(player_frame)[0])

    risk_level = _risk_bucket(churn_probability)

    return {
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "risk_color": _risk_color(risk_level),
        "prediction_label": int(churn_probability >= 0.5),
    }


def _render_risk_badge(risk_level: str, probability: float) -> None:
    color = _risk_color(risk_level)
    st.markdown(
        f"""
        <div style="padding:0.65rem 0.9rem;border-radius:0.6rem;background:{color}22;border:1px solid {color};display:inline-block;">
            <span style="font-weight:700;color:{color};">{risk_level} Risk</span>
            <span style="margin-left:0.55rem;color:#111827;">{probability:.1%} churn probability</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Player Churn Dashboard",
        page_icon="🎮",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to",
        [
            "Upload Data",
            "Model Training",
            "Model Evaluation",
            "Player Risk Analysis",
            "Decision Tree Explorer",
        ],
    )
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    df = None
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state["raw_df"] = df
        except Exception as exc:
            st.error(f"Failed to read uploaded file: {exc}")
            return
    elif "raw_df" in st.session_state:
        df = st.session_state["raw_df"]

    if df is not None and not df.empty:
        _initialize_target_state(df)

    if section == "Upload Data":
        st.subheader("Data Overview")
        if df is None:
            st.info("Upload a CSV file from the sidebar to begin.")
            return

        if df.empty:
            st.error("The uploaded CSV is empty.")
            return

        st.write("Raw Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        shape_col_1, shape_col_2 = st.columns(2)
        shape_col_1.metric("Rows", f"{df.shape[0]:,}")
        shape_col_2.metric("Columns", f"{df.shape[1]:,}")

        target_col = _find_column_case_insensitive(df, "Churn")
        if target_col is None:
            target_col = df.columns[-1]

        class_dist = df[target_col].astype(str).value_counts(dropna=False).rename_axis("Class").reset_index(name="Count")
        st.write(f"Class Distribution (`{target_col}`)")
        st.dataframe(class_dist, use_container_width=True)
    elif section == "Model Training":
        st.subheader("Model Training")
        if df is None:
            st.info("Upload data first in the sidebar.")
            return

        test_size = st.slider(
            "Train/Test split (test proportion)",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
        )
        depth_slider = st.slider(
            "Decision Tree max depth (0 = no limit)",
            min_value=0,
            max_value=20,
            value=6,
        )
        dt_max_depth = None if depth_slider == 0 else depth_slider

        if st.button("Train Model", type="primary", use_container_width=True):
            try:
                bundle = preprocess_data(
                    df,
                    st.session_state["target_col"],
                    positive_classes=st.session_state.get("positive_classes", []),
                )

                x_model = bundle["X_model"]
                y = bundle["y"]
                preprocessor = bundle["preprocessor"]
                class_weight = "balanced" if bundle["is_imbalanced"] else None

                stratify_y = y if y.value_counts().min() >= 2 else None
                x_train, x_test, y_train, y_test = train_test_split(
                    x_model,
                    y,
                    test_size=test_size,
                    random_state=RANDOM_STATE,
                    stratify=stratify_y,
                )

                decision_tree = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        (
                            "model",
                            DecisionTreeClassifier(
                                random_state=RANDOM_STATE,
                                class_weight=class_weight,
                                max_depth=dt_max_depth,
                                min_samples_leaf=2,
                            ),
                        ),
                    ]
                )
                decision_tree.fit(x_train, y_train)

                y_pred = decision_tree.predict(x_test)
                y_prob = decision_tree.predict_proba(x_test)[:, 1]
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1": f1_score(y_test, y_pred, zero_division=0),
                    "ROC-AUC": roc_auc_score(y_test, y_prob),
                }

                st.session_state["trained_models"] = {"Decision Tree": decision_tree}
                st.session_state["model_metrics"] = {"Decision Tree": metrics}
                st.session_state["confusion_matrices"] = {
                    "Decision Tree": confusion_matrix(y_test, y_pred, labels=[0, 1])
                }
                st.session_state["data_bundle"] = bundle

                st.success("Decision Tree training completed.")
                metric_cols = st.columns(5)
                metric_cols[0].metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                metric_cols[1].metric("Precision", f"{metrics['Precision']:.4f}")
                metric_cols[2].metric("Recall", f"{metrics['Recall']:.4f}")
                metric_cols[3].metric("F1", f"{metrics['F1']:.4f}")
                metric_cols[4].metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    elif section == "Model Evaluation":
        st.subheader("Model Evaluation")
        if "trained_models" not in st.session_state:
            st.info("Train models first from the Model Training section.")
            return

        metrics_map = st.session_state["model_metrics"]
        models = st.session_state["trained_models"]
        cm_map = st.session_state["confusion_matrices"]
        bundle = st.session_state["data_bundle"]

        comparison_df = pd.DataFrame(metrics_map).T
        st.write("Model Comparison")
        st.dataframe(comparison_df.round(4), use_container_width=True)

        selected_model = st.selectbox("Detailed evaluation model", options=list(models.keys()))
        selected_metrics = metrics_map[selected_model]

        metric_cols = st.columns(5)
        metric_cols[0].metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
        metric_cols[1].metric("Precision", f"{selected_metrics['Precision']:.4f}")
        metric_cols[2].metric("Recall", f"{selected_metrics['Recall']:.4f}")
        metric_cols[3].metric("F1", f"{selected_metrics['F1']:.4f}")
        metric_cols[4].metric("ROC-AUC", f"{selected_metrics['ROC-AUC']:.4f}")

        eval_left, eval_right = st.columns(2)
        with eval_left:
            cm = cm_map[selected_model]
            cm_fig, cm_ax = plt.subplots(figsize=(4.5, 4))
            im = cm_ax.imshow(cm, cmap="Blues")
            cm_fig.colorbar(im, ax=cm_ax)
            cm_ax.set_title(f"{selected_model} Confusion Matrix")
            cm_ax.set_xlabel("Predicted")
            cm_ax.set_ylabel("Actual")
            cm_ax.set_xticks([0, 1])
            cm_ax.set_yticks([0, 1])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    cm_ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
            plt.tight_layout()
            st.pyplot(cm_fig)

        probabilities = models[selected_model].predict_proba(bundle["X_model"])[:, 1]
        probability_frame = bundle["X_display"].copy()
        probability_frame["churn_probability"] = probabilities
        probability_frame["risk_level"] = probability_frame["churn_probability"].apply(_risk_bucket)

        with eval_right:
            hist_fig, hist_ax = plt.subplots(figsize=(6.5, 3.8))
            hist_ax.hist(probability_frame["churn_probability"], bins=20, color="#0ea5e9", edgecolor="white")
            hist_ax.set_title("Churn Probability Distribution")
            hist_ax.set_xlabel("Probability")
            hist_ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(hist_fig)

            pie_fig, pie_ax = plt.subplots(figsize=(4.5, 4.5))
            risk_counts = probability_frame["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
            pie_ax.pie(
                risk_counts.values,
                labels=risk_counts.index,
                autopct="%1.1f%%",
                colors=["#22c55e", "#f59e0b", "#ef4444"],
                startangle=90,
            )
            pie_ax.set_title("Risk Category Split")
            pie_ax.axis("equal")
            st.pyplot(pie_fig)

        st.write("Interactive Risk Filter")
        prob_range = st.slider(
            "Churn probability range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
        )

        categorical_columns = bundle["X_display"].select_dtypes(exclude=[np.number]).columns.tolist()
        filter_col = st.selectbox("Categorical filter feature", ["None"] + categorical_columns)

        filter_value = None
        if filter_col != "None":
            values = sorted(probability_frame[filter_col].dropna().astype(str).unique().tolist())
            filter_value = st.selectbox("Category value", values)

        filtered = probability_frame[probability_frame["churn_probability"].between(prob_range[0], prob_range[1])]
        if filter_col != "None" and filter_value is not None:
            filtered = filtered[filtered[filter_col].astype(str) == filter_value]

        st.write(f"Filtered players: **{len(filtered):,}**")
        st.dataframe(filtered.sort_values("churn_probability", ascending=False).head(500), use_container_width=True)

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Risk Predictions CSV",
            data=csv_data,
            file_name="player_churn_risk_predictions.csv",
            mime="text/csv",
        )

        st.write("Feature Interpretability")
        tree_model = models[selected_model].named_steps["model"]
        preprocessor = models[selected_model].named_steps["preprocessor"]
        importances = tree_model.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(10)
        )
        st.dataframe(importance_df, use_container_width=True)
    elif section == "Player Risk Analysis":
        st.info("Player Risk Analysis section coming next.")
    elif section == "Decision Tree Explorer":
        st.info("Decision Tree Explorer section coming next.")

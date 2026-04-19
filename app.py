import warnings
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from engagement_assistant import (
    build_report_payload,
    generate_engagement_optimization_report,
    generate_report_pdf_bytes,
    validate_report_payload,
    validation_error_message,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
SAMPLE_DATASET_PATH = Path(__file__).with_name("online_gaming_behavior_dataset.csv")
ACCENT_COLOR = "#FF4B4B"
BG_COLOR = "#080808"
SURFACE_COLOR = "#131313"
BORDER_COLOR = "#2A2A2A"
TEXT_PRIMARY = "#E5E2E1"
TEXT_MUTED = "#7E7A7A"


def _inject_global_styles() -> None:
    st.markdown(
        f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{
            background: transparent !important;
        }}
        .stApp {{
            background: {BG_COLOR};
            color: {TEXT_PRIMARY};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at top right, rgba(255,49,49,0.08), transparent 22%),
                radial-gradient(circle at 2px 2px, rgba(255,255,255,0.02) 1px, transparent 0),
                {BG_COLOR};
            background-size: auto, 38px 38px, auto;
        }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 1560px;
            padding-top: 0.9rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        [data-testid="stMainBlockContainer"] > [data-testid="stHorizontalBlock"]:first-of-type {{
            align-items: flex-start;
        }}
        [data-testid="stMainBlockContainer"] > [data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(1) {{
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            width: 260px;
            overflow-y: auto;
            background: #080808;
            z-index: 100;
            padding: 1.5rem 1rem;
            border-right: 1px solid #1E2530;
        }}
        [data-testid="stMainBlockContainer"] > [data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(2) {{
            margin-left: 260px;
            padding: 1.5rem;
            width: calc(100% - 260px);
            max-width: calc(100% - 260px);
        }}
        [data-testid="stSidebar"] {{
            display: none;
        }}
        .shell-nav {{
            position: sticky;
            top: 0.8rem;
        }}
        .nav-wordmark {{
            font-size: 1.4rem;
            font-weight: 900;
            color: {ACCENT_COLOR};
            letter-spacing: -0.02em;
        }}
        .nav-subtitle {{
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.24em;
            font-size: 0.6rem;
            font-weight: 800;
            margin-top: 0.15rem;
        }}
        .panel-label {{
            color: {TEXT_MUTED};
            font-size: 0.64rem;
            letter-spacing: 0.24em;
            font-weight: 800;
            text-transform: uppercase;
            margin: 0.2rem 0 0.65rem 0;
        }}
        .redline {{
            height: 2px;
            background: linear-gradient(90deg, {ACCENT_COLOR} 0%, rgba(255,49,49,0.08) 88%);
            margin: 1rem 0 1.1rem 0;
        }}
        .topbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(18,18,18,0.86);
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
            position: sticky;
            top: 0;
            z-index: 99;
            backdrop-filter: blur(12px);
        }}
        .topbar-search {{
            background: #080808;
            padding: 0.55rem 0.8rem;
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            font-weight: 600;
            min-width: 18rem;
        }}
        [data-testid="stRadio"] label {{
            background: transparent;
            border: none;
            border-left: 2px solid transparent;
            border-radius: 0;
            padding: 0.85rem 0.75rem;
            margin-bottom: 0.15rem;
            transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
        }}
        [data-testid="stRadio"] label:hover {{
            background: #1c1b1b;
            border-color: rgba(255,49,49,0.32);
        }}
        [data-testid="stRadio"] label[data-baseweb="radio"] {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: {TEXT_PRIMARY};
            transform: none;
        }}
        [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{
            display: none;
        }}
        [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {{
            display: none !important;
        }}
        [data-testid="stRadio"] label {{
            width: 100% !important;
            padding: 0.7rem 0.75rem !important;
            border-left: 2px solid transparent !important;
            border-radius: 0 !important;
            transition: all 0.15s ease !important;
        }}
        [data-testid="stRadio"] label:has(input:checked) {{
            border-left-color: #FF4B4B !important;
            background: rgba(255,75,75,0.08) !important;
        }}
        [data-testid="stRadio"] label[data-baseweb="radio"] p {{
            color: {TEXT_PRIMARY} !important;
            font-size: 0.88rem;
            font-weight: 600;
            margin: 0;
            letter-spacing: 0.01em;
            text-transform: none;
        }}
        [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
            background: #FF4B4B !important;
            box-shadow: 0 0 8px rgba(255,75,75,0.5) !important;
        }}
        [data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] {{
            background: #1E2530 !important;
        }}
        [data-testid="stSelectbox"] select,
        [data-baseweb="select"] > div {{
            background: #111519 !important;
            border: 1px solid #2A2A2A !important;
            border-radius: 0 !important;
            color: #F0F4F8 !important;
        }}
        [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div,
        [data-baseweb="select"]:focus-within > div {{
            border-color: #FF4B4B !important;
            box-shadow: none !important;
        }}
        [data-testid="stProgressBar"] > div > div {{
            background: #FF4B4B !important;
        }}
        .card {{
            background: linear-gradient(180deg, rgba(26,26,26,0.95), rgba(18,18,18,0.95));
            border: none;
            border-left: 2px solid rgba(255,49,49,0.24);
            border-radius: 0;
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        .card-title {{
            color: {TEXT_PRIMARY};
            font-size: 0.74rem;
            font-weight: 900;
            margin: 0 0 0.8rem 0;
            text-transform: uppercase;
            letter-spacing: 0.18em;
        }}
        .page-title {{
            color: {TEXT_PRIMARY};
            font-size: 2.8rem;
            font-weight: 900;
            letter-spacing: -0.04em;
            margin: 0;
            text-transform: uppercase;
            font-style: italic;
        }}
        .page-subtitle {{
            color: {TEXT_MUTED};
            font-size: 0.84rem;
            margin-top: 0.55rem;
            margin-bottom: 1.1rem;
            max-width: 68ch;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 700;
        }}
        .section-note {{
            color: {TEXT_MUTED};
            font-size: 0.84rem;
            margin-bottom: 0.6rem;
        }}
        .info-banner {{
            border-left: 2px solid {ACCENT_COLOR};
            background: #121212;
            border-radius: 0;
            padding: 0.9rem 1rem;
            color: {TEXT_PRIMARY};
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            font-weight: 700;
        }}
        .metric-panel {{
            background: {SURFACE_COLOR};
            border: none;
            border-left: 2px solid {ACCENT_COLOR};
            border-radius: 0;
            padding: 0.9rem 1rem;
        }}
        .metric-panel .metric-label {{
            color: {TEXT_MUTED};
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-weight: 900;
        }}
        .metric-panel .metric-value {{
            color: {TEXT_PRIMARY};
            font-size: 2rem;
            font-weight: 900;
            margin-top: 0.45rem;
            letter-spacing: -0.04em;
        }}
        .metric-panel .metric-note {{
            color: {TEXT_MUTED};
            font-size: 0.74rem;
            margin-top: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .step-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.45rem;
            margin: 0.35rem 0 0.9rem 0;
        }}
        .step-chip {{
            border: none;
            border-left: 2px solid rgba(255,255,255,0.06);
            border-radius: 0;
            padding: 0.75rem 0.85rem;
            background: #121212;
        }}
        .step-chip.active {{
            border-color: rgba(255,49,49,0.75);
            background: rgba(255,49,49,0.08);
        }}
        .step-chip.done {{
            border-color: rgba(255,49,49,0.28);
            background: #121212;
        }}
        .step-chip .step-title {{
            color: {TEXT_PRIMARY};
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }}
        .step-chip .step-copy {{
            color: {TEXT_MUTED};
            font-size: 0.7rem;
            margin-top: 0.34rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .chat-shell {{
            background: linear-gradient(180deg, rgba(26,26,26,0.95), rgba(18,18,18,0.98));
            border: none;
            border-left: 2px solid rgba(255,49,49,0.35);
            border-radius: 0;
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        .chat-heading {{
            color: {TEXT_PRIMARY};
            font-size: 1.05rem;
            font-weight: 750;
        }}
        .chat-copy {{
            color: {TEXT_MUTED};
            font-size: 0.9rem;
            margin-top: 0.22rem;
        }}
        .report-divider {{
            height: 1px;
            background: {BORDER_COLOR};
            margin: 1rem 0;
        }}
        .sample-card {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }}
        .sample-card .sample-icon {{
            font-size: 1.8rem;
        }}
        div.stButton > button,
        div.stDownloadButton > button {{
            background: #FF4B4B;
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            padding: 0.65rem 1rem;
            text-transform: none;
            letter-spacing: normal;
            box-shadow: none;
        }}
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {{
            background: #ff625c;
        }}
        div.stButton > button[kind="secondary"],
        button[kind="secondary"] {{
            background: #1c1b1b;
            color: {TEXT_PRIMARY};
            border: none;
            box-shadow: none;
        }}
        [data-testid="stFileUploader"] section {{
            background: #121212;
            border: none;
            border-left: 2px solid rgba(255,49,49,0.25);
            border-radius: 0;
            padding: 0.65rem;
        }}
        [data-testid="stDataFrame"] {{
            border: none;
            border-left: 2px solid rgba(255,49,49,0.25);
            border-radius: 0;
            overflow: hidden;
        }}
        [data-testid="stDataFrame"] thead tr th {{
            background: #121212 !important;
            color: {TEXT_PRIMARY} !important;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.68rem !important;
        }}
        [data-testid="stDataFrame"] tbody tr:nth-child(even) td {{
            background: rgba(255,255,255,0.02) !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: linear-gradient(180deg, rgba(19,19,19,0.98), rgba(14,14,14,0.98));
            border: none !important;
            border-left: 2px solid rgba(255,49,49,0.22);
            border-radius: 0 !important;
            padding: 0.2rem 0.25rem;
        }}
        [data-testid="stMetric"] {{
            background: {SURFACE_COLOR};
            border: none;
            border-left: 2px solid {ACCENT_COLOR};
            padding: 0.9rem 0.95rem;
            border-radius: 0;
        }}
        [data-testid="stExpander"] {{
            border: none;
            border-left: 2px solid rgba(255,49,49,0.18);
            border-radius: 0;
            background: #121212;
        }}
        .legend-note {{
            color: {TEXT_MUTED};
            font-size: 0.86rem;
            margin-top: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(content_fn, title: Optional[str] = None) -> None:
    with st.container(border=False):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if title:
            st.markdown(f"<p class='card-title'>{title}</p>", unsafe_allow_html=True)
        content_fn()
        st.markdown("</div>", unsafe_allow_html=True)


def _page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def _sidebar_label(text: str) -> None:
    st.sidebar.markdown(f"<div class='sidebar-label'>{text}</div>", unsafe_allow_html=True)


def _info_banner(text: str, icon: str = "") -> None:
    prefix = f"{icon} " if icon else ""
    st.markdown(f"<div class='info-banner'>{prefix}{text}</div>", unsafe_allow_html=True)


def _metric_panel(label: str, value: str, note: Optional[str] = None, accent: str = ACCENT_COLOR) -> None:
    st.markdown(
        f"""
        <div class="metric-panel" style="border-left-color:{accent};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _step_indicator(active_step: int) -> None:
    steps = [
        ("Ingest", "Load and validate player data."),
        ("Prepare", "Build target and feature pipeline."),
        ("Train", "Fit the selected churn model."),
        ("Evaluate", "Review metrics and risk outputs."),
    ]
    html = ["<div class='step-row'>"]
    for index, (title, copy) in enumerate(steps, start=1):
        state_class = "done" if index < active_step else "active" if index == active_step else ""
        html.append(
            f"<div class='step-chip {state_class}'><div class='step-title'>{index}. {title}</div>"
            f"<div class='step-copy'>{copy}</div></div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY),
        margin=dict(l=24, r=24, t=42, b=24),
    )
    return fig


def _plotly_confusion_matrix(cm: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred Non-Churn", "Pred Churn"],
            y=["Actual Non-Churn", "Actual Churn"],
            colorscale=[[0, "#1A1A1A"], [1, ACCENT_COLOR]],
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(title=title)
    return _plotly_theme(fig)


def _plotly_probability_hist(probabilities: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Histogram(
                x=probabilities,
                nbinsx=24,
                marker=dict(color=ACCENT_COLOR, line=dict(color="#111111", width=1)),
                opacity=0.92,
            )
        ]
    )
    fig.update_layout(title="Churn Probability Distribution", xaxis_title="Predicted probability", yaxis_title="Players")
    return _plotly_theme(fig)


def _plotly_risk_distribution(probability_frame: pd.DataFrame) -> go.Figure:
    risk_counts = probability_frame["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    fig = go.Figure(
        data=[
            go.Bar(
                x=risk_counts.index.tolist(),
                y=risk_counts.values.tolist(),
                marker=dict(color=["#22C55E", "#F59E0B", "#EF4444"]),
            )
        ]
    )
    fig.update_layout(title="Risk Bucket Distribution", xaxis_title="Risk level", yaxis_title="Players")
    return _plotly_theme(fig)


def _plotly_roc_curve(y_true: pd.Series, probabilities: np.ndarray, roc_auc: float) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=ACCENT_COLOR, width=3), name="ROC"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=TEXT_MUTED, width=1, dash="dash"),
            name="Baseline",
        )
    )
    fig.update_layout(title=f"ROC Curve (AUC {roc_auc:.3f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _plotly_theme(fig)


def _plotly_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    top_df = importance_df.head(10).sort_values("importance", ascending=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=top_df["importance"],
                y=top_df["feature"],
                orientation="h",
                marker=dict(color=ACCENT_COLOR),
            )
        ]
    )
    fig.update_layout(title="Feature Importance", xaxis_title="Importance", yaxis_title="Feature")
    return _plotly_theme(fig)


def _format_file_size(size_bytes: Optional[int]) -> str:
    if not size_bytes:
        return "Unknown"
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def _page_divider() -> None:
    st.markdown("<div class='report-divider'></div>", unsafe_allow_html=True)


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


def load_sample_data() -> pd.DataFrame:
    if not SAMPLE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Sample dataset not found at {SAMPLE_DATASET_PATH}")
    return pd.read_csv(SAMPLE_DATASET_PATH)


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
        <div style="padding:0.75rem 1rem;border-radius:0.9rem;background:{color}18;border:1px solid {color};display:inline-block;">
            <span style="font-weight:700;color:{color};">{risk_level} Risk</span>
            <span style="margin-left:0.55rem;color:{TEXT_PRIMARY};">{probability:.1%} churn probability</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _truncate_text(value: object, max_chars: int = 140) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _render_chip_row(items: List[str], accent: str = "#0f172a") -> None:
    if not items:
        return
    chips = "".join(
        f"<span style='display:inline-block;padding:0.18rem 0.55rem;margin:0 0.35rem 0.35rem 0;"
        f"border-radius:999px;background:{accent}14;border:1px solid {accent}33;color:{accent};"
        f"font-size:0.78rem;font-weight:600;'>{item}</span>"
        for item in items
    )
    st.markdown(chips, unsafe_allow_html=True)


def _render_summary_card(label: str, value: str, note: Optional[str] = None, accent: str = "#2563eb") -> None:
    _metric_panel(label, value, note=note, accent=accent)


def _render_recommendation_preview(rec, index: int) -> None:
    with st.container():
        title_cols = st.columns([3, 1])
        with title_cols[0]:
            st.markdown(f"**{index}. {rec.title}**")
        with title_cols[1]:
            st.caption(rec.effort)

        st.write(_truncate_text(rec.rationale, 170))
        st.write(f"Expected impact: {_truncate_text(rec.expected_impact, 120)}")
        _render_chip_row([rec.risk] + rec.metrics_to_track[:2], accent="#0f766e")

        if rec.supporting_signals:
            st.caption("Signals: " + ", ".join(rec.supporting_signals[:2]))

        with st.expander("Details", expanded=False):
            if rec.action_steps:
                st.markdown("**Action steps**")
                for step in rec.action_steps[:3]:
                    st.write(f"- {step}")
            if rec.metrics_to_track:
                st.markdown("**Metrics to track**")
                st.write(", ".join(rec.metrics_to_track))
            if rec.references:
                st.markdown("**References**")
                for ref in rec.references:
                    if ref.url:
                        st.markdown(f"- [{ref.title}]({ref.url}) ({ref.source})")
                    else:
                        st.write(f"- {ref.title} ({ref.source})")
            if rec.uncertainty_notes:
                st.markdown("**Uncertainty notes**")
                for note in rec.uncertainty_notes:
                    st.write(f"- {note}")


def _render_strategies_preview(report) -> None:
    if not report.retrieved_strategies:
        return

    with st.expander("Retrieved strategies", expanded=False):
        for strategy in report.retrieved_strategies:
            with st.container():
                st.markdown(f"**{strategy.title}**")
                st.caption(f"{strategy.source} · score {strategy.score:.2f}")
                st.write(_truncate_text(strategy.when_to_use, 180))
                if strategy.matched_signals:
                    _render_chip_row(strategy.matched_signals[:3], accent="#7c3aed")
                if strategy.url:
                    st.markdown(f"[Open source]({strategy.url})")


def _render_quality_summary(report) -> None:
    summary_cols = st.columns(4)
    summary_cols[0].metric("Risk", f"{report.churn_risk_interpretation['risk_level']} ({report.churn_risk_interpretation['risk_probability']:.0%})")
    summary_cols[1].metric("Mode", report.workflow_mode.title())
    summary_cols[2].metric("Retrieved", f"{len(report.retrieved_strategies)}")
    summary_cols[3].metric("Data Coverage", f"{report.analysis_summary.get('coverage_score', 0.0):.0%}")

    status_bits = []
    status_bits.append("Retrieval-backed")
    if report.data_quality_notes:
        status_bits.append("Some noisy data")
    _render_chip_row(status_bits, accent="#1d4ed8")


def _render_hero_card(report) -> None:
    top_rec = report.engagement_and_retention_recommendations[0] if report.engagement_and_retention_recommendations else None
    accent = _risk_color(report.churn_risk_interpretation.get("risk_level", "Medium"))
    hero_style = f"""
        <div style="
            padding: 1.45rem 1.55rem;
            border-radius: 1.15rem;
            background: linear-gradient(135deg, rgba(26,26,26,0.98) 0%, {accent}15 100%);
            border: 1px solid {accent}28;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24);
            margin: 0.25rem 0 0.35rem 0;
        ">
            <div style="font-size:0.75rem;letter-spacing:0.16em;text-transform:uppercase;color:{accent};font-weight:800;">
                Executive focus
            </div>
            <div style="font-size:1.3rem;font-weight:850;margin-top:0.35rem;color:{TEXT_PRIMARY};line-height:1.2;">
                {top_rec.title if top_rec else 'No recommendation generated'}
            </div>
            <div style="margin-top:0.55rem;color:{TEXT_MUTED};line-height:1.55;font-size:0.98rem;max-width:62ch;">
                {_truncate_text(top_rec.rationale, 180) if top_rec else 'The assistant did not generate a recommendation for this player.'}
            </div>
        </div>
    """
    with st.container():
        st.markdown(hero_style, unsafe_allow_html=True)
        if top_rec:
            chips = [
                f"Effort: {top_rec.effort}",
                f"Risk: {top_rec.risk}",
                f"Confidence: {int((top_rec.confidence or 0.0) * 100)}%",
            ]
            _render_chip_row(chips, accent=accent)


def _render_dashboard_cards(report) -> None:
    top_recommendations = report.engagement_and_retention_recommendations[:3]
    if not top_recommendations:
        st.info("No recommendations were generated for this player.")
        return

    card_count = min(3, len(top_recommendations))
    rec_cols = st.columns(card_count)
    for idx, rec in enumerate(top_recommendations):
        with rec_cols[idx]:
            _render_recommendation_preview(rec, idx + 1)


def _render_executive_summary(report) -> None:
    top_rec = report.engagement_and_retention_recommendations[0] if report.engagement_and_retention_recommendations else None
    st.markdown("### Summary")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Risk", f"{report.churn_risk_interpretation['risk_level']} ({report.churn_risk_interpretation['risk_probability']:.0%})")
    summary_cols[1].metric("Data coverage", f"{report.analysis_summary.get('coverage_score', 0.0):.0%}")
    summary_cols[2].metric("Retrieved strategies", f"{len(report.retrieved_strategies)}")

    if top_rec:
        st.markdown("### Next Best Action")
        _render_recommendation_preview(top_rec, 1)

    if report.data_quality_notes:
        with st.expander("Data quality notes", expanded=False):
            for note in report.data_quality_notes[:4]:
                st.write(f"- {note}")


def _render_full_details(report, state, player_identifier: str) -> None:
    with st.expander("Full analysis", expanded=True):
        st.markdown("### Player Behavior Summary")
        st.json(report.player_behavior_summary)
        _page_divider()
        st.markdown("### Churn Risk Interpretation")
        st.json(report.churn_risk_interpretation)
        _page_divider()
        st.markdown("### Analysis Summary")
        st.json(report.analysis_summary)

    _render_strategies_preview(report)

    with st.expander("Supporting references", expanded=False):
        for ref in report.supporting_references:
            if ref.url:
                st.markdown(f"- [{ref.title}]({ref.url}) ({ref.source})")
            else:
                st.write(f"- {ref.title} ({ref.source})")

    if report.data_quality_notes:
        with st.expander("Data quality notes", expanded=False):
            for note in report.data_quality_notes:
                st.write(f"- {note}")

    with st.expander("Ethics and UX notes", expanded=False):
        for disclaimer in report.ethical_and_ux_disclaimers:
            st.write(f"- {disclaimer}")

    with st.expander("Agent state / audit trail", expanded=False):
        st.write(f"Current step: `{state.step}`")
        st.json([e.__dict__ for e in state.events])


def _render_report_exports(report, player_identifier: str) -> None:
    report_json = report.to_dict()
    payload = build_report_payload(report)
    missing_sections = validate_report_payload(payload)

    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button(
            "Download Report (JSON)",
            data=json.dumps(report_json, indent=2, ensure_ascii=True),
            file_name=f"engagement_optimization_report_{player_identifier}.json",
            mime="application/json",
            use_container_width=True,
        )

    with export_cols[1]:
        if missing_sections:
            st.error(validation_error_message(missing_sections))
        else:
            pdf_bytes = generate_report_pdf_bytes(payload)
            st.download_button(
                "Download as PDF",
                data=pdf_bytes,
                file_name=f"engagement_optimization_report_{player_identifier}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


def _render_active_section(section: str, df: Optional[pd.DataFrame]) -> None:
    if section == "Upload Data":
        _page_header(
            "Data Overview",
            "Ingest your player dataset, inspect schema quality, and confirm the churn target before training.",
        )

        def sample_card() -> None:
            sample_cols = st.columns([4, 1.2])
            with sample_cols[0]:
                st.markdown(
                    """
                    <div class='sample-card'>
                        <div>
                            <div style='color:#F5F5F5;font-weight:700;font-size:1.05rem;margin-top:0.35rem;'>Sample Dataset</div>
                            <div class='section-note'>Load the bundled online gaming behavior CSV for a fast demo of the full churn workflow.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if SAMPLE_DATASET_PATH.exists():
                    st.caption(f"Bundled file: `{SAMPLE_DATASET_PATH.name}`")
            with sample_cols[1]:
                if st.button("Use Sample Dataset", type="primary", use_container_width=True, key="use_sample_dataset"):
                    try:
                        sample_df = load_sample_data()
                        st.session_state["raw_df"] = sample_df
                        st.session_state["dataset_source"] = "sample"
                        st.session_state["dataset_name"] = SAMPLE_DATASET_PATH.name
                        st.session_state["dataset_size_bytes"] = SAMPLE_DATASET_PATH.stat().st_size
                        st.success("Sample dataset loaded.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to load sample dataset: {exc}")

        card(sample_card)

        if df is None:
            _info_banner("Upload a CSV from the left control rail or load the bundled sample dataset to begin.")
            return

        if df.empty:
            st.error("The uploaded CSV is empty.")
            return

        dataset_label = st.session_state.get("dataset_name", "Uploaded CSV")
        dataset_source = st.session_state.get("dataset_source", "upload")
        dataset_size = st.session_state.get("dataset_size_bytes")

        target_col = _find_column_case_insensitive(df, "Churn")
        if target_col is None:
            target_col = df.columns[-1]
        churn_rate = None
        try:
            churn_rate = df[target_col].astype(str).str.strip().str.lower().eq("low").mean()
        except Exception:
            churn_rate = None

        metric_cols = st.columns(4)
        with metric_cols[0]:
            _metric_panel("Total Players", f"{df.shape[0]:,}", note=dataset_label)
        with metric_cols[1]:
            _metric_panel("Features", f"{df.shape[1]:,}", note=f"Source: {dataset_source}")
        with metric_cols[2]:
            _metric_panel("Churn Rate", f"{(churn_rate or 0):.1%}" if churn_rate is not None else "N/A", note=f"Target: {target_col}")
        with metric_cols[3]:
            _metric_panel("File Size", _format_file_size(dataset_size), note="Current dataset")

        _info_banner(f"Current dataset: {dataset_label} ({dataset_source})")

        def preview_card() -> None:
            st.markdown("<div class='section-note'>Preview the first 20 rows to confirm feature names and data types.</div>", unsafe_allow_html=True)
            st.dataframe(df.head(20), use_container_width=True)

        card(preview_card, title="Raw Dataset Preview")

        class_dist = df[target_col].astype(str).value_counts(dropna=False).rename_axis("Class").reset_index(name="Count")
        card(lambda: st.dataframe(class_dist, use_container_width=True), title=f"Class Distribution ({target_col})")
        return

    if section == "Model Training":
        _page_header(
            "Model Training",
            "Configure the split, train the decision tree pipeline, and inspect the model metrics in a cleaner workflow.",
        )
        if df is None:
            _info_banner("Upload data first from the left control rail before training.")
            return

        _step_indicator(3)

        def parameter_card() -> None:
            st.markdown("<div class='section-note'>Tune the training split and tree depth before fitting the model.</div>", unsafe_allow_html=True)
            nonlocal_test_size = st.slider(
                "Train/Test split (test proportion)",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
            )
            nonlocal_depth_slider = st.slider(
                "Decision Tree max depth (0 = no limit)",
                min_value=0,
                max_value=20,
                value=6,
            )
            st.session_state["_ui_test_size"] = nonlocal_test_size
            st.session_state["_ui_depth_slider"] = nonlocal_depth_slider

        card(parameter_card, title="Training Parameters")
        test_size = st.session_state.get("_ui_test_size", 0.20)
        depth_slider = st.session_state.get("_ui_depth_slider", 6)
        dt_max_depth = None if depth_slider == 0 else depth_slider

        if st.button("Train Model", type="primary", use_container_width=True):
            try:
                progress = st.progress(0.0, text="Preparing training pipeline")
                bundle = preprocess_data(
                    df,
                    st.session_state["target_col"],
                    positive_classes=st.session_state.get("positive_classes", []),
                )

                x_model = bundle["X_model"]
                y = bundle["y"]
                preprocessor = bundle["preprocessor"]
                class_weight = "balanced" if bundle["is_imbalanced"] else None
                progress.progress(0.35, text="Splitting data")

                stratify_y = y if y.value_counts().min() >= 2 else None
                x_train, x_test, y_train, y_test = train_test_split(
                    x_model,
                    y,
                    test_size=test_size,
                    random_state=RANDOM_STATE,
                    stratify=stratify_y,
                )
                progress.progress(0.65, text="Training decision tree model")

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
                progress.progress(0.9, text="Computing evaluation metrics")
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
                st.session_state["evaluation_bundle"] = {
                    "X_test": x_test,
                    "y_test": y_test,
                    "y_prob": {"Decision Tree": y_prob},
                }
                progress.progress(1.0, text="Training complete")

                st.success("Decision Tree training completed.")
                metric_cols = st.columns(5)
                with metric_cols[0]:
                    _metric_panel("Accuracy", f"{metrics['Accuracy']:.4f}", note="Test split")
                with metric_cols[1]:
                    _metric_panel("Precision", f"{metrics['Precision']:.4f}")
                with metric_cols[2]:
                    _metric_panel("Recall", f"{metrics['Recall']:.4f}")
                with metric_cols[3]:
                    _metric_panel("F1", f"{metrics['F1']:.4f}")
                with metric_cols[4]:
                    _metric_panel("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
        return

    if section == "Model Evaluation":
        _page_header(
            "Model Evaluation",
            "Compare performance, inspect risk distributions, and drill into the prediction output with dark-mode analytics cards.",
        )
        if "trained_models" not in st.session_state:
            _info_banner("Train models first from the Model Training section.")
            return

        metrics_map = st.session_state["model_metrics"]
        models = st.session_state["trained_models"]
        cm_map = st.session_state["confusion_matrices"]
        bundle = st.session_state["data_bundle"]
        evaluation_bundle = st.session_state.get("evaluation_bundle", {})

        comparison_df = pd.DataFrame(metrics_map).T
        card(lambda: st.dataframe(comparison_df.round(4), use_container_width=True), title="Model Comparison")

        selected_model = st.selectbox("Detailed evaluation model", options=list(models.keys()))
        selected_metrics = metrics_map[selected_model]

        metric_cols = st.columns(5)
        with metric_cols[0]:
            _metric_panel("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
        with metric_cols[1]:
            _metric_panel("Precision", f"{selected_metrics['Precision']:.4f}")
        with metric_cols[2]:
            _metric_panel("Recall", f"{selected_metrics['Recall']:.4f}")
        with metric_cols[3]:
            _metric_panel("F1", f"{selected_metrics['F1']:.4f}")
        with metric_cols[4]:
            _metric_panel("ROC-AUC", f"{selected_metrics['ROC-AUC']:.4f}")

        eval_left, eval_right = st.columns(2)
        with eval_left:
            cm = cm_map[selected_model]

            def confusion_card() -> None:
                st.markdown("<div class='section-note'>Inspect false positives and false negatives on the test split.</div>", unsafe_allow_html=True)
                st.plotly_chart(
                    _plotly_confusion_matrix(cm, f"{selected_model} Confusion Matrix"),
                    use_container_width=True,
                )

            card(confusion_card, title="Confusion Matrix")

        probabilities = models[selected_model].predict_proba(bundle["X_model"])[:, 1]
        probability_frame = bundle["X_display"].copy()
        probability_frame["churn_probability"] = probabilities
        probability_frame["risk_level"] = probability_frame["churn_probability"].apply(_risk_bucket)

        with eval_right:
            def probability_card() -> None:
                st.markdown("<div class='section-note'>Review how confidently the model separates low- and high-risk players.</div>", unsafe_allow_html=True)
                st.plotly_chart(_plotly_probability_hist(probability_frame["churn_probability"].to_numpy()), use_container_width=True)

            card(probability_card, title="Probability Distribution")

        lower_row_left, lower_row_right = st.columns(2)
        with lower_row_left:
            y_test = evaluation_bundle.get("y_test")
            y_prob_map = evaluation_bundle.get("y_prob", {})

            def roc_card() -> None:
                st.markdown("<div class='section-note'>Threshold-independent performance view for the selected model.</div>", unsafe_allow_html=True)
                if y_test is None or selected_model not in y_prob_map:
                    st.warning("ROC curve is unavailable until a test split is stored for this model.")
                    return
                st.plotly_chart(
                    _plotly_roc_curve(y_test, np.asarray(y_prob_map[selected_model]), float(selected_metrics["ROC-AUC"])),
                    use_container_width=True,
                )

            card(roc_card, title="ROC Curve")

        with lower_row_right:
            def risk_card() -> None:
                st.markdown("<div class='section-note'>Operational view of how many players fall into each risk band.</div>", unsafe_allow_html=True)
                st.plotly_chart(_plotly_risk_distribution(probability_frame), use_container_width=True)

            card(risk_card, title="Risk Distribution")

        _page_divider()
        st.markdown("### Interactive Risk Filter")
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

        _info_banner(f"Filtered players available for export: {len(filtered):,}")
        card(
            lambda: st.dataframe(filtered.sort_values("churn_probability", ascending=False).head(500), use_container_width=True),
            title="Filtered Risk Table",
        )

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Risk Predictions CSV",
            data=csv_data,
            file_name="player_churn_risk_predictions.csv",
            mime="text/csv",
        )

        tree_model = models[selected_model].named_steps["model"]
        preprocessor = models[selected_model].named_steps["preprocessor"]
        importances = tree_model.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(10)
        )
        card(
            lambda: st.plotly_chart(_plotly_feature_importance(importance_df), use_container_width=True),
            title="Feature Importance",
        )
        return

    if section == "Player Risk Analysis":
        _page_header(
            "Player Risk Analysis",
            "Inspect an existing player profile or simulate a new one to understand churn risk at the individual level.",
        )
        if "trained_models" not in st.session_state:
            _info_banner("Train models first from the Model Training section.")
            return

        models = st.session_state["trained_models"]
        bundle = st.session_state["data_bundle"]

        selected_model_name = st.selectbox("Model for prediction", options=list(models.keys()))
        selected_model = models[selected_model_name]

        def selector_card() -> None:
            st.markdown("<div class='section-note'>Choose a player from the dataset and score them with the trained model.</div>", unsafe_allow_html=True)
            selected = st.selectbox("Select player row index", options=bundle["X_model"].index.tolist())
            st.session_state["_selected_player_row"] = selected

        card(selector_card, title="Existing Player Predictor")
        row_index = st.session_state.get("_selected_player_row", bundle["X_model"].index.tolist()[0])
        card(lambda: st.dataframe(bundle["X_display"].loc[[row_index]], use_container_width=True), title="Player Behavior Snapshot")

        if st.button("Predict Selected Player", use_container_width=True):
            selected_row = bundle["X_model"].loc[[row_index]]
            probability = float(selected_model.predict_proba(selected_row)[0, 1])
            risk = _risk_bucket(probability)
            risk_cols = st.columns(2)
            with risk_cols[0]:
                _metric_panel("Churn Probability", f"{probability:.4f}", note=selected_model_name, accent=_risk_color(risk))
            with risk_cols[1]:
                _metric_panel("Risk Level", risk, note="Selected player", accent=_risk_color(risk))

        _page_divider()
        st.markdown("### Single Player Simulator")
        numeric_features = bundle["numeric_features"]
        x_model = bundle["X_model"]

        if not numeric_features:
            st.warning("No numeric features available for simulator inputs.")
            return

        input_values = {}

        def simulator_card() -> None:
            input_cols = st.columns(3)
            for idx, feature in enumerate(numeric_features):
                default_value = float(pd.to_numeric(x_model[feature], errors="coerce").median())
                with input_cols[idx % 3]:
                    input_values[feature] = st.number_input(
                        feature,
                        value=default_value if not np.isnan(default_value) else 0.0,
                        key=f"sim_{feature}",
                    )

        card(simulator_card, title="Simulation Inputs")

        simulator_row = {}
        for feature in x_model.columns:
            if feature in input_values:
                simulator_row[feature] = input_values[feature]
            elif feature in bundle["categorical_modes"]:
                simulator_row[feature] = bundle["categorical_modes"][feature]
            else:
                simulator_row[feature] = 0

        simulator_df = pd.DataFrame([simulator_row], columns=x_model.columns)
        sim_probability = float(selected_model.predict_proba(simulator_df)[0, 1])
        sim_risk = _risk_bucket(sim_probability)

        sim_col_1, sim_col_2 = st.columns([1, 2])
        with sim_col_1:
            _metric_panel("Simulated Churn Probability", f"{sim_probability:.4f}", accent=_risk_color(sim_risk))
        with sim_col_2:
            _metric_panel("Simulated Risk Level", sim_risk, note="Behavioral simulation", accent=_risk_color(sim_risk))
        return

    if section == "Decision Tree Explorer":
        _page_header(
            "Decision Tree Explorer",
            "Explore the learned decision path visually and inspect the exact split rules behind the churn model.",
        )
        if "trained_models" not in st.session_state:
            _info_banner("Train models first from the Model Training section.")
            return

        dt_pipeline = st.session_state["trained_models"]["Decision Tree"]
        dt_model = dt_pipeline.named_steps["model"]

        max_tree_depth = max(1, dt_model.get_depth())
        view_depth = st.slider(
            "Display tree depth",
            min_value=1,
            max_value=max_tree_depth,
            value=min(3, max_tree_depth),
        )

        preprocessor = dt_pipeline.named_steps["preprocessor"]
        feature_names = list(preprocessor.get_feature_names_out())

        def tree_card() -> None:
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(
                dt_model,
                feature_names=feature_names,
                class_names=["Non-Churn", "Churn"],
                filled=True,
                rounded=True,
                max_depth=view_depth,
                impurity=False,
                proportion=True,
                fontsize=8,
                ax=ax,
            )
            ax.set_title("Decision Tree Explorer")
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(
                "<div class='legend-note'>Legend: darker red nodes indicate stronger churn propensity; follow left/right branches to understand threshold logic.</div>",
                unsafe_allow_html=True,
            )

        card(tree_card, title="Tree Visualization")

        split_text = export_text(dt_model, feature_names=feature_names, max_depth=view_depth)
        card(lambda: st.code(split_text), title="Decision Tree Feature Splits")
        return

    if section == "Engagement Optimization Assistant":
        _page_header(
            "Engagement Optimization Assistant",
            "Generate a retrieval-backed retention report for an individual player with structured recommendations and export options.",
        )

        if "trained_models" not in st.session_state:
            _info_banner("Train a model first from the Model Training section.")
            return

        models = st.session_state["trained_models"]
        bundle = st.session_state["data_bundle"]

        def assistant_controls() -> None:
            st.markdown("<div class='chat-heading'>Assistant Inputs</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='chat-copy'>Pick a trained model, choose the player row, and decide how much detail to show in the generated report.</div>",
                unsafe_allow_html=True,
            )
            st.selectbox("Model", options=list(models.keys()), key="assistant_model")
            st.selectbox(
                "Select player row index",
                options=bundle["X_model"].index.tolist(),
                key="assistant_row",
            )
            st.radio(
                "Presentation mode",
                ["Executive summary", "Dashboard", "Full details"],
                horizontal=True,
                index=0,
                key="assistant_presentation_mode",
            )

        card(assistant_controls, title="Assistant Control Panel")

        selected_model_name = st.session_state["assistant_model"]
        selected_model = models[selected_model_name]
        id_col = bundle.get("id_col")
        row_index = st.session_state["assistant_row"]
        player_display_row = bundle["X_display"].loc[row_index]

        player_identifier = None
        if id_col and id_col in bundle["X_display"].columns:
            try:
                player_identifier = str(player_display_row.get(id_col))
            except Exception:
                player_identifier = None
        if not player_identifier:
            player_identifier = f"row_{row_index}"
        presentation_mode = st.session_state["assistant_presentation_mode"]

        with st.expander("Player snapshot", expanded=False):
            st.dataframe(bundle["X_display"].loc[[row_index]], use_container_width=True)

        if st.button("Generate Engagement Optimization Report", type="primary", use_container_width=True):
            try:
                player_model_row = bundle["X_model"].loc[[row_index]]
                if hasattr(selected_model, "predict_proba"):
                    churn_probability = float(selected_model.predict_proba(player_model_row)[0, 1])
                elif hasattr(selected_model, "decision_function"):
                    decision_value = float(selected_model.decision_function(player_model_row)[0])
                    churn_probability = 1.0 / (1.0 + np.exp(-decision_value))
                else:
                    churn_probability = float(selected_model.predict(player_model_row)[0])

                report, state = generate_engagement_optimization_report(
                    player_features=player_display_row.to_dict(),
                    churn_probability=churn_probability,
                    player_identifier=player_identifier,
                )

                st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
                _render_risk_badge(report.churn_risk_interpretation["risk_level"], report.churn_risk_interpretation["risk_probability"])
                _render_quality_summary(report)
                st.caption("Workflow mode: retrieval-backed recommendations only.")
                _render_hero_card(report)
                _render_report_exports(report, player_identifier)
                _page_divider()

                if presentation_mode == "Executive summary":
                    _render_executive_summary(report)
                elif presentation_mode == "Dashboard":
                    st.markdown("### Top actions")
                    _render_dashboard_cards(report)
                    with st.expander("Why these recommendations?", expanded=False):
                        st.write(_truncate_text(report.analysis_summary.get("risk_summary", {}).get("risk_text", ""), 240))
                        if report.data_quality_notes:
                            st.write("Data quality notes:")
                            for note in report.data_quality_notes[:3]:
                                st.write(f"- {note}")
                    _render_strategies_preview(report)
                else:
                    st.markdown("### Top actions")
                    _render_dashboard_cards(report)
                    with st.expander("Why these recommendations?", expanded=False):
                        st.write(_truncate_text(report.analysis_summary.get("risk_summary", {}).get("risk_text", ""), 240))
                        if report.data_quality_notes:
                            st.write("Data quality notes:")
                            for note in report.data_quality_notes[:3]:
                                st.write(f"- {note}")
                    _render_full_details(report, state, player_identifier)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Failed to generate report: {exc}")
        return


def main() -> None:
    st.set_page_config(
        page_title="ChurnIQ",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_global_styles()

    shell_left, shell_right = st.columns([1.15, 4.25], gap="large")

    with shell_left:
        st.markdown("<div class='shell-nav'>", unsafe_allow_html=True)
        st.markdown("<div class='nav-wordmark'>ChurnIQ</div>", unsafe_allow_html=True)
        st.markdown("<div class='nav-subtitle'>Tactical Analytics</div>", unsafe_allow_html=True)
        st.markdown("<div class='redline'></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-label'>Navigation</div>", unsafe_allow_html=True)
        section = st.radio(
            "Go to",
            [
                "Upload Data",
                "Model Training",
                "Model Evaluation",
                "Player Risk Analysis",
                "Decision Tree Explorer",
                "Engagement Optimization Assistant",
            ],
            label_visibility="collapsed",
        )
        st.markdown("<div class='redline'></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-label'>Data Ingest</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if st.button("Use Sample Dataset", type="primary", use_container_width=True, key="sidebar_use_sample_dataset"):
            try:
                sample_df = load_sample_data()
                st.session_state["raw_df"] = sample_df
                st.session_state["dataset_source"] = "sample"
                st.session_state["dataset_name"] = SAMPLE_DATASET_PATH.name
                st.session_state["dataset_size_bytes"] = SAMPLE_DATASET_PATH.stat().st_size
                st.success("Sample dataset loaded.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load sample dataset: {exc}")
        st.markdown("<div class='redline'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='panel-label'>System</div><div class='section-note'>Latency: 24ms · Cluster link stable · Tactical mode active</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    df = None
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state["raw_df"] = df
            st.session_state["dataset_source"] = "upload"
            st.session_state["dataset_name"] = getattr(uploaded_file, "name", "uploaded.csv")
            st.session_state["dataset_size_bytes"] = getattr(uploaded_file, "size", None)
        except Exception as exc:
            st.error(f"Failed to read uploaded file: {exc}")
            return
    elif "raw_df" in st.session_state:
        df = st.session_state["raw_df"]

    if df is not None and not df.empty:
        _initialize_target_state(df)

    with shell_right:
        st.markdown(
            """
            <div class='topbar'>
                <div class='topbar-search'>Terminal Query...</div>
                <div class='panel-label' style='margin:0;'>ChurnIQ // Mission Console</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_active_section(section, df)


if __name__ == "__main__":
    main()

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

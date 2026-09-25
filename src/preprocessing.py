import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_dataset


def build_preprocessor() -> ColumnTransformer:
    """
    Creates a ColumnTransformer to scale numerical features and one-hot encode categorical features.
    """
    num_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts transformed feature names after OneHotEncoding.
    """
    feature_names = list(NUMERICAL_FEATURES)
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cats = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names.extend(encoded_cats)
    except Exception:
        feature_names.extend([f"cat_{i}" for i in range(10)])
    return feature_names


def prepare_data(df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer, List[str]]:
    """
    Loads dataset, splits into train/test sets, fits preprocessor, and returns ready arrays.
    """
    if df is None:
        df = load_dataset()

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    feature_names = get_feature_names(preprocessor)

    return X_train, X_test, y_train, y_test, preprocessor, feature_names

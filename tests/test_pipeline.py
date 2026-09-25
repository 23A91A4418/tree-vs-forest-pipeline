import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import generate_synthetic_data, load_dataset
from src.preprocessing import prepare_data, build_preprocessor
from src.models import train_decision_tree, train_random_forest, get_feature_importances
from src.evaluate import evaluate_classification_model, select_champion_model


def test_data_generation():
    df = generate_synthetic_data(num_samples=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "is_churn" in df.columns
    assert set(df["is_churn"].unique()).issubset({0, 1})


def test_preprocessing_pipeline():
    df = generate_synthetic_data(num_samples=200)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    assert X_train.shape[0] == 160
    assert X_test.shape[0] == 40
    assert len(y_train) == 160
    assert len(y_test) == 40
    assert len(feature_names) > 0


def test_decision_tree_training():
    df = generate_synthetic_data(num_samples=200)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)
    
    dt_model, params, fit_time = train_decision_tree(X_train, y_train)
    assert dt_model is not None
    assert fit_time >= 0

    metrics = evaluate_classification_model(dt_model, X_test, y_test, model_name="Decision Tree")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert "confusion_matrix" in metrics


def test_random_forest_training():
    df = generate_synthetic_data(num_samples=200)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)
    
    rf_model, params, fit_time = train_random_forest(X_train, y_train)
    assert rf_model is not None

    metrics = evaluate_classification_model(rf_model, X_test, y_test, model_name="Random Forest")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_champion_selection():
    dt_metrics = {"f1_score": 0.75, "accuracy": 0.85}
    rf_metrics = {"f1_score": 0.82, "accuracy": 0.89}
    winner = select_champion_model(dt_metrics, rf_metrics, primary_metric="f1_score")
    assert winner == "Random Forest"

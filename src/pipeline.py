import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import joblib
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import mlflow
import mlflow.sklearn
from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    ARTIFACTS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    EVAL_METRIC,
)
from src.data_loader import load_dataset
from src.preprocessing import prepare_data
from src.models import (
    train_decision_tree,
    train_random_forest,
    get_feature_importances,
)
from src.evaluate import (
    evaluate_classification_model,
    select_champion_model,
)
from src.visualize import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importance_comparison,
    plot_tree_structure,
    plot_depth_overfitting_curve,
)


def run_bakeoff_pipeline() -> Dict[str, Any]:
    """
    Runs the complete Decision Tree vs. Random Forest ML Bake-Off pipeline with MLflow experiment tracking.
    Saves mandatory results to results/metrics.json and results/roc_comparison.png.
    """
    print("=" * 60)
    print("Starting Machine Learning Bake-Off Pipeline: Decision Tree vs Random Forest")
    print("=" * 60)

    # 1. Setup MLflow Tracking
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        print(f"[Pipeline] MLflow setup warning: {e}. Falling back to local tracking.")

    # 2. Data Preparation
    print("[Pipeline] Preparing dataset and building preprocessing pipeline...")
    df = load_dataset()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    # Save Preprocessor Artifact
    preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(preprocessor, RESULTS_DIR / "preprocessor.joblib")
    print(f"[Pipeline] Saved preprocessor to {preprocessor_path}")

    # 3. Train Decision Tree
    print("\n[Pipeline] Training & Tuning Decision Tree Classifier...")
    dt_model, dt_params, dt_train_time = train_decision_tree(X_train, y_train)
    dt_metrics = evaluate_classification_model(dt_model, X_test, y_test, model_name="Decision Tree")
    dt_importances = get_feature_importances(dt_model, feature_names)

    print(f" -> Decision Tree F1: {dt_metrics['f1_score']}, Accuracy: {dt_metrics['accuracy']}, Training Time: {dt_train_time}s")

    # 4. Train Random Forest
    print("\n[Pipeline] Training & Tuning Random Forest Classifier...")
    rf_model, rf_params, rf_train_time = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_classification_model(rf_model, X_test, y_test, model_name="Random Forest")
    rf_importances = get_feature_importances(rf_model, feature_names)

    print(f" -> Random Forest F1: {rf_metrics['f1_score']}, Accuracy: {rf_metrics['accuracy']}, Training Time: {rf_train_time}s")

    # 5. Generate Comparison Plots (saves results/roc_comparison.png)
    print("\n[Pipeline] Generating visualization plots...")
    plot_cm_path = plot_confusion_matrices(dt_metrics["confusion_matrix"], rf_metrics["confusion_matrix"])
    plot_roc_path = plot_roc_curves(dt_model, rf_model, X_test, y_test)  # Mandatory results/roc_comparison.png
    plot_imp_path = plot_feature_importance_comparison(dt_importances, rf_importances)
    plot_tree_path = plot_tree_structure(dt_model, feature_names)
    plot_depth_path = plot_depth_overfitting_curve(X_train, y_train, X_test, y_test)

    dt_run_id = "dt_run_local"
    rf_run_id = "rf_run_local"

    # 6. MLflow Logging - Decision Tree Run
    try:
        with mlflow.start_run(run_name="Decision_Tree_GridSearch") as dt_run:
            mlflow.log_params(dt_params)
            mlflow.log_param("model_type", "DecisionTreeClassifier")
            mlflow.log_param("train_time_sec", dt_train_time)
            
            mlflow.log_metrics({
                "accuracy": dt_metrics["accuracy"],
                "precision": dt_metrics["precision"],
                "recall": dt_metrics["recall"],
                "f1_score": dt_metrics["f1_score"],
                "roc_auc": dt_metrics["roc_auc"],
                "log_loss": dt_metrics["log_loss"],
                "avg_latency_ms": dt_metrics["avg_latency_ms"],
            })

            mlflow.sklearn.log_model(dt_model, "model")
            mlflow.log_artifact(str(plot_tree_path))
            dt_run_id = dt_run.info.run_id

        # 7. MLflow Logging - Random Forest Run
        with mlflow.start_run(run_name="Random_Forest_GridSearch") as rf_run:
            rf_log_params = {k: str(v) for k, v in rf_params.items()}
            mlflow.log_params(rf_log_params)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("train_time_sec", rf_train_time)

            mlflow.log_metrics({
                "accuracy": rf_metrics["accuracy"],
                "precision": rf_metrics["precision"],
                "recall": rf_metrics["recall"],
                "f1_score": rf_metrics["f1_score"],
                "roc_auc": rf_metrics["roc_auc"],
                "log_loss": rf_metrics["log_loss"],
                "avg_latency_ms": rf_metrics["avg_latency_ms"],
            })

            mlflow.sklearn.log_model(rf_model, "model")
            mlflow.log_artifact(str(plot_cm_path))
            mlflow.log_artifact(str(plot_roc_path))
            mlflow.log_artifact(str(plot_imp_path))
            rf_run_id = rf_run.info.run_id
    except Exception as e:
        print(f"[Pipeline] MLflow logging skipped/failed: {e}")

    # 8. Champion Model Selection & Registration
    champion_name = select_champion_model(dt_metrics, rf_metrics, primary_metric=EVAL_METRIC)
    print(f"\n[BAKEOFF WINNER] Champion Model Selected: {champion_name}")

    if champion_name == "Random Forest":
        champion_model = rf_model
        champion_run_id = rf_run_id
        champion_metrics = rf_metrics
        champion_params = rf_params
        champion_importances = rf_importances
    else:
        champion_model = dt_model
        champion_run_id = dt_run_id
        champion_metrics = dt_metrics
        champion_params = dt_params
        champion_importances = dt_importances

    # Save Champion Model Artifact for Direct FastAPI Serving
    champion_model_path = ARTIFACTS_DIR / "champion_model.joblib"
    joblib.dump(champion_model, champion_model_path)
    joblib.dump(champion_model, RESULTS_DIR / "champion_model.joblib")

    dt_model_path = ARTIFACTS_DIR / "decision_tree_model.joblib"
    rf_model_path = ARTIFACTS_DIR / "random_forest_model.joblib"
    joblib.dump(dt_model, dt_model_path)
    joblib.dump(rf_model, rf_model_path)

    # Save Bake-Off Summary Metadata JSON (Mandatory results/metrics.json)
    bakeoff_summary = {
        "timestamp": datetime.now().isoformat(),
        "primary_metric": EVAL_METRIC,
        "champion_model": champion_name,
        "champion_run_id": champion_run_id,
        "feature_names": feature_names,
        "models": {
            "decision_tree": {
                "run_id": dt_run_id,
                "params": dt_params,
                "metrics": dt_metrics,
                "feature_importances": dt_importances,
            },
            "random_forest": {
                "run_id": rf_run_id,
                "params": {k: str(v) for k, v in rf_params.items()},
                "metrics": rf_metrics,
                "feature_importances": rf_importances,
            },
        },
    }

    # Save to mandatory results/metrics.json and artifacts/latest_bakeoff_summary.json
    metrics_json_path = RESULTS_DIR / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(bakeoff_summary, f, indent=2)

    summary_path = ARTIFACTS_DIR / "latest_bakeoff_summary.json"
    with open(summary_path, "w") as f:
        json.dump(bakeoff_summary, f, indent=2)

    print(f"[Pipeline] Saved champion model to {champion_model_path}")
    print(f"[Pipeline] Saved mandatory results to {metrics_json_path} and {plot_roc_path}")
    print("=" * 60)

    return bakeoff_summary


if __name__ == "__main__":
    run_bakeoff_pipeline()

import time
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)


def evaluate_classification_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluates a classification model on test data and computes detailed metrics and latency.
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    total_latency_sec = time.time() - start_time
    avg_latency_ms = round((total_latency_sec / len(X_test)) * 1000, 4)

    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    precision = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
    recall = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

    roc_auc = round(float(roc_auc_score(y_test, y_pred_proba)), 4) if y_pred_proba is not None else 0.0
    loss = round(float(log_loss(y_test, y_pred_proba)), 4) if y_pred_proba is not None else 0.0

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "log_loss": loss,
        "avg_latency_ms": avg_latency_ms,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }

    return metrics


def select_champion_model(
    dt_metrics: Dict[str, Any],
    rf_metrics: Dict[str, Any],
    primary_metric: str = "f1_score",
) -> str:
    """
    Compares metrics between Decision Tree and Random Forest and picks winner.
    """
    dt_val = dt_metrics.get(primary_metric, 0)
    rf_val = rf_metrics.get(primary_metric, 0)

    if rf_val >= dt_val:
        return "Random Forest"
    else:
        return "Decision Tree"

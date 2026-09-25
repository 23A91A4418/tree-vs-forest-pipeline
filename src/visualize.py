import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, auc
from src.config import PLOTS_DIR, RESULTS_DIR

sns.set_theme(style="darkgrid")


def plot_confusion_matrices(
    dt_cm: dict,
    rf_cm: dict,
    save_path: Path = RESULTS_DIR / "confusion_matrix_comparison.png",
):
    """
    Plots side-by-side confusion matrix heatmaps for Decision Tree vs Random Forest.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dt_matrix = np.array([[dt_cm["tn"], dt_cm["fp"]], [dt_cm["fn"], dt_cm["tp"]]])
    rf_matrix = np.array([[rf_cm["tn"], rf_cm["fp"]], [rf_cm["fn"], rf_cm["tp"]]])

    sns.heatmap(dt_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Decision Tree Confusion Matrix", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_xticklabels(["No Churn", "Churn"])
    axes[0].set_yticklabels(["No Churn", "Churn"])

    sns.heatmap(rf_matrix, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False)
    axes[1].set_title("Random Forest Confusion Matrix", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_xticklabels(["No Churn", "Churn"])
    axes[1].set_yticklabels(["No Churn", "Churn"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "confusion_matrix_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_roc_curves(
    dt_model,
    rf_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path = RESULTS_DIR / "roc_comparison.png",
):
    """
    Plots ROC Curves comparing Decision Tree and Random Forest models.
    Saves to mandatory path results/roc_comparison.png.
    """
    plt.figure(figsize=(8, 6))

    if hasattr(dt_model, "predict_proba"):
        dt_probs = dt_model.predict_proba(X_test)[:, 1]
        fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
        auc_dt = auc(fpr_dt, tpr_dt)
        plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.3f})", color="#e74c3c", lw=2)

    if hasattr(rf_model, "predict_proba"):
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
        auc_rf = auc(fpr_rf, tpr_rf)
        plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})", color="#2ecc71", lw=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance", lw=1.5)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve Comparison: Decision Tree vs Random Forest", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    
    # Save to mandatory results/ path and plots/ path
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "roc_curve_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_feature_importance_comparison(
    dt_importances: dict,
    rf_importances: dict,
    top_n: int = 10,
    save_path: Path = RESULTS_DIR / "feature_importance_comparison.png",
):
    """
    Plots side-by-side feature importance bar charts.
    """
    top_features = list(rf_importances.keys())[:top_n]
    
    dt_vals = [dt_importances.get(f, 0.0) for f in top_features]
    rf_vals = [rf_importances.get(f, 0.0) for f in top_features]

    df_plot = pd.DataFrame({
        "Feature": top_features * 2,
        "Importance": dt_vals + rf_vals,
        "Model": ["Decision Tree"] * len(top_features) + ["Random Forest"] * len(top_features),
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x="Importance", y="Feature", hue="Model", palette=["#e74c3c", "#2ecc71"])
    plt.title(f"Top {top_n} Feature Importances Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Gini Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "feature_importance_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_tree_structure(
    dt_model,
    feature_names: list,
    max_depth: int = 3,
    save_path: Path = RESULTS_DIR / "decision_tree_structure.png",
):
    """
    Plots Decision Tree nodes up to max_depth.
    """
    plt.figure(figsize=(16, 8))
    plot_tree(
        dt_model,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=["No Churn", "Churn"],
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title(f"Decision Tree Visualization (Max Depth = {max_depth})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "decision_tree_structure.png", dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_depth_overfitting_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path = RESULTS_DIR / "depth_overfitting_analysis.png",
):
    """
    Plots Train vs Test Accuracy/F1 as tree depth increases to illustrate overfitting dynamics.
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score

    depths = list(range(1, 16))
    train_scores = []
    test_scores = []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        train_scores.append(f1_score(y_train, clf.predict(X_train)))
        test_scores.append(f1_score(y_test, clf.predict(X_test)))

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_scores, "o-", label="Train F1 Score (Overfitting)", color="#3498db", lw=2)
    plt.plot(depths, test_scores, "s-", label="Test F1 Score (Generalization)", color="#e67e22", lw=2)
    plt.xlabel("Decision Tree Max Depth", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("Decision Tree Overfitting Analysis: Depth vs Performance", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "depth_overfitting_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()
    return save_path

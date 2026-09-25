import time
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.config import (
    DECISION_TREE_PARAM_GRID,
    RANDOM_FOREST_PARAM_GRID,
    RANDOM_STATE,
)


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Any] = None,
    cv: int = 3,
) -> Tuple[DecisionTreeClassifier, Dict[str, Any], float]:
    """
    Trains a DecisionTreeClassifier using GridSearchCV over the hyperparameter grid.
    Returns (best_estimator, best_params, fit_time_seconds).
    """
    if param_grid is None:
        param_grid = DECISION_TREE_PARAM_GRID

    base_dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=base_dt,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    fit_time = round(time.time() - start_time, 4)

    return grid_search.best_estimator_, grid_search.best_params_, fit_time


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Any] = None,
    cv: int = 3,
) -> Tuple[RandomForestClassifier, Dict[str, Any], float]:
    """
    Trains a RandomForestClassifier using GridSearchCV over the hyperparameter grid.
    Returns (best_estimator, best_params, fit_time_seconds).
    """
    if param_grid is None:
        param_grid = RANDOM_FOREST_PARAM_GRID

    base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    fit_time = round(time.time() - start_time, 4)

    return grid_search.best_estimator_, grid_search.best_params_, fit_time


def get_feature_importances(model: Any, feature_names: list) -> Dict[str, float]:
    """
    Extracts feature importance dictionary sorted in descending order.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_dict = {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, importances)
        }
        return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    return {}

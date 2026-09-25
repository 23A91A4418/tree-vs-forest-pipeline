import os
from pathlib import Path

# Paths Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE_PATH = DATA_DIR / "customer_churn.csv"

# Dataset Feature Definitions
NUMERICAL_FEATURES = [
    "age",
    "annual_income",
    "credit_score",
    "tenure_months",
    "account_balance",
    "monthly_usage_hours",
    "support_tickets",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "device_category",
]

FEATURE_NAMES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "is_churn"

# MLflow & Server Configuration from Environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:///{BASE_DIR.as_posix()}/mlruns")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Decision_Tree_vs_Random_Forest_Bakeoff")
REGISTERED_MODEL_NAME = "Champion_Churn_Predictor"

# Training & Hyperparameter Search Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
EVAL_METRIC = os.getenv("EVAL_METRIC", "f1_score")

# Hyperparameter Grids for Grid Search
DECISION_TREE_PARAM_GRID = {
    "max_depth": [3, 5, 8],
    "min_samples_split": [2, 5],
    "criterion": ["gini", "entropy"],
}

RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [30, 60],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt"],
    "bootstrap": [True],
}

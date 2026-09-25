import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    DATA_FILE_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RANDOM_STATE,
)


def generate_synthetic_data(num_samples: int = 2500, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a realistic tabular customer churn dataset with synthetic non-linear
    relationships, interactions, and noise suitable for Decision Tree vs. Random Forest evaluation.
    """
    np.random.seed(random_state)

    age = np.random.randint(18, 76, size=num_samples)
    annual_income = np.random.uniform(25000, 180000, size=num_samples).round(2)
    credit_score = np.random.randint(350, 850, size=num_samples)
    tenure_months = np.random.randint(1, 72, size=num_samples)
    account_balance = np.random.uniform(0, 150000, size=num_samples).round(2)
    monthly_usage_hours = np.random.uniform(1.0, 120.0, size=num_samples).round(1)
    support_tickets = np.random.poisson(lam=2.2, size=num_samples)

    contract_types = ["Month-to-Month", "One-Year", "Two-Year"]
    contract_type = np.random.choice(contract_types, size=num_samples, p=[0.55, 0.25, 0.20])

    payment_methods = ["Credit Card", "Bank Transfer", "Electronic Check"]
    payment_method = np.random.choice(payment_methods, size=num_samples, p=[0.40, 0.35, 0.25])

    device_categories = ["Mobile", "Desktop", "Tablet"]
    device_category = np.random.choice(device_categories, size=num_samples, p=[0.50, 0.35, 0.15])

    # Log-odds probability model with non-linear interactions & step functions
    logit = (
        -1.5
        + (support_tickets > 3) * 1.8
        + (support_tickets > 6) * 1.5
        + (contract_type == "Month-to-Month") * 1.2
        - (contract_type == "Two-Year") * 1.5
        - (tenure_months / 12.0) * 0.4
        + (age > 50) * 0.6
        - (annual_income / 100000.0) * 0.5
        + (payment_method == "Electronic Check") * 0.7
        - (credit_score > 720) * 0.8
        + (monthly_usage_hours < 15.0) * 1.1
        + np.random.normal(0, 0.6, size=num_samples)
    )

    prob = 1.0 / (1.0 + np.exp(-logit))
    is_churn = (prob > 0.48).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "credit_score": credit_score,
            "tenure_months": tenure_months,
            "account_balance": account_balance,
            "monthly_usage_hours": monthly_usage_hours,
            "support_tickets": support_tickets,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "device_category": device_category,
            "is_churn": is_churn,
        }
    )

    return df


def load_dataset(force_regenerate: bool = False) -> pd.DataFrame:
    """
    Loads dataset from disk if available, otherwise generates and saves it.
    """
    if force_regenerate or not DATA_FILE_PATH.exists():
        df = generate_synthetic_data()
        df.to_csv(DATA_FILE_PATH, index=False)
        print(f"[DataLoader] Generated fresh dataset at {DATA_FILE_PATH} with shape {df.shape}")
    else:
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"[DataLoader] Loaded existing dataset from {DATA_FILE_PATH} with shape {df.shape}")
    return df

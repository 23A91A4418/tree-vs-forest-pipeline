import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "champion_model_name" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "champion_model" in data
    assert "dt_metrics" in data
    assert "rf_metrics" in data


def test_bakeoff_results_endpoint():
    response = client.get("/bakeoff/results")
    assert response.status_code == 200
    data = response.json()
    assert "primary_metric" in data
    assert "champion_model" in data


def test_single_predict_endpoint():
    sample_payload = {
        "age": 42,
        "annual_income": 65000.0,
        "credit_score": 680,
        "tenure_months": 24,
        "account_balance": 12500.0,
        "monthly_usage_hours": 45.5,
        "support_tickets": 2,
        "contract_type": "Month-to-Month",
        "payment_method": "Electronic Check",
        "device_category": "Mobile",
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction" in data
    assert data["prediction"]["is_churn"] in (0, 1)


def test_compare_predict_endpoint():
    sample_payload = {
        "age": 55,
        "annual_income": 45000.0,
        "credit_score": 580,
        "tenure_months": 6,
        "account_balance": 500.0,
        "monthly_usage_hours": 10.0,
        "support_tickets": 6,
        "contract_type": "Month-to-Month",
        "payment_method": "Electronic Check",
        "device_category": "Mobile",
    }
    response = client.post("/predict/compare", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "decision_tree" in data
    assert "random_forest" in data
    assert "agreement" in data


def test_batch_predict_endpoint():
    sample_records = [
        {
            "age": 30,
            "annual_income": 50000.0,
            "credit_score": 700,
            "tenure_months": 12,
            "account_balance": 5000.0,
            "monthly_usage_hours": 30.0,
            "support_tickets": 1,
            "contract_type": "One-Year",
            "payment_method": "Credit Card",
            "device_category": "Desktop",
        },
        {
            "age": 60,
            "annual_income": 30000.0,
            "credit_score": 500,
            "tenure_months": 3,
            "account_balance": 100.0,
            "monthly_usage_hours": 5.0,
            "support_tickets": 8,
            "contract_type": "Month-to-Month",
            "payment_method": "Electronic Check",
            "device_category": "Mobile",
        },
    ]
    response = client.post("/predict/batch", json={"records": sample_records})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2

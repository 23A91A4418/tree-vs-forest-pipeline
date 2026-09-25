from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CustomerFeaturesInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age in years", json_schema_extra={"example": 42})
    annual_income: float = Field(..., ge=0.0, description="Annual income in USD", json_schema_extra={"example": 65000.0})
    credit_score: int = Field(..., ge=300, le=850, description="Credit score", json_schema_extra={"example": 680})
    tenure_months: int = Field(..., ge=1, le=120, description="Account tenure in months", json_schema_extra={"example": 24})
    account_balance: float = Field(..., ge=0.0, description="Account balance in USD", json_schema_extra={"example": 12500.0})
    monthly_usage_hours: float = Field(..., ge=0.0, description="Monthly platform usage hours", json_schema_extra={"example": 45.5})
    support_tickets: int = Field(..., ge=0, description="Support tickets submitted", json_schema_extra={"example": 2})
    contract_type: str = Field(..., description="Contract type (Month-to-Month, One-Year, Two-Year)", json_schema_extra={"example": "Month-to-Month"})
    payment_method: str = Field(..., description="Payment method", json_schema_extra={"example": "Electronic Check"})
    device_category: str = Field(..., description="Primary device category", json_schema_extra={"example": "Mobile"})


class PredictionOutput(BaseModel):
    is_churn: int = Field(..., description="Binary prediction: 1 = Churn, 0 = Retain")
    churn_probability: float = Field(..., description="Predicted probability of churn (0.0 to 1.0)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    model_name: str = Field(..., description="Name of the model used for prediction")


class SinglePredictionResponse(BaseModel):
    success: bool
    prediction: PredictionOutput
    latency_ms: float


class ComparisonPredictionResponse(BaseModel):
    success: bool
    input_summary: Dict[str, Any]
    decision_tree: PredictionOutput
    random_forest: PredictionOutput
    champion: PredictionOutput
    agreement: bool


class BatchPredictionInput(BaseModel):
    records: List[CustomerFeaturesInput]


class BatchPredictionResponse(BaseModel):
    success: bool
    total_records: int
    predictions: List[PredictionOutput]
    churn_count: int
    retention_count: int


class ModelInfoResponse(BaseModel):
    champion_model: str
    champion_run_id: str
    primary_metric: str
    feature_names: List[str]
    dt_metrics: Dict[str, Any]
    rf_metrics: Dict[str, Any]
    timestamp: str


class BakeoffRunResponse(BaseModel):
    success: bool
    message: str
    summary: Dict[str, Any]

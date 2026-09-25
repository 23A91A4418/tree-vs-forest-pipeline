import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import time
from typing import Dict, Any, List
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import (
    BASE_DIR,
    ARTIFACTS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.pipeline import run_bakeoff_pipeline
from src.api.schemas import (
    CustomerFeaturesInput,
    PredictionOutput,
    SinglePredictionResponse,
    ComparisonPredictionResponse,
    BatchPredictionInput,
    BatchPredictionResponse,
    ModelInfoResponse,
    BakeoffRunResponse,
)

app = FastAPI(
    title="ML Pipeline: Decision Tree vs. Random Forest Bake-Off",
    description="Production Machine Learning Serving API & Experiment Tracking Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded artifacts
preprocessor = None
champion_model = None
dt_model = None
rf_model = None
bakeoff_summary = {}


def load_artifacts():
    global preprocessor, champion_model, dt_model, rf_model, bakeoff_summary
    
    preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"
    champion_path = ARTIFACTS_DIR / "champion_model.joblib"
    dt_path = ARTIFACTS_DIR / "decision_tree_model.joblib"
    rf_path = ARTIFACTS_DIR / "random_forest_model.joblib"
    summary_path = RESULTS_DIR / "metrics.json"

    if not summary_path.exists():
        summary_path = ARTIFACTS_DIR / "latest_bakeoff_summary.json"

    if not (preprocessor_path.exists() and champion_path.exists() and summary_path.exists()):
        print("[FastAPI] Artifacts not found. Running bake-off pipeline on startup...")
        run_bakeoff_pipeline()

    preprocessor = joblib.load(preprocessor_path)
    champion_model = joblib.load(champion_path)
    if dt_path.exists():
        dt_model = joblib.load(dt_path)
    if rf_path.exists():
        rf_model = joblib.load(rf_path)

    if summary_path.exists():
        with open(summary_path, "r") as f:
            bakeoff_summary = json.load(f)
    print("[FastAPI] All ML artifacts loaded successfully.")


# Load artifacts upon module import so TestClient & FastAPI load immediately
load_artifacts()

# Mount Static directory & Plots directory
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")


def predict_single_model(model: Any, input_df: pd.DataFrame, model_name: str) -> PredictionOutput:
    X_transformed = preprocessor.transform(input_df)
    pred_class = int(model.predict(X_transformed)[0])
    
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_transformed)[0][1])
    else:
        prob = float(pred_class)
        
    prob = round(prob, 4)
    label = "High Churn Risk" if pred_class == 1 else "Low Churn Risk (Retained)"

    return PredictionOutput(
        is_churn=pred_class,
        churn_probability=prob,
        prediction_label=label,
        model_name=model_name,
    )


@app.get("/")
def read_root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "ML Pipeline API is running. Access /docs for API documentation."}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "artifacts_loaded": preprocessor is not None and champion_model is not None,
        "champion_model_name": bakeoff_summary.get("champion_model", "Unknown"),
        "timestamp": pd.Timestamp.now().isoformat(),
    }


@app.post("/predict", response_model=SinglePredictionResponse)
def predict_champion(data: CustomerFeaturesInput):
    if preprocessor is None or champion_model is None:
        raise HTTPException(status_code=500, detail="ML artifacts not loaded.")

    start_time = time.time()
    input_df = pd.DataFrame([data.model_dump()])
    champion_name = bakeoff_summary.get("champion_model", "Champion Model")
    
    prediction = predict_single_model(champion_model, input_df, model_name=champion_name)
    latency_ms = round((time.time() - start_time) * 1000, 3)

    return SinglePredictionResponse(
        success=True,
        prediction=prediction,
        latency_ms=latency_ms,
    )


@app.post("/predict/compare", response_model=ComparisonPredictionResponse)
def predict_compare(data: CustomerFeaturesInput):
    if preprocessor is None or dt_model is None or rf_model is None:
        raise HTTPException(status_code=500, detail="Decision Tree or Random Forest model not loaded.")

    input_df = pd.DataFrame([data.model_dump()])
    dt_pred = predict_single_model(dt_model, input_df, model_name="Decision Tree")
    rf_pred = predict_single_model(rf_model, input_df, model_name="Random Forest")
    
    champ_name = bakeoff_summary.get("champion_model", "Random Forest")
    champ_pred = rf_pred if champ_name == "Random Forest" else dt_pred

    return ComparisonPredictionResponse(
        success=True,
        input_summary=data.model_dump(),
        decision_tree=dt_pred,
        random_forest=rf_pred,
        champion=champ_pred,
        agreement=dt_pred.is_churn == rf_pred.is_churn,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch_data: BatchPredictionInput):
    if preprocessor is None or champion_model is None:
        raise HTTPException(status_code=500, detail="ML artifacts not loaded.")

    records = [record.model_dump() for record in batch_data.records]
    input_df = pd.DataFrame(records)
    
    X_transformed = preprocessor.transform(input_df)
    preds = champion_model.predict(X_transformed)
    
    probs = (
        champion_model.predict_proba(X_transformed)[:, 1]
        if hasattr(champion_model, "predict_proba")
        else preds
    )

    champion_name = bakeoff_summary.get("champion_model", "Champion Model")
    prediction_outputs = []
    churn_count = 0
    retention_count = 0

    for pred, prob in zip(preds, probs):
        p_class = int(pred)
        p_prob = round(float(prob), 4)
        if p_class == 1:
            churn_count += 1
            lbl = "High Churn Risk"
        else:
            retention_count += 1
            lbl = "Low Churn Risk (Retained)"

        prediction_outputs.append(
            PredictionOutput(
                is_churn=p_class,
                churn_probability=p_prob,
                prediction_label=lbl,
                model_name=champion_name,
            )
        )

    return BatchPredictionResponse(
        success=True,
        total_records=len(records),
        predictions=prediction_outputs,
        churn_count=churn_count,
        retention_count=retention_count,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    if not bakeoff_summary:
        raise HTTPException(status_code=404, detail="Bake-off summary not available.")

    models_meta = bakeoff_summary.get("models", {})
    dt_meta = models_meta.get("decision_tree", {})
    rf_meta = models_meta.get("random_forest", {})

    return ModelInfoResponse(
        champion_model=bakeoff_summary.get("champion_model", "Unknown"),
        champion_run_id=bakeoff_summary.get("champion_run_id", "Unknown"),
        primary_metric=bakeoff_summary.get("primary_metric", "f1_score"),
        feature_names=bakeoff_summary.get("feature_names", []),
        dt_metrics=dt_meta.get("metrics", {}),
        rf_metrics=rf_meta.get("metrics", {}),
        timestamp=bakeoff_summary.get("timestamp", ""),
    )


@app.get("/bakeoff/results")
def get_bakeoff_results():
    if not bakeoff_summary:
        raise HTTPException(status_code=404, detail="Bake-off results not found.")
    return bakeoff_summary


@app.post("/bakeoff/run", response_model=BakeoffRunResponse)
def trigger_bakeoff(background_tasks: BackgroundTasks):
    try:
        summary = run_bakeoff_pipeline()
        load_artifacts()
        return BakeoffRunResponse(
            success=True,
            message="Machine Learning Bake-Off pipeline executed successfully and champion model updated.",
            summary=summary,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bake-off pipeline execution failed: {str(e)}")

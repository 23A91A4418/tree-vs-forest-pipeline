# Production ML Pipeline: Decision Tree vs. Random Forest Bake-Off


An end-to-end production Machine Learning pipeline and serving solution comparing **Decision Tree** vs. **Random Forest** classification models on tabular customer churn data. Experiments, metrics, hyperparameter grids, and visual artifacts are automatically tracked using **MLflow**, and the winning model is automatically selected and served via a **FastAPI** REST engine with an interactive dark glassmorphic web dashboard.

---

##  Mandatory Submission Compliance

- **`src/` Package**: All Python source code is organized inside `src/`.
- **`Dockerfile.api`**: FastAPI application container definition.
- **`Dockerfile`**: ML pipeline training execution container definition.
- **`docker-compose.yml`**: Orchestrates 3 services:
  1. `mlflow_server`: Tracking server on port 5000.
  2. `pipeline`: Automated training pipeline execution service.
  3. `api_server`: FastAPI inference API on port 8000.
- **`.env.example`**: Documents required environment variables (`MLFLOW_TRACKING_URI`, `EVAL_METRIC`, `API_HOST`, `API_PORT`).
- **`results/` Artifact Outputs**: `src/pipeline.py` outputs `results/metrics.json` and `results/roc_comparison.png`.

---

## Repository Structure

```
tree-vs-forest-pipeline/
├── .env.example            # Environment variable documentation
├── .gitignore              # Git ignore rules
├── Dockerfile              # Pipeline service container definition
├── Dockerfile.api          # FastAPI API service container definition
├── docker-compose.yml      # Orchestrates MLflow, Pipeline & FastAPI services
├── requirements.txt        # Python dependency manifest
├── pytest.ini              # Pytest configuration
├── conftest.py             # Pytest root import path configuration
├── README.md               # Production documentation
├── src/                    # Python source code package
│   ├── __init__.py
│   ├── config.py           # Paths, feature definitions, hyperparameter grids
│   ├── data_loader.py      # Tabular customer dataset generator
│   ├── preprocessing.py   # Scikit-Learn ColumnTransformer pipeline
│   ├── models.py          # Decision Tree & Random Forest GridSearch trainers
│   ├── evaluate.py        # Classification metric calculator & champion selector
│   ├── visualize.py       # Plot generators (ROC curves, confusion matrices, etc.)
│   ├── pipeline.py        # MLflow bake-off execution & results output generator
│   └── api/
│       ├── __init__.py
│       ├── main.py        # FastAPI endpoints & artifact lifecycle
│       ├── schemas.py     # Pydantic request & response DTOs
│       └── static/
│           ├── index.html # Dark mode Web UI dashboard
│           ├── css/
│           │   └── style.css
│           └── js/
│               └── app.js
├── tests/
│   ├── test_pipeline.py   # Unit tests for ML engine & data processing
│   └── test_api.py        # Integration tests for FastAPI endpoints
├── data/                  # Customer churn CSV dataset
├── artifacts/             # Joblib model binaries & summary JSON
├── results/               # Mandatory output results (metrics.json & roc_comparison.png)
└── plots/                 # Visual plots directory
```

---

## Quick Start (Local Setup)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML Bake-Off Pipeline

To train both models, execute grid search, log runs to MLflow, and generate `results/metrics.json` & `results/roc_comparison.png`:

```bash
python -m src.pipeline
```

### 3. Start the FastAPI Production Server

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Web UI Dashboard**: Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive OpenAPI Documentation**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Docker Deployment

To launch all 3 services (**MLflow Tracking UI**, **Training Pipeline**, **FastAPI Server**) together using Docker Compose:

```bash
docker-compose up --build
```

- **FastAPI Web App & UI**: [http://localhost:8000](http://localhost:8000)
- **MLflow Tracking Server UI**: [http://localhost:5000](http://localhost:5000)

---

## Automated Testing

Run the automated Pytest test suite:

```bash
pytest -v tests/
```

All 11 tests check:
- Data generation & schema validation
- Scikit-Learn preprocessing transformers
- Decision Tree & Random Forest GridSearch training
- Metric evaluation & Champion model selection
- FastAPI endpoint responses (`/health`, `/predict`, `/predict/compare`, `/predict/batch`, `/model/info`, `/bakeoff/results`)
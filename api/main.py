import json
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing import DROP_COLUMNS, TARGET


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "churn_model.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"


# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained Machine Learning pipeline.",
    version="1.0.0"
)


# Load model and threshold
model = None
threshold = None

if MODEL_PATH.exists() and THRESHOLD_PATH.exists():
    model = joblib.load(MODEL_PATH)

    with THRESHOLD_PATH.open("r", encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Model threshold must be between 0 and 1.")


# Request schema
class PredictionRequest(BaseModel):
    customers: list[dict] = Field(min_length=1)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running"
    }


# Health endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "model_not_loaded"
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    if model is None or threshold is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first."
        )

    try:
        customers = pd.DataFrame(request.customers)

        # Remove target and excluded columns
        features = customers.drop(
            columns=[TARGET] + DROP_COLUMNS,
            errors="ignore"
        )

        # Validate expected features
        expected_features = list(model.feature_names_in_)

        missing_features = sorted(
            set(expected_features) - set(features.columns)
        )

        if missing_features:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required features: {missing_features}"
            )

        features = features.loc[:, expected_features]

        # Generate predictions
        probabilities = model.predict_proba(features)[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results = []

        for probability, prediction in zip(
            probabilities,
            predictions
        ):
            results.append({
                "churn_probability": round(float(probability), 4),
                "predicted_churn": int(prediction)
            })

        return {
            "model": "Customer Churn Prediction",
            "threshold": threshold,
            "predictions": results
        }

    except HTTPException:
        raise

    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid customer data: {error}"
        )
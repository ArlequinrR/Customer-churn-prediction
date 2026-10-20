"""Generate customer churn predictions using a previously trained pipeline.

Run from the project root:
    python -m src.predict --input data/new_customers.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import DROP_COLUMNS, TARGET


def predict_customers(input_path: str, output_path: str, model_dir: str) -> None:
    """Score new customer records and export their churn probabilities."""
    model_directory = Path(model_dir)
    model_path = model_directory / "churn_model.joblib"
    threshold_path = model_directory / "threshold.json"

    if not model_path.is_file() or not threshold_path.is_file():
        raise FileNotFoundError(
            "Missing trained model or threshold. Run 'python -m src.train' first."
        )

    # Only load model artifacts created by you or another trusted source.
    model = joblib.load(model_path)
    with threshold_path.open("r", encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("The saved threshold must be between 0 and 1.")

    # Read incoming records using the same delimiter as the training dataset.
    customers = pd.read_csv(input_path, sep=";")
    if customers.empty:
        raise ValueError("The input CSV contains no customer records.")

    # The target must not be included among the prediction features.
    features = customers.drop(columns=[TARGET] + DROP_COLUMNS, errors="ignore")

    # Validate the feature schema before scoring.
    expected_features = list(model.feature_names_in_)
    missing_features = sorted(set(expected_features) - set(features.columns))
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")
    features = features.loc[:, expected_features]

    # The saved pipeline automatically applies the training-time preprocessing.
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = customers.copy()
    results["churn_probability"] = probabilities.round(4)
    results["predicted_churn"] = predictions

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False)
    print(f"Scored {len(results)} customers using threshold {threshold:.2f}.")
    print(f"Predictions saved to: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict customer churn from a CSV file.")
    parser.add_argument("--input", required=True, help="Input CSV with customer features (semicolon-delimited).")
    parser.add_argument("--output", default="outputs/churn_predictions.csv", help="Output CSV path.")
    parser.add_argument("--model-dir", default="models", help="Directory with saved model and threshold.")
    args = parser.parse_args()
    predict_customers(args.input, args.output, args.model_dir)


if __name__ == "__main__":
    main()

"""Train XGBoost, tune the threshold on training data, and save artifacts."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.preprocessing import prepare_features, build_preprocessor
from src.evaluate import evaluate

def main(data_path):
    df = pd.read_csv(data_path, sep=";")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", XGBClassifier(n_estimators=300, max_depth=3,
            learning_rate=0.05, n_jobs=-1, scale_pos_weight=5.9, random_state=42)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.arange(0.05, 0.96, 0.01)
    threshold = float(max(thresholds, key=lambda t: f1_score(y_train, oof >= t, zero_division=0)))
    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test, threshold)
    output = Path("models")
    output.mkdir(exist_ok=True)
    joblib.dump(model, output / "churn_model.joblib")
    (output / "threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2))
    print("Model and threshold saved in models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn-analysis.csv")
    args = parser.parse_args()
    main(args.data)

# Customer Churn Prediction | Machine Learning Portfolio Project

## Business Problem

Customer churn occurs when customers stop using a company's services. For subscription-based businesses, losing customers can reduce recurring revenue and increase the need for costly customer acquisition. Identifying customers at risk of leaving can help retention teams prioritize outreach and design targeted interventions.

## Project Objective

Build and evaluate a binary classification model that estimates customer churn risk using historical customer records. The intended application is **decision support** for customer retention—not an automatically deployed retention system.

## Dataset

The project uses the `churn-analysis.csv` dataset from the original analysis. The target variable is `churn` (1 = churn, 0 = no churn). The original notebook removes `area code` and `phone number` before modeling. The CSV is **not included** in this repository; place a legally shareable copy at `data/churn-analysis.csv`. The file is expected to use a semicolon (`;`) delimiter.

## Methodology

1. **Exploratory Data Analysis (EDA):** inspect feature distributions, churn frequencies, and numerical correlations.
2. **Data preparation:** drop selected identifier-like fields, perform an 80/20 stratified train/test split, and build a scikit-learn `ColumnTransformer` with numeric imputation/scaling and categorical imputation/one-hot encoding.
3. **Model comparison:** evaluate Logistic Regression, Random Forest, and XGBoost using five-fold stratified cross-validation on the training set, scored by ROC-AUC.
4. **Threshold optimization:** obtain out-of-fold probabilities from XGBoost and select a classification threshold maximizing F1 on the training data.
5. **Final evaluation:** fit the selected model on the training split and report test ROC-AUC, precision, recall, F1, and the confusion matrix.

## Results from the Original Notebook

| Model | Mean cross-validation ROC-AUC |
| --- | ---: |
| Logistic Regression | 0.8124 |
| Random Forest | 0.9046 |
| **XGBoost** | **0.9147** |

The original notebook reports the following XGBoost holdout results:

| Metric | Reported value |
| --- | ---: |
| ROC-AUC | 0.8706 |
| Accuracy | 0.92 |
| Precision (churn class) | 0.71 |
| Recall (churn class) | 0.75 |
| F1 (churn class) | 0.73 |
| Selected decision threshold | approximately 0.60 |

These are **historical notebook results**, not independently reproduced results from this packaged repository. Exact values may vary with the dataset and library versions.

## Business Interpretation

A churn recall of approximately 75% indicates that the model identified around three out of four customers who churned in the evaluated test sample. Such scores could be used to rank customers for retention review. Actual financial impact would require additional information, including outreach cost, customer value, intervention success rates, and operational constraints. No financial savings or production outcomes are claimed.

## Repository Structure

customer-churn-prediction/
├── api/
│   ├── __init__.py
│   └── main.py
├── data/
│   └── README.md
├── notebooks/
│   └── 01_customer_churn_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── .gitkeep
├── outputs/
├── create_sample.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE

## Installation

Run the following commands from the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Put the original semicolon-delimited dataset at `data/churn-analysis.csv`.
2. Train and evaluate the XGBoost model:

```bash
python -m src.train
```

Or provide a custom CSV location:

```bash
python -m src.train --data path/to/churn-analysis.csv
```

The script saves the fitted pipeline to `models/churn_model.joblib` and the selected threshold to `models/threshold.json`. Both files are excluded from version control. To explore the full three-model comparison, open `notebooks/01_customer_churn_analysis.ipynb` in Jupyter and run it from the repository root.

## Limitations

- During earlier development, test metrics from multiple experiments were inspected; therefore, the reported holdout results should **not** be presented as a fully untouched, unbiased final benchmark.
- The modular training script trains XGBoost only; the three-model comparison remains in the notebook.
- The training and prediction workflow has been tested locally. Batch predictions and FastAPI predictions were verified to produce matching results for the same five      sample customers.
- Model performance may shift across populations and time; no external or production validation has been performed.
- The threshold is optimized for F1, not expected business profit.

## Batch Predictions

The trained model can generate churn predictions for new customers
without requiring an API.

Run:

```bash
python -m src.predict --input data/new_customers.csv
```

The predictions are saved to:

`outputs/churn_predictions.csv`

The output includes:

- `churn_probability`: estimated probability of customer churn.
- `predicted_churn`: binary prediction based on the selected threshold.

## FastAPI — Model Inference

This project includes a REST API built with FastAPI to make
predictions available to other applications.

The API loads the previously trained XGBoost pipeline and
classification threshold.

### Start the API

First, install the dependencies and train the model using
the instructions above.

Then run:

```bash
python -m uvicorn api.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### Available Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Basic API information |
| GET | `/health` | Check API and model status |
| POST | `/predict` | Generate customer churn predictions |

### Prediction Request

The `/predict` endpoint accepts a JSON object containing
a list of customers:

```json
{
  "customers": [
    {
      "feature_1": "value",
      "feature_2": 10
    }
  ]
}
```

The fields above illustrate the request structure.
Actual requests must include the features expected by
the trained model.

### Prediction Response

Example response:

```json
{
  "model": "Customer Churn Prediction",
  "threshold": 0.6,
  "predictions": [
    {
      "churn_probability": 0.1756,
      "predicted_churn": 0
    }
  ]
}
```

### Deployment Status

The API has been tested locally using HTTP requests.

It is not deployed to a public server.

For periodic batch predictions, the Python prediction
script can be used without FastAPI.





## Next Steps

- Evaluate on a new, genuinely untouched dataset.
- Add automated tests, data validation, and reproducible dependency locking.
- Evaluate business-cost-sensitive thresholds and probability calibration.
- Explore feature importance and model monitoring before any deployment.

## License

Source code is licensed under the MIT License. The dataset has separate ownership and usage terms.

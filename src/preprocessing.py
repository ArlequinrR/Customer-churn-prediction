"""Feature preparation based on the original analysis notebook."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLUMNS = ["area code", "phone number"]
TARGET = "churn"

def prepare_features(df):
    """Return features and target; churn must be boolean or encoded as 0/1."""
    if TARGET not in df.columns:
        raise ValueError("Missing target column 'churn'.")
    y = df[TARGET].replace({True: 1, False: 0}).astype(int)
    X = df.drop(columns=[TARGET] + DROP_COLUMNS, errors="ignore")
    return X, y

def build_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include="object").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

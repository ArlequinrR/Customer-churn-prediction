"""Binary classification evaluation metrics."""
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate(model, X_test, y_test, threshold):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    print(f"Threshold: {threshold:.2f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

import numpy as np
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

print("=" * 70)
print("LOADING PROCESSED TON_IOT DATA")
print("=" * 70)

X_train = load_npz("X_train_processed.npz")
X_test = load_npz("X_test_processed.npz")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, model_name):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Probability for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None

    print("\nPerformance:")
    print("-" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0
    ))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    }


# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_results = evaluate_model(
    logistic_model,
    "LOGISTIC REGRESSION"
)


# ============================================================
# MODEL 2: RANDOM FOREST
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_results = evaluate_model(
    random_forest_model,
    "RANDOM FOREST"
)


# ============================================================
# COMPARE MODELS
# ============================================================

results = [
    logistic_results,
    random_forest_results
]

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for result in results:

    print("\n", result["Model"])
    print(f"Accuracy : {result['Accuracy']:.4f}")
    print(f"Precision: {result['Precision']:.4f}")
    print(f"Recall   : {result['Recall']:.4f}")
    print(f"F1-score : {result['F1']:.4f}")

    if result["ROC_AUC"] is not None:
        print(f"ROC-AUC  : {result['ROC_AUC']:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

import pandas as pd

results_df = pd.DataFrame(results)

results_df.to_csv(
    "baseline_model_results.csv",
    index=False
)

print("\nResults saved as: baseline_model_results.csv")

print("\n" + "=" * 70)
print("BASELINE MODELING COMPLETE")
print("=" * 70)
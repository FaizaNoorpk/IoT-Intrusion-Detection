import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

print("=" * 75)
print("TON_IOT SOURCE-IP ROBUSTNESS EXPERIMENT")
print("=" * 75)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("ton_iot_cleaned.csv")

print("\nDataset:", df.shape)
print("Unique source IPs:", df["src_ip"].nunique())

# ============================================================
# 2. CREATE DNS STRUCTURAL FEATURES
# ============================================================

def query_length(x):
    if x == "-":
        return 0
    return len(str(x))


def query_num_labels(x):
    if x == "-":
        return 0
    return len(str(x).split("."))


def query_digit_ratio(x):
    if x == "-":
        return 0.0

    x = str(x)
    return sum(c.isdigit() for c in x) / max(len(x), 1)


def query_entropy(x):
    if x == "-":
        return 0.0

    x = str(x)

    counts = pd.Series(list(x)).value_counts()
    probabilities = counts / counts.sum()

    return -(probabilities * np.log2(probabilities)).sum()


df["dns_present"] = (df["dns_query"] != "-").astype(int)
df["dns_query_length"] = df["dns_query"].apply(query_length)
df["dns_query_num_labels"] = df["dns_query"].apply(query_num_labels)
df["dns_query_digit_ratio"] = df["dns_query"].apply(query_digit_ratio)
df["dns_query_entropy"] = df["dns_query"].apply(query_entropy)

# ============================================================
# 3. FEATURES
# ============================================================

numeric_features = [
    "src_port",
    "dst_port",
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_present",
    "dns_query_length",
    "dns_query_num_labels",
    "dns_query_digit_ratio",
    "dns_query_entropy",
    "http_request_body_len",
    "http_response_body_len"
]

categorical_features = [
    "proto",
    "service",
    "conn_state",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "http_status_code"
]

features = numeric_features + categorical_features

X = df[features]
y = df["label"]
groups = df["src_ip"]

# ============================================================
# 4. SOURCE-IP GROUPED SPLIT
# ============================================================

print("\nCreating source-IP grouped split...")

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

print("\nTrain samples:", len(X_train))
print("Test samples :", len(X_test))

print("\nTrain source IPs:", groups_train.nunique())
print("Test source IPs :", groups_test.nunique())

overlap = set(groups_train.unique()) & set(groups_test.unique())

print("Source IP overlap:", len(overlap))

print("\nTest source IPs:")
for ip in sorted(groups_test.unique()):
    print(" ", ip)

# ============================================================
# 5. LABEL DISTRIBUTION
# ============================================================

print("\nTraining label distribution:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

print("\nTesting label distribution:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True))

# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ============================================================
# 7. RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])

pipeline.fit(X_train, y_train)

# ============================================================
# 8. EVALUATION
# ============================================================

print("\nEvaluating...")

y_pred = pipeline.predict(X_test)
y_probability = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 75)
print("SOURCE-IP ROBUSTNESS RESULTS")
print("=" * 75)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Attack"],
    zero_division=0
))

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{
    "Experiment": "Source-IP Grouped Split",
    "Train_Samples": len(X_train),
    "Test_Samples": len(X_test),
    "Train_Source_IPs": groups_train.nunique(),
    "Test_Source_IPs": groups_test.nunique(),
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC_AUC": roc_auc
}])

results.to_csv(
    "source_ip_robustness_results.csv",
    index=False
)

print("\nSaved:")
print("source_ip_robustness_results.csv")

print("\nExperiment completed.")
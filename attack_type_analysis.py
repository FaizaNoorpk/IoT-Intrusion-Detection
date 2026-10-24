import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 80)
print("TON_IOT ATTACK-TYPE ANALYSIS ON UNSEEN SOURCE IPS")
print("=" * 80)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("ton_iot_cleaned.csv")

print("\nDataset shape:", df.shape)
print("Unique source IPs:", df["src_ip"].nunique())

# ============================================================
# 2. DNS FEATURE ENGINEERING
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

print("\nDNS features created.")

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

print("\nCreating unseen-source-IP split...")

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

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("Training IPs:", df_train["src_ip"].nunique())
print("Testing IPs :", df_test["src_ip"].nunique())

print(
    "IP overlap:",
    len(
        set(df_train["src_ip"].unique())
        &
        set(df_test["src_ip"].unique())
    )
)

# ============================================================
# 5. PREPROCESSING
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
# 6. RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])

model.fit(X_train, y_train)

# ============================================================
# 7. BINARY PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

# ============================================================
# 8. OVERALL BINARY PERFORMANCE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\n" + "=" * 80)
print("OVERALL UNSEEN-SOURCE PERFORMANCE")
print("=" * 80)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# ============================================================
# 9. ATTACK-TYPE PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("ATTACK-TYPE PERFORMANCE")
print("=" * 80)

# Attach predictions to test dataframe
df_test["prediction"] = y_pred

# Only attack samples
attack_test = df_test[df_test["type"] != "normal"].copy()

attack_types = sorted(
    attack_test["type"].unique()
)

rows = []

for attack_type in attack_types:

    subset = attack_test[
        attack_test["type"] == attack_type
    ]

    # All are actual attacks, therefore:
    # recall = correctly identified attacks / total attacks
    detected = (subset["prediction"] == 1).sum()
    total = len(subset)

    attack_recall = detected / total if total > 0 else 0

    rows.append({
        "Attack_Type": attack_type,
        "Samples": total,
        "Detected": detected,
        "Missed": total - detected,
        "Recall": attack_recall
    })

attack_results = pd.DataFrame(rows)

attack_results = attack_results.sort_values(
    "Recall"
)

print("\nPer-attack recall:")
print(
    attack_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# 10. ATTACK-TYPE FALSE POSITIVE INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("PREDICTED ATTACKS BY ACTUAL TYPE")
print("=" * 80)

prediction_by_type = (
    df_test
    .groupby("type")["prediction"]
    .agg(
        total_samples="count",
        predicted_attack="sum"
    )
    .reset_index()
)

prediction_by_type["attack_detection_rate"] = (
    prediction_by_type["predicted_attack"]
    /
    prediction_by_type["total_samples"]
)

print(
    prediction_by_type.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# 11. MULTI-CLASS ATTACK TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("MULTI-CLASS ATTACK-TYPE ANALYSIS")
print("=" * 80)

# For this experiment, create a multiclass target:
# normal + individual attack categories

target_classes = sorted(df_test["type"].unique())

print("\nClasses:")
for i, name in enumerate(target_classes):
    print(i, "=", name)

# Train a second RF directly for attack-type classification
# using exactly the same unseen-source split.

y_type_train = df_train["type"]
y_type_test = df_test["type"]

multi_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

multi_model = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])),
    ("classifier", multi_rf)
])

print("\nTraining multiclass Random Forest...")

multi_model.fit(
    X_train,
    y_type_train
)

type_pred = multi_model.predict(X_test)

print("\nMulticlass classification report:")

print(
    classification_report(
        y_type_test,
        type_pred,
        zero_division=0
    )
)

# ============================================================
# 12. MULTICLASS CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_type_test,
    type_pred,
    labels=target_classes
)

cm_df = pd.DataFrame(
    cm,
    index=target_classes,
    columns=target_classes
)

print("\nMulticlass confusion matrix:")
print(cm_df)

# ============================================================
# 13. SAVE RESULTS
# ============================================================

attack_results.to_csv(
    "attack_type_recall_results.csv",
    index=False
)

prediction_by_type.to_csv(
    "attack_type_prediction_summary.csv",
    index=False
)

cm_df.to_csv(
    "attack_type_confusion_matrix.csv"
)

overall_results = pd.DataFrame([{
    "Experiment": "Unseen Source IP",
    "Model": "Random Forest",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
}])

overall_results.to_csv(
    "final_robustness_results.csv",
    index=False
)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print("attack_type_recall_results.csv")
print("attack_type_prediction_summary.csv")
print("attack_type_confusion_matrix.csv")
print("final_robustness_results.csv")

print("\nAttack-type analysis completed.")
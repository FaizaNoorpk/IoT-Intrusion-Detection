import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Add this line
import seaborn as sns 

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

print("=" * 80)
print("TON_IOT BALANCED SOURCE-AWARE EVALUATION")
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
# 4. FIND SOURCE IPS WITH MULTIPLE ATTACK TYPES
# ============================================================

print("\nAnalyzing source IP / attack-type coverage...")

source_attack_table = (
    df[df["label"] == 1]
    .groupby("src_ip")["type"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nAttack-type diversity by source IP:")
print(source_attack_table.head(20))

# ============================================================
# 5. CANDIDATE SOURCE IP GROUPED SPLITS
# ============================================================

print("\nSearching for a source-aware split with good attack coverage...")

best_split = None
best_score = -1

gss = GroupShuffleSplit(
    n_splits=100,
    test_size=0.20,
    random_state=42
)

for train_idx, test_idx in gss.split(X, y, groups=groups):

    test_df = df.iloc[test_idx]

    test_attack = test_df[test_df["label"] == 1]

    if len(test_attack) == 0:
        continue

    attack_types = test_attack["type"].nunique()

    attack_samples = len(test_attack)

    normal_samples = (
        test_df["label"] == 0
    ).sum()

    # Prefer:
    # 1. more attack types
    # 2. more attack samples
    # 3. reasonable normal/attack representation

    attack_ratio = attack_samples / len(test_df)

    score = (
        attack_types * 10000
        + min(attack_samples, 5000)
        + min(attack_ratio * 1000, 500)
    )

    if score > best_score:
        best_score = score
        best_split = (train_idx, test_idx)

if best_split is None:
    raise RuntimeError(
        "Could not find a suitable source-aware split."
    )

train_idx, test_idx = best_split

# ============================================================
# 6. CREATE FINAL SPLIT
# ============================================================

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

train_ips = set(df_train["src_ip"].unique())
test_ips = set(df_test["src_ip"].unique())

overlap = train_ips & test_ips

print("\n" + "=" * 80)
print("SELECTED SOURCE-AWARE SPLIT")
print("=" * 80)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("Training IPs:", len(train_ips))
print("Testing IPs :", len(test_ips))

print("Source IP overlap:", len(overlap))

print("\nTest source IPs:")
for ip in sorted(test_ips):
    print(" ", ip)

# ============================================================
# 7. TEST DISTRIBUTION
# ============================================================

print("\nTest label distribution:")
print(
    df_test["label"]
    .value_counts()
)

print("\nTest label percentages:")
print(
    df_test["label"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nAttack types in test set:")
print(
    df_test[df_test["label"] == 1]["type"]
    .value_counts()
)

# ============================================================
# 8. PREPROCESSING
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
# 9. RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
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
# 10. PREDICTION
# ============================================================

print("\nEvaluating...")

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# ============================================================
# 11. METRICS
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
roc_auc = roc_auc_score(
    y_test, y_probability
)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 80)
print("BALANCED SOURCE-AWARE RESULTS")
print("=" * 80)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)

# ============================================================
# NEW CODE: GENERATE GRAPHS
# ============================================================
from sklearn.metrics import roc_curve, auc

print("\nGenerating evaluation plots...")

# 1. Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Attack'], 
            yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300) # Saves the file to your folder
plt.show()

# 2. Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probability)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300) # Saves the file to your folder
plt.show()


# ============================================================
# 12. PER-ATTACK-TYPE RECALL
# ============================================================

df_test["prediction"] = y_pred

attack_test = df_test[
    df_test["label"] == 1
].copy()

attack_results = []

for attack_type in sorted(
    attack_test["type"].unique()
):

    subset = attack_test[
        attack_test["type"] == attack_type
    ]

    total = len(subset)

    detected = (
        subset["prediction"] == 1
    ).sum()

    missed = total - detected

    recall_type = (
        detected / total
        if total > 0
        else 0
    )

    attack_results.append({
        "Attack_Type": attack_type,
        "Samples": total,
        "Detected": detected,
        "Missed": missed,
        "Recall": recall_type
    })

attack_results = pd.DataFrame(
    attack_results
).sort_values("Recall")

print("\n" + "=" * 80)
print("PER-ATTACK-TYPE RECALL")
print("=" * 80)

print(
    attack_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# 13. SAVE RESULTS
# ============================================================

final_results = pd.DataFrame([{
    "Experiment": "Balanced Source-Aware Evaluation",
    "Train_Samples": len(X_train),
    "Test_Samples": len(X_test),
    "Train_IPs": len(train_ips),
    "Test_IPs": len(test_ips),
    "IP_Overlap": len(overlap),
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC_AUC": roc_auc
}])

final_results.to_csv(
    "balanced_source_results.csv",
    index=False
)

attack_results.to_csv(
    "balanced_source_attack_results.csv",
    index=False
)

pd.DataFrame(cm).to_csv(
    "balanced_source_confusion_matrix.csv",
    index=False
)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print("balanced_source_results.csv")
print("balanced_source_attack_results.csv")
print("balanced_source_confusion_matrix.csv")

print("\nBalanced source-aware evaluation completed.")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

df = pd.read_csv("ton_iot_cleaned.csv")

print("=" * 70)
print("TON_IOT DATA PREPARATION")
print("=" * 70)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. CREATE DNS STRUCTURAL FEATURES
# ============================================================

def query_length(x):
    if x == "-":
        return 0
    return len(x)


def query_num_labels(x):
    if x == "-":
        return 0
    return len(x.split("."))


def query_digit_ratio(x):
    if x == "-":
        return 0

    return sum(c.isdigit() for c in x) / max(len(x), 1)


def query_entropy(x):
    if x == "-":
        return 0

    counts = pd.Series(list(x)).value_counts()
    probabilities = counts / counts.sum()

    return -(probabilities * np.log2(probabilities)).sum()


df["dns_present"] = (df["dns_query"] != "-").astype(int)

df["dns_query_length"] = df["dns_query"].apply(query_length)

df["dns_query_num_labels"] = df["dns_query"].apply(query_num_labels)

df["dns_query_digit_ratio"] = df["dns_query"].apply(query_digit_ratio)

df["dns_query_entropy"] = df["dns_query"].apply(query_entropy)


# ============================================================
# 3. DEFINE FEATURES
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


# ============================================================
# 4. SELECT X AND y
# ============================================================

X = df[numeric_features + categorical_features].copy()

y = df["label"].copy()


print("\nNumber of features before encoding:", X.shape[1])

print("\nNumeric features:")
print(len(numeric_features))

print("\nCategorical features:")
print(len(categorical_features))


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================
#
# IMPORTANT:
# The original dataset is ordered by attack type.
#
# Therefore, we use a randomized stratified split.
#
# We stratify using attack TYPE rather than only binary LABEL
# so every attack category is represented in both sets.
# TYPE is NOT used as a model feature.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=df["type"]
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting label distribution:")
print(y_test.value_counts(normalize=True) * 100)


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    sparse_threshold=1.0
)


# ============================================================
# 7. FIT ONLY ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 8. CHECK RESULT
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING RESULT")
print("=" * 70)

print("\nOriginal training shape:")
print(X_train.shape)

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nOriginal testing shape:")
print(X_test.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)


# ============================================================
# 9. SAVE DATA
# ============================================================

from scipy.sparse import save_npz

save_npz(
    "X_train_processed.npz",
    X_train_processed
)

save_npz(
    "X_test_processed.npz",
    X_test_processed
)

np.save(
    "y_train.npy",
    y_train.to_numpy()
)

np.save(
    "y_test.npy",
    y_test.to_numpy()
)


print("\nProcessed datasets saved successfully.")

print("\nFiles created:")
print("X_train_processed.npz")
print("X_test_processed.npz")
print("y_train.npy")
print("y_test.npy")

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETE")
print("=" * 70)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("ton_iot_cleaned.csv")


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


X = df[numeric_features + categorical_features]
y = df["label"]


# ============================================================
# 4. SAME TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=df["type"]
)


# ============================================================
# 5. SAME PREPROCESSING
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
# 6. FIT ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)


# ============================================================
# 7. GET REAL FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()


print("=" * 70)
print("ENCODED FEATURE NAMES")
print("=" * 70)

print("\nTotal features:", len(feature_names))

print("\nAll encoded features:\n")

for i, name in enumerate(feature_names):
    print(f"{i:2d} : {name}")


# ============================================================
# 8. SAVE FEATURE NAMES
# ============================================================

feature_names_df = pd.DataFrame({
    "feature_index": range(len(feature_names)),
    "feature_name": feature_names
})

feature_names_df.to_csv(
    "encoded_feature_names.csv",
    index=False
)

print("\nSaved as:")
print("encoded_feature_names.csv")
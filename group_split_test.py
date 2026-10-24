import pandas as pd
import numpy as np

# ============================================================
# LOAD CLEANED TON_IOT DATA
# ============================================================

df = pd.read_csv("ton_iot_cleaned.csv")

print("=" * 70)
print("TON_IOT DATASET")
print("=" * 70)

print("Dataset shape:", df.shape)
print("Unique source IPs:", df["src_ip"].nunique())


# ============================================================
# CREATE DNS STRUCTURAL FEATURES
# ============================================================

def query_length(x):
    if x == "-" or pd.isna(x):
        return 0
    return len(str(x))


def query_num_labels(x):
    if x == "-" or pd.isna(x):
        return 0
    return len(str(x).split("."))


def query_digit_ratio(x):
    if x == "-" or pd.isna(x):
        return 0

    x = str(x)

    return sum(c.isdigit() for c in x) / max(len(x), 1)


def query_entropy(x):
    if x == "-" or pd.isna(x):
        return 0

    x = str(x)

    if len(x) == 0:
        return 0

    counts = pd.Series(list(x)).value_counts()
    probabilities = counts / counts.sum()

    return -(probabilities * np.log2(probabilities)).sum()


df["dns_present"] = (
    df["dns_query"] != "-"
).astype(int)

df["dns_query_length"] = (
    df["dns_query"].apply(query_length)
)

df["dns_query_num_labels"] = (
    df["dns_query"].apply(query_num_labels)
)

df["dns_query_digit_ratio"] = (
    df["dns_query"].apply(query_digit_ratio)
)

df["dns_query_entropy"] = (
    df["dns_query"].apply(query_entropy)
)


# ============================================================
# VERIFY
# ============================================================

new_features = [
    "dns_present",
    "dns_query_length",
    "dns_query_num_labels",
    "dns_query_digit_ratio",
    "dns_query_entropy"
]

print("\nCreated DNS features:")

for feature in new_features:
    print(
        f"{feature:30s} "
        f"unique={df[feature].nunique()}"
    )


# ============================================================
# CHECK FOR MISSING COLUMNS
# ============================================================

required_features = [
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
    "http_response_body_len",

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

missing = [
    col for col in required_features
    if col not in df.columns
]

print("\nMissing columns:")

if len(missing) == 0:
    print("None")
    print("\nAll required features are available.")
else:
    print(missing)
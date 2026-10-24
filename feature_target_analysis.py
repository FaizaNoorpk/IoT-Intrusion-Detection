import pandas as pd

# Load cleaned dataset
df = pd.read_csv("ton_iot_cleaned.csv")

print("=" * 70)
print("FEATURE - TARGET ANALYSIS")
print("=" * 70)

print("\nDataset shape:", df.shape)

# ---------------------------------------------------------
# 1. Target distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(df["label"].value_counts())
print("\nPercentage:")
print((df["label"].value_counts(normalize=True) * 100).round(2))


# ---------------------------------------------------------
# 2. Numerical feature analysis
# ---------------------------------------------------------

numerical_features = [
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
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_request_body_len",
    "http_response_body_len",
    "http_status_code"
]

print("\n" + "=" * 70)
print("NUMERICAL FEATURES BY LABEL")
print("=" * 70)

for col in numerical_features:

    print("\n" + "-" * 60)
    print("Feature:", col)

    result = df.groupby("label")[col].agg(
        ["count", "mean", "median", "min", "max"]
    )

    print(result)


# ---------------------------------------------------------
# 3. Categorical feature analysis
# ---------------------------------------------------------

categorical_features = [
    "proto",
    "service",
    "conn_state",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected"
]

print("\n" + "=" * 70)
print("CATEGORICAL FEATURES BY LABEL")
print("=" * 70)

for col in categorical_features:

    print("\n" + "-" * 60)
    print("Feature:", col)

    table = pd.crosstab(
        df[col],
        df["label"],
        normalize="columns"
    ) * 100

    print(table.round(2))


# ---------------------------------------------------------
# 4. Save numerical summary
# ---------------------------------------------------------

numerical_summary = df.groupby("label")[numerical_features].agg(
    ["mean", "median", "min", "max"]
)

numerical_summary.to_csv("numerical_feature_target_summary.csv")

print("\n" + "=" * 70)
print("Analysis completed.")
print("Saved: numerical_feature_target_summary.csv")
print("=" * 70)
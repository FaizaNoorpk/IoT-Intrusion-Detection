import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("ton_iot_cleaned.csv")

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print("Shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True) * 100)

print("\nAttack type distribution:")
print(df["type"].value_counts())


# ============================================================
# 2. NUMERICAL FEATURE DISTRIBUTION
# ============================================================

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
print("NUMERICAL FEATURE DISTRIBUTIONS")
print("=" * 70)

for col in numerical_features:

    print("\n" + "-" * 70)
    print("Feature:", col)
    print("-" * 70)

    print(df.groupby("label")[col].quantile(
        [0, 0.01, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999, 1]
    ).unstack())


# ============================================================
# 3. ZERO-VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ZERO-VALUE ANALYSIS")
print("=" * 70)

for col in numerical_features:

    zero_count = (df[col] == 0).sum()
    zero_percentage = zero_count / len(df) * 100

    print(
        f"{col:30s} "
        f"zeros={zero_count:8d} "
        f"percentage={zero_percentage:8.2f}%"
    )


# ============================================================
# 4. FEATURE VALUES BY ATTACK TYPE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE MEDIANS BY ATTACK TYPE")
print("=" * 70)

important_numeric = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes"
]

for col in important_numeric:

    print("\n" + "-" * 70)
    print("Feature:", col)
    print("-" * 70)

    print(
        df.groupby("type")[col]
        .agg(["count", "median", "mean", "max"])
        .sort_values("median", ascending=False)
    )


# ============================================================
# 5. LABEL DISTRIBUTION WITHIN ATTACK TYPES
# ============================================================

print("\n" + "=" * 70)
print("LABEL BY ATTACK TYPE")
print("=" * 70)

label_type_table = pd.crosstab(
    df["type"],
    df["label"],
    normalize="index"
) * 100

print(label_type_table)


# ============================================================
# 6. UNIQUE IP ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("IP ADDRESS ANALYSIS")
print("=" * 70)

for col in ["src_ip", "dst_ip"]:

    print("\nFeature:", col)

    print("Unique values:", df[col].nunique())

    print("\nMost common values:")
    print(df[col].value_counts().head(15))


# ============================================================
# 7. LABEL DISTRIBUTION BY SOURCE IP
# ============================================================

print("\n" + "=" * 70)
print("SOURCE IP vs LABEL")
print("=" * 70)

src_ip_label = pd.crosstab(
    df["src_ip"],
    df["label"],
    normalize="index"
) * 100

print(src_ip_label.sort_values(1, ascending=False).head(20))


# ============================================================
# 8. LABEL DISTRIBUTION BY DESTINATION IP
# ============================================================

print("\n" + "=" * 70)
print("DESTINATION IP vs LABEL")
print("=" * 70)

dst_ip_label = pd.crosstab(
    df["dst_ip"],
    df["label"],
    normalize="index"
) * 100

print(dst_ip_label.sort_values(1, ascending=False).head(20))


# ============================================================
# 9. CATEGORICAL FEATURE LABEL PURITY
# ============================================================

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
print("CATEGORICAL FEATURE vs LABEL")
print("=" * 70)

for col in categorical_features:

    print("\n" + "-" * 70)
    print("Feature:", col)
    print("-" * 70)

    table = pd.crosstab(
        df[col],
        df["label"],
        normalize="index"
    ) * 100

    print(table)


# ============================================================
# 10. ATTACK TYPE vs CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("ATTACK TYPE vs CATEGORICAL FEATURES")
print("=" * 70)

for col in categorical_features:

    print("\n" + "-" * 70)
    print("Feature:", col)
    print("-" * 70)

    table = pd.crosstab(
        df["type"],
        df[col],
        normalize="index"
    ) * 100

    print(table)


# ============================================================
# 11. HIGH-CARDINALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CARDINALITY FEATURES")
print("=" * 70)

for col in df.columns:

    if col not in ["label", "type"]:

        unique_count = df[col].nunique()

        if unique_count > 100:

            print(
                f"{col:30s} "
                f"unique values = {unique_count}"
            )


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("Dataset shape:", df.shape)
print("Number of features:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())
print("Analysis finished successfully.")
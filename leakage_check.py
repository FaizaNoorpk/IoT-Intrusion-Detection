import pandas as pd
import numpy as np

df = pd.read_csv("ton_iot_cleaned.csv")

print("=" * 70)
print("LEAKAGE AND FEATURE PURITY CHECK")
print("=" * 70)


# ------------------------------------------------------------
# 1. SINGLE-FEATURE CATEGORICAL PURITY
# ------------------------------------------------------------

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
print("CATEGORICAL FEATURE PURITY")
print("=" * 70)

for col in categorical_features:

    table = pd.crosstab(df[col], df["label"])

    table["total"] = table.sum(axis=1)

    if 0 in table.columns:
        normal = table[0]
    else:
        normal = 0

    if 1 in table.columns:
        attack = table[1]
    else:
        attack = 0

    table["normal_pct"] = normal / table["total"] * 100
    table["attack_pct"] = attack / table["total"] * 100

    print("\nFeature:", col)
    print(table.sort_values("total", ascending=False))


# ------------------------------------------------------------
# 2. IP PURITY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IP PURITY")
print("=" * 70)

for col in ["src_ip", "dst_ip"]:

    table = pd.crosstab(df[col], df["label"])

    table["total"] = table.sum(axis=1)

    if 0 in table.columns:
        table["normal_pct"] = table[0] / table["total"] * 100

    if 1 in table.columns:
        table["attack_pct"] = table[1] / table["total"] * 100

    table = table.sort_values("total", ascending=False)

    print("\nFeature:", col)

    print("Number of unique values:", df[col].nunique())

    print("\nMost frequent values:")
    print(table.head(20))


# ------------------------------------------------------------
# 3. TYPE AND FEATURE RELATIONSHIP
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ATTACK TYPE COUNTS")
print("=" * 70)

print(df["type"].value_counts())


# ------------------------------------------------------------
# 4. TYPE BLOCK ORDER
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TYPE ORDER")
print("=" * 70)

type_changes = df["type"].ne(df["type"].shift())

blocks = df.loc[type_changes, "type"].tolist()

print("Number of blocks:", len(blocks))
print("Block order:")

for i, attack_type in enumerate(blocks, start=1):
    print(i, attack_type)


# ------------------------------------------------------------
# 5. IDENTICAL FEATURE VECTORS WITH DIFFERENT LABELS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE VECTOR LABEL CONFLICT CHECK")
print("=" * 70)

feature_columns = [
    col for col in df.columns
    if col not in ["label", "type"]
]

grouped = (
    df.groupby(feature_columns, dropna=False)["label"]
    .nunique()
)

conflicts = (grouped > 1).sum()

print("Feature vectors having both normal and attack labels:", conflicts)


# ------------------------------------------------------------
# 6. UNIQUE ROW RATIO
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("UNIQUENESS")
print("=" * 70)

print("Total rows:", len(df))
print("Unique complete rows:", df.drop_duplicates().shape[0])
print(
    "Unique row percentage:",
    df.drop_duplicates().shape[0] / len(df) * 100
)


print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)
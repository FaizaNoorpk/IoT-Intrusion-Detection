import pandas as pd


# ============================================================
# 1. LOAD FILES
# ============================================================

feature_names = pd.read_csv("encoded_feature_names.csv")
importance = pd.read_csv("random_forest_feature_importance.csv")


# ============================================================
# 2. CONNECT USING FEATURE INDEX
# ============================================================

importance["feature_index"] = (
    importance["feature"]
    .str.replace("feature_", "", regex=False)
    .astype(int)
)

df = feature_names.merge(
    importance[["feature_index", "importance"]],
    on="feature_index",
    how="inner"
)


# ============================================================
# 3. EXTRACT ORIGINAL FEATURE
# ============================================================

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


def get_original_feature(name):

    if name.startswith("num__"):
        return name.replace("num__", "")

    if name.startswith("cat__"):
        name = name.replace("cat__", "")

        for feature in categorical_features:
            if name.startswith(feature + "_"):
                return feature

    return name


df["original_feature"] = df["feature_name"].apply(
    get_original_feature
)


# ============================================================
# 4. GROUP IMPORTANCE
# ============================================================

grouped = (
    df.groupby("original_feature")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)


# ============================================================
# 5. DISPLAY
# ============================================================

print("=" * 70)
print("GROUPED RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

print("\n")

for i, row in grouped.iterrows():

    print(
        f"{i + 1:2d}. "
        f"{row['original_feature']:30s} "
        f"{row['importance']:.6f}"
    )


# ============================================================
# 6. SAVE
# ============================================================

grouped.to_csv(
    "grouped_feature_importance.csv",
    index=False
)

print("\n" + "=" * 70)
print("RESULT SAVED")
print("=" * 70)

print("\nFile:")
print("grouped_feature_importance.csv")
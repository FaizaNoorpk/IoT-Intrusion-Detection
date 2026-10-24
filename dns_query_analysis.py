import pandas as pd
import numpy as np

df = pd.read_csv("ton_iot_cleaned.csv")

print("=" * 70)
print("DNS QUERY ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# 1. Basic information
# ---------------------------------------------------------

print("\nUnique DNS query values:")
print(df["dns_query"].nunique())

print("\nMost frequent DNS query values:")
print(df["dns_query"].value_counts().head(20))


# ---------------------------------------------------------
# 2. Placeholder values
# ---------------------------------------------------------

dash_count = (df["dns_query"] == "-").sum()
dash_percentage = dash_count / len(df) * 100

print("\nDNS '-' values:")
print("Count:", dash_count)
print("Percentage:", round(dash_percentage, 2), "%")


# ---------------------------------------------------------
# 3. DNS query presence vs label
# ---------------------------------------------------------

df["dns_present"] = df["dns_query"] != "-"

print("\nDNS query presence vs label:")
print(
    pd.crosstab(
        df["dns_present"],
        df["label"],
        normalize="index"
    ) * 100
)


# ---------------------------------------------------------
# 4. Create structural DNS features
# ---------------------------------------------------------

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


df["dns_query_length"] = df["dns_query"].apply(query_length)

df["dns_query_num_labels"] = df["dns_query"].apply(query_num_labels)

df["dns_query_digit_ratio"] = df["dns_query"].apply(query_digit_ratio)

df["dns_query_entropy"] = df["dns_query"].apply(query_entropy)


# ---------------------------------------------------------
# 5. Compare DNS structural features by label
# ---------------------------------------------------------

dns_features = [
    "dns_query_length",
    "dns_query_num_labels",
    "dns_query_digit_ratio",
    "dns_query_entropy"
]

print("\nDNS structural features by label:")
print(
    df.groupby("label")[dns_features].agg(
        ["mean", "median", "max"]
    )
)


# ---------------------------------------------------------
# 6. Compare DNS structural features by attack type
# ---------------------------------------------------------

print("\nDNS structural features by attack type:")

print(
    df.groupby("type")[dns_features].mean().round(3)
)


# ---------------------------------------------------------
# 7. Check whether structural features are useful
# ---------------------------------------------------------

print("\nCorrelation with binary label:")

for col in dns_features:
    correlation = df[col].corr(df["label"])
    print(
        f"{col}: {correlation:.4f}"
    )


print("\n" + "=" * 70)
print("DNS ANALYSIS COMPLETE")
print("=" * 70)
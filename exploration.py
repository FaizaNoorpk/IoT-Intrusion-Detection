import pandas as pd

df = pd.read_csv("train_test_network.csv")

print(df.shape)
print(df.head())

print(df.columns)

print(df.info())
print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True) * 100)

print("\nAttack types:")
print(df["type"].value_counts())

print("\nDuplicate rows:")
print(df.duplicated().sum())
print("\nDash values by column:")
print((df == "-").sum().sort_values(ascending=False))

duplicate_rows = df[df.duplicated(keep=False)]

print("Total duplicate records:", len(duplicate_rows))

print("\nDuplicate records by label:")
print(duplicate_rows["label"].value_counts())

print("\nDuplicate records by type:")
print(duplicate_rows["type"].value_counts())
df.duplicated().sum()
print("\nColumns containing '-' values:")

dash_counts = (df == "-").sum()

print(dash_counts[dash_counts > 0])
dash_columns = (df == "-").sum()

dash_percentage = (dash_columns / len(df)) * 100

dash_summary = pd.DataFrame({
    "dash_count": dash_columns,
    "dash_percentage": dash_percentage
})

print(dash_summary[dash_summary["dash_count"] > 0]
      .sort_values("dash_percentage", ascending=False))
categorical_columns = [
    "proto",
    "service",
    "conn_state",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_version",
    "http_method"
]

for col in categorical_columns:
    print("\n" + "=" * 50)
    print(f"Column: {col}")
    print(df[col].value_counts(dropna=False).head(10))
    print("Duplicate percentage:")
print((df.duplicated().sum() / len(df)) * 100)

print("\nUnique records:")
print(df.drop_duplicates().shape)

print("\nAttack types in full dataset:")
print(df["type"].value_counts())
print("\nFirst 20 attack types:")
print(df["type"].head(20).to_list())

print("\nLast 20 attack types:")
print(df["type"].tail(20).to_list())

print("\nType transitions:")
print(df["type"].ne(df["type"].shift()).sum())
print("\nDuplicate rows by attack type:")
duplicate_rows = df[df.duplicated(keep=False)]

print(
    duplicate_rows["type"]
    .value_counts()
)
print("\nDuplicate rows by label:")
print(
    duplicate_rows["label"]
    .value_counts()
)
print("\nDuplicate rows with multiple attack types:")

duplicate_groups = (
    df.groupby(
        [col for col in df.columns if col not in ["label", "type"]]
    )["type"]
    .nunique()
)

print((duplicate_groups > 1).sum())

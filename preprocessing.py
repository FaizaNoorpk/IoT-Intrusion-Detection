import pandas as pd

# Load dataset
df = pd.read_csv("train_test_network.csv")

print("Original shape:", df.shape)

# Check duplicates
duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)

# Remove exact duplicate rows
df_clean = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df_clean.shape)

# Verify duplicates are gone
print("Remaining duplicate rows:", df_clean.duplicated().sum())

# Save cleaned dataset
df_clean.to_csv("ton_iot_cleaned.csv", index=False)

print("\nCleaned dataset saved as: ton_iot_cleaned.csv")
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("ton_iot_cleaned.csv")

print("=" * 70)
print("TON_IoT FEATURE ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

# Create feature summary
summary = []

for col in df.columns:
    summary.append({
        "feature": col,
        "dtype": str(df[col].dtype),
        "unique_values": df[col].nunique(dropna=False),
        "missing_values": df[col].isna().sum(),
        "missing_percentage": round(df[col].isna().mean() * 100, 2),
        "dash_values": (df[col] == "-").sum(),
        "dash_percentage": round((df[col] == "-").mean() * 100, 2)
    })

feature_summary = pd.DataFrame(summary)

print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

print(feature_summary.to_string(index=False))

# Save report
feature_summary.to_csv("feature_analysis_report.csv", index=False)

print("\n" + "=" * 70)
print("Report saved as: feature_analysis_report.csv")
print("=" * 70)

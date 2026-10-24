import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

print("=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

X_train = load_npz("X_train_processed.npz")
X_test = load_npz("X_test_processed.npz")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# 2. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("\nRandom Forest training complete.")


# ============================================================
# 3. GET FEATURE IMPORTANCE
# ============================================================

importance = rf.feature_importances_

print("\nNumber of model features:", len(importance))


# ============================================================
# 4. CREATE FEATURE IMPORTANCE TABLE
# ============================================================

feature_names = [
    f"feature_{i}"
    for i in range(X_train.shape[1])
]

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


# ============================================================
# 5. DISPLAY TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FEATURES")
print("=" * 70)

print(
    importance_df.head(20).to_string(index=False)
)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

importance_df.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

print("\nFeature importance saved as:")
print("random_forest_feature_importance.csv")

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 70)
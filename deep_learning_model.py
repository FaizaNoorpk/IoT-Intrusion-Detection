import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("TON_IOT DEEP LEARNING MODEL")
print("=" * 70)

# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

print("\nLoading processed data...")

X_train = sp.load_npz("X_train_processed.npz")
X_test = sp.load_npz("X_test_processed.npz")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# TensorFlow Dense layers need dense arrays.
X_train = X_train.toarray().astype("float32")
X_test = X_test.toarray().astype("float32")

print("Dense X_train:", X_train.shape)
print("Dense X_test :", X_test.shape)

# ============================================================
# 2. BUILD NEURAL NETWORK
# ============================================================

print("\nBuilding neural network...")

model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.30),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.25),

    Dense(32, activation="relu"),
    Dropout(0.20),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# ============================================================
# 3. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ============================================================
# 4. TRAIN
# ============================================================

print("\nTraining model...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.15,
    epochs=30,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1
)

# ============================================================
# 5. PREDICTION
# ============================================================

print("\nEvaluating model...")

y_probability = model.predict(
    X_test,
    batch_size=512,
    verbose=0
).ravel()

y_pred = (y_probability >= 0.5).astype(int)

# ============================================================
# 6. METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 70)
print("DEEP LEARNING RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Attack"]
))

# ============================================================
# 7. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{
    "Model": "Deep Neural Network",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "ROC_AUC": roc_auc
}])

results.to_csv(
    "deep_learning_results.csv",
    index=False
)

# ============================================================
# 8. SAVE MODEL
# ============================================================

model.save("ton_iot_deep_learning_model.keras")

print("\nSaved:")
print("  deep_learning_results.csv")
print("  ton_iot_deep_learning_model.keras")

print("\nDeep learning stage completed.")
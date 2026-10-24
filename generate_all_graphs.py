
import numpy as np
import pandas as pd
import matplotlib
# Forces a graphical window popup on your computer screen
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("graphs")
OUT.mkdir(exist_ok=True)

# ============================================================
# 1. MODEL COMPARISON
# ============================================================
results = pd.DataFrame([
    ["Logistic Regression", "Conventional", 0.9839, 0.9888, 0.9906, 0.9897, 0.9942],
    ["Random Forest", "Conventional", 0.9990, 0.9990, 0.9997, 0.9994, 1.0000],
    ["DNN", "Conventional", 0.9936, 0.9948, 0.9970, 0.9959, 0.9988],
    ["Random Forest", "Source-aware", 0.9428, 0.9453, 0.9921, 0.9682, 0.9654],
], columns=["Model","Evaluation","Accuracy","Precision","Recall","F1","ROC-AUC"])

x = np.arange(len(results))
width = 0.16

fig, ax = plt.subplots(figsize=(11, 6))
for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
    ax.bar(x + (i-2)*width, results[metric], width, label=metric)

ax.set_xticks(x)
ax.set_xticklabels([
    "Logistic Regression\nConventional",
    "Random Forest\nConventional",
    "DNN\nConventional",
    "Random Forest\nSource-aware"
])
ax.set_ylabel("Score")
ax.set_ylim(0.85, 1.01)
ax.set_title("TON_IoT Model Performance")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "01_model_comparison.png", dpi=300)
plt.close(fig)
plt.show()

# ============================================================
# 2. CONFUSION MATRIX - LOGISTIC REGRESSION
# ============================================================
cm_lr = np.array([[8074, 334],
                  [278, 29409]])

def save_cm(cm, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Attack"])
    ax.set_yticklabels(["Normal", "Attack"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center", fontsize=13)

    fig.colorbar(im, ax=ax, label="Samples")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300)
    plt.close(fig)
    plt.show()

save_cm(
    cm_lr,
    "Logistic Regression — Conventional Split",
    "02_confusion_matrix_logistic_regression.png"
)

# ============================================================
# 3. CONFUSION MATRIX - RANDOM FOREST CONVENTIONAL
# ============================================================
cm_rf = np.array([[8379, 29],
                  [8, 29679]])

save_cm(
    cm_rf,
    "Random Forest — Conventional Split",
    "03_confusion_matrix_random_forest_conventional.png"
)

# ============================================================
# 4. CONFUSION MATRIX - DNN
# THIS WAS THE MISSING ONE
# ============================================================
cm_dnn = np.array([[8252, 156],
                   [88, 29599]])

save_cm(
    cm_dnn,
    "Deep Neural Network — Conventional Split",
    "04_confusion_matrix_dnn.png"
)

# ============================================================
# 5. CONFUSION MATRIX - RANDOM FOREST SOURCE-AWARE
# ============================================================
cm_source = np.array([[5108, 3493],
                      [478, 60358]])

save_cm(
    cm_source,
    "Random Forest — Source-Aware Split",
    "05_confusion_matrix_random_forest_source_aware.png"
)

# ============================================================
# 6. ATTACK-TYPE RECALL
# ============================================================
attack = pd.DataFrame([
    ["MITM", 0.8926],
    ["XSS", 0.9773],
    ["Backdoor", 0.9800],
    ["DoS", 0.9970],
    ["DDoS", 0.9988],
    ["Password", 0.9992],
    ["Scanning", 0.9996],
    ["Ransomware", 0.9998],
    ["Injection", 0.9999],
], columns=["Attack Type", "Recall"])

attack = attack.sort_values("Recall")

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(attack["Attack Type"], attack["Recall"])
ax.set_xlabel("Recall")
ax.set_xlim(0.85, 1.01)
ax.set_title("Attack-Type Recall — Source-Aware Test Set")
fig.tight_layout()
fig.savefig(OUT / "06_attack_type_recall.png", dpi=300)
plt.close(fig)
plt.show()

# ============================================================
# 7. RANDOM SPLIT VS SOURCE-AWARE
# ============================================================
comparison = pd.DataFrame([
    ["Conventional split", 0.9990, 0.9994],
    ["Source-aware split", 0.9428, 0.9682],
], columns=["Evaluation", "Accuracy", "F1"])

x = np.arange(len(comparison))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 0.15, comparison["Accuracy"], 0.30, label="Accuracy")
ax.bar(x + 0.15, comparison["F1"], 0.30, label="F1")
ax.set_xticks(x)
ax.set_xticklabels(comparison["Evaluation"])
ax.set_ylabel("Score")
ax.set_ylim(0.85, 1.01)
ax.set_title("Conventional vs Source-Aware Evaluation")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "07_conventional_vs_source_aware.png", dpi=300)
plt.close(fig)
plt.show()

# ============================================================
# 8. RF FEATURE IMPORTANCE
# ============================================================
importance = pd.DataFrame([
["proto",.187547],["src_pkts",.106918],["conn_state",.084720],
["dns_rejected",.081666],["src_ip_bytes",.079094],["service",.056586],
["src_port",.052394],["dst_port",.051836],["dns_RA",.041042],
["duration",.034179],["dst_pkts",.033676],["dst_ip_bytes",.031083],
["dns_query_num_labels",.026818],["dns_AA",.015289],
["dns_query_length",.015118],["dns_query_entropy",.014640],
["dns_RD",.014157],["dst_bytes",.013858],["src_bytes",.012015],
["dns_present",.011545],["dns_qclass",.010042],["dns_qtype",.009142],
["dns_query_digit_ratio",.005163],["dns_rcode",.003835],
["http_status_code",.003644],["http_response_body_len",.002174],
["missed_bytes",.001759],["http_request_body_len",.000063]
], columns=["Feature","Importance"])

top = importance.sort_values("Importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top["Feature"], top["Importance"])
ax.set_xlabel("Grouped importance")
ax.set_title("Top Random Forest Feature Importances")
fig.tight_layout()
fig.savefig(OUT / "08_rf_feature_importance_top15.png", dpi=300)
plt.close(fig)
plt.show()

# ============================================================
# SAVE TABLE
# ============================================================
results.to_csv(OUT / "model_results.csv", index=False)
attack.to_csv(OUT / "attack_type_recall.csv", index=False)
importance.to_csv(OUT / "grouped_feature_importance.csv", index=False)

print("All graphs created in:", OUT.resolve())

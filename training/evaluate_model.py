import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, ConfusionMatrixDisplay,
    roc_curve, auc
)

# -----------------------------
# CONSTANTS (SONAR FIX)
# -----------------------------
ACCURACY = "Accuracy"
PRECISION = "Precision"
RECALL = "Recall"
F1 = "F1 Score"

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("../dataset/telco_customer_churn.csv")

df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

# Encode categorical columns
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encoder.fit_transform(df[col])

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Load trained model
# -----------------------------
with open("../model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

print("\nModel Used:", type(model).__name__)

# -----------------------------
# Predictions (FINAL MODEL)
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Metrics (FINAL MODEL)
# -----------------------------
print("\n=== FINAL MODEL PERFORMANCE ===")
print(f"{ACCURACY} :", accuracy_score(y_test, y_pred))
print(f"{PRECISION}:", precision_score(y_test, y_pred))
print(f"{RECALL}   :", recall_score(y_test, y_pred))
print(f"{F1}       :", f1_score(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

# -----------------------------
# Feature Importance (only if RF)
# -----------------------------
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-10:]

    plt.figure()
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), X.columns[indices])
    plt.title("Top 10 Feature Importance")
    plt.savefig("feature_importance.png")
    plt.show()

# -----------------------------
# MODEL COMPARISON
# -----------------------------
print("\n=== MODEL COMPARISON ===")

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\n🔹 Logistic Regression")
print(f"{ACCURACY} :", accuracy_score(y_test, lr_pred))
print(f"{PRECISION}:", precision_score(y_test, lr_pred))
print(f"{RECALL}   :", recall_score(y_test, lr_pred))
print(f"{F1}       :", f1_score(y_test, lr_pred))

# Random Forest (SONAR FIXED)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n🌳 Random Forest")
print(f"{ACCURACY} :", accuracy_score(y_test, rf_pred))
print(f"{PRECISION}:", precision_score(y_test, rf_pred))
print(f"{RECALL}   :", recall_score(y_test, rf_pred))
print(f"{F1}       :", f1_score(y_test, rf_pred))

# -----------------------------
# Comparison Graph
# -----------------------------
models = ["Logistic Regression", "Random Forest"]
accuracies = [
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, rf_pred)
]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("model_comparison.png")
plt.show()
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ✅ IMPORT PREPROCESS FUNCTION (IMPORTANT)
from preprocess_data import preprocess_data

# ----------------------------
# CONSTANT
# ----------------------------
TARGET = "Churn"

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("../dataset/telco_customer_churn.csv")

# ✅ APPLY PREPROCESSING (REMOVED DUPLICATION)
df = preprocess_data(df)

# ----------------------------
# Separate features and target
# ----------------------------
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Logistic Regression
# ----------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print("\n🔹 Logistic Regression")
print("Accuracy:", lr_accuracy)
print("F1 Score:", lr_f1)

# ----------------------------
# Random Forest
# ----------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\n🌳 Random Forest")
print("Accuracy:", rf_accuracy)
print("F1 Score:", rf_f1)

# ----------------------------
# Select Best Model
# ----------------------------
if rf_f1 > lr_f1:
    best_model = rf
    selected_model_name = "Random Forest"
else:
    best_model = lr
    selected_model_name = "Logistic Regression"

print("\n✅ Selected Model:", selected_model_name)

# ----------------------------
# Save model
# ----------------------------
with open("../model/churn_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved as churn_model.pkl")
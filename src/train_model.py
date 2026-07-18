import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Loan_default.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create models folder if it does not exist
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "loan_default_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove LoanID
df.drop("LoanID", axis=1, inplace=True)

categorical_columns = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

# Encode categorical columns
df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
)

# Separate features and target
X = df.drop("Default", axis=1)
y = df["Default"]

feature_columns = X.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save model
with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

# Save scaler
with open(SCALER_PATH, "wb") as file:
    pickle.dump(scaler, file)

# Save feature column order
with open(FEATURES_PATH, "wb") as file:
    pickle.dump(feature_columns, file)

print("\nSaved files:")
print(MODEL_PATH)
print(SCALER_PATH)
print(FEATURES_PATH)
import pandas as pd

# Load Dataset
df = pd.read_csv("data/raw/Loan_default.csv")

# Display data types
print(df.dtypes)

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("data/raw/Loan_default.csv")

# Remove ID column
df = df.drop("LoanID", axis=1)

# One-Hot Encoding
categorical_columns = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# Separate features and target
X = df.drop("Default", axis=1)
y = df["Default"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing Completed Successfully!")
print("Training Features:", X_train_scaled.shape)
print("Testing Features:", X_test_scaled.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)
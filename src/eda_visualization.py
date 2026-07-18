import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/raw/Loan_default.csv")

# Target Variable Count Plot
df["Default"].value_counts().plot(kind="bar")

plt.title("Loan Default Distribution")
plt.xlabel("Default")
plt.ylabel("Count")

plt.show()
# ==============================
# Age Distribution
# ==============================

plt.figure(figsize=(8,5))

plt.hist(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()
# ==============================
# Income Distribution
# ==============================

plt.figure(figsize=(8,5))

plt.hist(df["Income"], bins=20, color="green", edgecolor="black")

plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")

plt.show()
# ==============================
# Loan Amount Distribution
# ==============================

plt.figure(figsize=(8,5))

plt.hist(df["LoanAmount"], bins=20, color="orange", edgecolor="black")

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")

plt.show()
# ==============================
# Credit Score Distribution
# ==============================

plt.figure(figsize=(8,5))

plt.hist(df["CreditScore"], bins=20, color="purple", edgecolor="black")

plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Frequency")

plt.show()
print("All graphs completed successfully.")
# ==============================
# Correlation Heatmap
# ==============================

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
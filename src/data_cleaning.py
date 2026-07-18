import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/Loan_default.csv")

# Dataset Information
print(df.info())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())
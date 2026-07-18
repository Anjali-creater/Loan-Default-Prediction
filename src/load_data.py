import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/Loan_default.csv")

# Show first 5 rows
print(df.head())
#shape
print("\nShape of Dataset:")
print(df.shape)
#columns name 
print("\nColumns in Dataset:")
print(df.columns)
#dataset info
print("\nDataset Info:")
print(df.info())
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())
# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())
# Unique values in each column
print("\nUnique Values:")
print(df.nunique())
# Target Variable Count
print("\nTarget Variable Distribution:")
print(df["Default"].value_counts())



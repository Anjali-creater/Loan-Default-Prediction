import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/Loan_default.csv")

# Display all column names
print("Columns in Dataset:\n")
print(df.columns)
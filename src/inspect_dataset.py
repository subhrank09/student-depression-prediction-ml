import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Student Depression Dataset.csv")

# Dataset shape
print("Shape:", df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# First 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Target distribution
print("\nTarget Distribution:")
print(df["Depression"].value_counts())
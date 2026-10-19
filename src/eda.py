import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("dataset/Student Depression Dataset.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Shape:", df.shape)

# ============================================
# 2. BASIC INFORMATION
# ============================================

print("\nColumns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\nData Types:")
print(df.dtypes)

# ============================================
# 3. MISSING VALUES
# ============================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

print(missing[missing > 0])

# ============================================
# 4. DUPLICATES
# ============================================

print("\nDuplicate Rows:", df.duplicated().sum())

# ============================================
# 5. TARGET DISTRIBUTION
# ============================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df["Depression"].value_counts()

print(target_counts)

print("\nTarget Percentage:")
print(
    (df["Depression"].value_counts(normalize=True) * 100)
    .round(2)
)

# ============================================
# 6. NUMERICAL FEATURES
# ============================================

print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print(numeric_columns.tolist())

print("\nStatistical Summary:")
print(df[numeric_columns].describe())

# ============================================
# 7. CATEGORICAL FEATURES
# ============================================

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

print(categorical_columns.tolist())

# ============================================
# 8. UNIQUE VALUES
# ============================================

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for column in categorical_columns:
    print(f"\n{column}")
    print("Number of unique values:", df[column].nunique())
    print(df[column].value_counts().head(15))

# ============================================
# 9. TARGET VISUALIZATION
# ============================================

plt.figure(figsize=(7, 5))

df["Depression"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Depression Distribution")
plt.xlabel("Depression")
plt.ylabel("Number of Students")

plt.xticks(
    ticks=[0, 1],
    labels=["No Depression", "Depression"],
    rotation=0
)

plt.tight_layout()
plt.show()

# ============================================
# 10. AGE DISTRIBUTION
# ============================================

plt.figure(figsize=(8, 5))

plt.hist(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()

# ============================================
# 11. CGPA DISTRIBUTION
# ============================================

plt.figure(figsize=(8, 5))

plt.hist(df["CGPA"], bins=20)

plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()

# ============================================
# 12. ACADEMIC PRESSURE VS DEPRESSION
# ============================================

pd.crosstab(
    df["Academic Pressure"],
    df["Depression"]
).plot(kind="bar", figsize=(8, 5))

plt.title("Academic Pressure vs Depression")
plt.xlabel("Academic Pressure")
plt.ylabel("Number of Students")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# ============================================
# 13. SLEEP DURATION VS DEPRESSION
# ============================================

pd.crosstab(
    df["Sleep Duration"],
    df["Depression"]
).plot(kind="bar", figsize=(10, 5))

plt.title("Sleep Duration vs Depression")
plt.xlabel("Sleep Duration")
plt.ylabel("Number of Students")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ============================================
# 14. FAMILY HISTORY VS DEPRESSION
# ============================================

pd.crosstab(
    df["Family History of Mental Illness"],
    df["Depression"]
).plot(kind="bar", figsize=(7, 5))

plt.title("Family History vs Depression")
plt.xlabel("Family History")
plt.ylabel("Number of Students")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# ============================================
# 15. FINANCIAL STRESS VS DEPRESSION
# ============================================

pd.crosstab(
    df["Financial Stress"],
    df["Depression"]
).plot(kind="bar", figsize=(8, 5))

plt.title("Financial Stress vs Depression")
plt.xlabel("Financial Stress")
plt.ylabel("Number of Students")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

print("\nEDA completed successfully.")
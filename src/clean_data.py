import pandas as pd

# ============================================================
# 1. LOAD DATASET
# ============================================================

INPUT_FILE = "dataset/Student Depression Dataset.csv"
OUTPUT_FILE = "dataset/cleaned_student_depression.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)

print("Shape:", df.shape)


# ============================================================
# 2. REMOVE DUPLICATE ROWS
# ============================================================

duplicates = df.duplicated().sum()

print("\nDuplicate rows found:", duplicates)

df = df.drop_duplicates()


# ============================================================
# 3. REMOVE ID COLUMN
# ============================================================

if "id" in df.columns:
    df = df.drop(columns=["id"])

print("\nRemoved: id")


# ============================================================
# 4. REMOVE LOW-VARIANCE / UNINFORMATIVE FEATURES
# ============================================================

columns_to_remove = [
    "Profession",
    "Work Pressure",
    "Job Satisfaction"
]

existing_columns = [
    column for column in columns_to_remove
    if column in df.columns
]

df = df.drop(columns=existing_columns)

print("\nRemoved columns:")
for column in existing_columns:
    print("-", column)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Financial Stress has only 3 missing values.
# Median is used because Financial Stress is numerical
# and the median is robust to extreme values.

if "Financial Stress" in df.columns:
    median_value = df["Financial Stress"].median()

    df["Financial Stress"] = df["Financial Stress"].fillna(
        median_value
    )

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# ============================================================
# 6. CLEAN STRING COLUMNS
# ============================================================

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()


# ============================================================
# 7. DISPLAY CLEANED DATASET
# ============================================================

print("\n" + "=" * 60)
print("CLEANED DATASET")
print("=" * 60)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 8. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df["Depression"].value_counts())

print("\nTarget Percentage:")

percentage = (
    df["Depression"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(percentage)


# ============================================================
# 9. SAVE CLEANED DATASET
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("Cleaned dataset saved to:")
print(OUTPUT_FILE)
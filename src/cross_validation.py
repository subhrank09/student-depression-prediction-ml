import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "dataset/cleaned_student_depression.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

TARGET = "Depression"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 3. IDENTIFY FEATURE TYPES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ============================================================
# 4. NUMERICAL PIPELINE
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 5. CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 6. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 7. ORIGINAL XGBOOST
# ============================================================

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 8. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            xgb_model
        )
    ]
)


# ============================================================
# 9. STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 10. EVALUATION METRICS
# ============================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ============================================================
# 11. RUN CROSS-VALIDATION
# ============================================================

print("\nRunning 5-fold cross-validation...")

results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)


# ============================================================
# 12. DISPLAY FOLD RESULTS
# ============================================================

print("\n" + "=" * 70)
print("INDIVIDUAL FOLD RESULTS")
print("=" * 70)

for i in range(5):

    print(f"\nFold {i + 1}")

    print(
        f"Accuracy : "
        f"{results['test_accuracy'][i]:.4f}"
    )

    print(
        f"Precision: "
        f"{results['test_precision'][i]:.4f}"
    )

    print(
        f"Recall   : "
        f"{results['test_recall'][i]:.4f}"
    )

    print(
        f"F1 Score : "
        f"{results['test_f1'][i]:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{results['test_roc_auc'][i]:.4f}"
    )


# ============================================================
# 13. CALCULATE MEAN AND STANDARD DEVIATION
# ============================================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

summary = []

for metric in metrics:

    scores = results[
        f"test_{metric}"
    ]

    summary.append({
        "Metric": metric.upper(),
        "Mean": scores.mean(),
        "Std": scores.std(),
        "Min": scores.min(),
        "Max": scores.max()
    })


summary_df = pd.DataFrame(summary)


# ============================================================
# 14. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

summary_df.to_csv(
    "cross_validation_summary.csv",
    index=False
)

print(
    "\nCross-validation summary saved to:"
)

print(
    "cross_validation_summary.csv"
)


# ============================================================
# 16. FINAL INTERPRETATION
# ============================================================

f1_mean = results["test_f1"].mean()
f1_std = results["test_f1"].std()

roc_mean = results["test_roc_auc"].mean()
roc_std = results["test_roc_auc"].std()

print("\n" + "=" * 70)
print("MODEL STABILITY")
print("=" * 70)

print(
    f"\nMean F1 Score: "
    f"{f1_mean:.4f} ± {f1_std:.4f}"
)

print(
    f"Mean ROC-AUC: "
    f"{roc_mean:.4f} ± {roc_std:.4f}"
)

print("\nCross-validation completed successfully.")
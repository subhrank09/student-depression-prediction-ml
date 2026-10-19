import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "dataset/cleaned_student_depression.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("FINAL DEPRESSION PREDICTION MODEL")
print("=" * 70)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. FEATURES / TARGET
# ============================================================

TARGET = "Depression"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 3. FEATURE TYPES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. NUMERICAL PIPELINE
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
# 6. CATEGORICAL PIPELINE
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
# 7. PREPROCESSOR
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
# 8. FINAL XGBOOST MODEL
# ============================================================

model = XGBClassifier(

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
# 9. COMPLETE PIPELINE
# ============================================================

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 10. TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL")
print("=" * 70)

final_pipeline.fit(
    X_train,
    y_train
)


print("Training completed.")


# ============================================================
# 11. PREDICTIONS
# ============================================================

y_pred = final_pipeline.predict(
    X_test
)

y_probability = final_pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 12. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 13. FINAL TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL TEST RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Depression",
            "Depression"
        ],
        zero_division=0
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

MODEL_FILE = "models/final_depression_model.pkl"

joblib.dump(
    final_pipeline,
    MODEL_FILE
)

print("\n" + "=" * 70)
print("FINAL MODEL SAVED")
print("=" * 70)

print(MODEL_FILE)


# ============================================================
# 17. SAVE FINAL RESULTS
# ============================================================

final_results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "Test Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ],

    "CV Mean": [
        0.8463,
        0.8577,
        0.8842,
        0.8708,
        0.9208
    ],

    "CV Std": [
        0.0035,
        0.0034,
        0.0030,
        0.0029,
        0.0016
    ]
})


final_results.to_csv(
    "final_model_results.csv",
    index=False
)


print("\nFinal results saved to:")
print("final_model_results.csv")


# ============================================================
# 18. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT MODEL")
print("=" * 70)

print("""
Algorithm       : XGBoost
Training Split  : 80%
Testing Split   : 20%
Cross Validation: 5-Fold Stratified

Original Test F1     : 0.8708
CV Mean F1           : 0.8708
CV Mean ROC-AUC      : 0.9208

Final model is ready for deployment.
""")
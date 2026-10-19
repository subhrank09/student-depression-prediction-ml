import pandas as pd
import numpy as np
import joblib
import time

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

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
print("HYPERPARAMETER TUNING - XGBOOST")
print("=" * 70)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
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
# 8. BASE XGBOOST MODEL
# ============================================================

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 9. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            xgb
        )
    ]
)


# ============================================================
# 10. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {

    "model__n_estimators": [
        100,
        150,
        200,
        250,
        300,
        400
    ],

    "model__max_depth": [
        3,
        4,
        5,
        6,
        7,
        8
    ],

    "model__learning_rate": [
        0.01,
        0.03,
        0.05,
        0.08,
        0.1,
        0.15
    ],

    "model__subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__min_child_weight": [
        1,
        3,
        5,
        7
    ],

    "model__gamma": [
        0,
        0.1,
        0.2,
        0.3
    ]
}


# ============================================================
# 11. STRATIFIED K-FOLD
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 12. RANDOMIZED SEARCH
# ============================================================

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,

    n_iter=25,

    scoring="f1",

    cv=cv,

    verbose=2,

    random_state=42,

    n_jobs=-1,

    return_train_score=True
)


# ============================================================
# 13. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING HYPERPARAMETER SEARCH")
print("=" * 70)

print("\nSearch iterations:", 25)
print("Cross-validation folds:", 5)
print("Total model fits:", 25 * 5)

start_time = time.time()

search.fit(
    X_train,
    y_train
)

end_time = time.time()

print("\nTraining time:")
print(
    f"{(end_time - start_time) / 60:.2f} minutes"
)


# ============================================================
# 14. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")


# ============================================================
# 15. BEST CROSS-VALIDATION SCORE
# ============================================================

print("\n" + "=" * 70)
print("BEST CROSS-VALIDATION RESULT")
print("=" * 70)

print(
    f"Best CV F1 Score: "
    f"{search.best_score_:.4f}"
)


# ============================================================
# 16. EVALUATE ON UNSEEN TEST DATA
# ============================================================

best_model = search.best_estimator_

y_pred = best_model.predict(X_test)

y_probability = best_model.predict_proba(
    X_test
)[:, 1]


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
# 17. FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TUNED MODEL - TEST RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 18. CLASSIFICATION REPORT
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
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 20. SAVE TUNED MODEL
# ============================================================

MODEL_FILE = "models/tuned_xgboost_model.pkl"

joblib.dump(
    best_model,
    MODEL_FILE
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_FILE)


# ============================================================
# 21. SAVE SEARCH RESULTS
# ============================================================

results = pd.DataFrame(
    search.cv_results_
)

results = results.sort_values(
    by="rank_test_score"
)

results.to_csv(
    "xgboost_hyperparameter_results.csv",
    index=False
)

print(
    "\nHyperparameter results saved to:"
)

print(
    "xgboost_hyperparameter_results.csv"
)


# ============================================================
# 22. COMPARE WITH ORIGINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL VS TUNED XGBOOST")
print("=" * 70)

print("\nOriginal XGBoost:")
print("Accuracy : 0.8468")
print("Precision: 0.8603")
print("Recall   : 0.8816")
print("F1 Score : 0.8708")
print("ROC-AUC  : 0.9198")

print("\nTuned XGBoost:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)
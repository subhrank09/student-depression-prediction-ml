import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "dataset/cleaned_student_depression.csv"
MODEL_FILE = "models/best_depression_model.pkl"

df = pd.read_csv(DATA_FILE)

# Separate X and y
X = df.drop(columns=["Depression"])
y = df["Depression"]

# Load trained XGBoost pipeline
model_pipeline = joblib.load(MODEL_FILE)

# ============================================================
# 2. GET PREPROCESSOR AND MODEL
# ============================================================

preprocessor = model_pipeline.named_steps["preprocessor"]
model = model_pipeline.named_steps["model"]

# ============================================================
# 3. GET FEATURE NAMES AFTER ONE-HOT ENCODING
# ============================================================

feature_names = preprocessor.get_feature_names_out()

print("=" * 70)
print("TRANSFORMED FEATURES")
print("=" * 70)

print("Number of transformed features:", len(feature_names))

# ============================================================
# 4. GET XGBOOST FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP 20 FEATURES")
print("=" * 70)

print(
    importance_df.head(20).to_string(
        index=False
    )
)

# ============================================================
# 5. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

# ============================================================
# 6. PLOT TOP 15 FEATURES
# ============================================================

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title(
    "Top 15 Features - XGBoost Feature Importance"
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "xgboost_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFeature importance saved to:")
print("feature_importance.csv")

print("\nGraph saved to:")
print("xgboost_feature_importance.png")
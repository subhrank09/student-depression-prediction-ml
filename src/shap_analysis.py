import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "dataset/cleaned_student_depression.csv"
MODEL_FILE = "models/best_depression_model.pkl"

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Depression"])
y = df["Depression"]


# ============================================================
# 2. LOAD MODEL
# ============================================================

pipeline = joblib.load(MODEL_FILE)

preprocessor = pipeline.named_steps["preprocessor"]
xgb_model = pipeline.named_steps["model"]


# ============================================================
# 3. USE A SAMPLE FOR SHAP
# ============================================================

_, X_sample = train_test_split(
    X,
    test_size=2000,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. TRANSFORM DATA
# ============================================================

X_transformed = preprocessor.transform(X_sample)

feature_names = preprocessor.get_feature_names_out()


# ============================================================
# 5. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(
    X_transformed
)


# ============================================================
# 6. SHAP SUMMARY PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nSHAP analysis completed.")

print("\nSHAP graph saved to:")
print("shap_summary.png")
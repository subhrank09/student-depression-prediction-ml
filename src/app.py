# ============================================================
# STUDENT MENTAL HEALTH PREDICTION
# Streamlit Application
# ============================================================

import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Mental Health Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "final_depression_model.pkl"
)

DATA_PATH = os.path.join(
    "dataset",
    "cleaned_student_depression.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained ML pipeline."""

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found:\n\n`{MODEL_PATH}`"
        )
        st.stop()

    try:
        return joblib.load(MODEL_PATH)

    except Exception as e:
        st.error(
            f"Unable to load the model.\n\nError: {e}"
        )
        st.stop()


model = load_model()


# ============================================================
# LOAD DATASET CATEGORIES
# ============================================================

@st.cache_data
def load_dataset_categories():
    """
    Load categorical values directly from the cleaned dataset.

    This guarantees that the Streamlit inputs match the
    categories used during model training.
    """

    if not os.path.exists(DATA_PATH):
        st.error(
            f"Cleaned dataset not found:\n\n`{DATA_PATH}`"
        )
        st.stop()

    try:
        df = pd.read_csv(DATA_PATH)

    except Exception as e:
        st.error(
            f"Unable to read the cleaned dataset.\n\nError: {e}"
        )
        st.stop()

    required_columns = [
        "Gender",
        "City",
        "Sleep Duration",
        "Dietary Habits",
        "Degree",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            "The cleaned dataset is missing the following columns:\n\n"
            + "\n".join(
                f"- {column}" for column in missing_columns
            )
        )
        st.stop()

    return {
        "gender": sorted(
            df["Gender"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        ),

        "cities": sorted(
            df["City"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        ),

        "degrees": sorted(
            df["Degree"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        ),

        "sleep": sorted(
            df["Sleep Duration"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        ),

        "diet": sorted(
            df["Dietary Habits"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        ),

        "suicidal": sorted(
            df["Have you ever had suicidal thoughts ?"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        ),

        "family_history": sorted(
            df["Family History of Mental Illness"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
    }


categories = load_dataset_categories()


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Student Mental Health Prediction")

st.markdown(
    """
    ### Machine Learning Based Depression Risk Prediction

    This application uses an **XGBoost machine-learning model**
    trained on academic, lifestyle, financial and family-history
    information from the Student Depression Dataset.

    **Important:** This is an academic machine-learning project.
    The prediction is **not a medical diagnosis** and should not
    replace evaluation or support from a qualified mental-health
    professional.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Model Information")

    st.write("**Algorithm:** XGBoost")

    st.write("**Training samples:** 22,320")

    st.write("**Test samples:** 5,581")

    st.write("**Test Accuracy:** 84.68%")

    st.write("**Test F1 Score:** 87.08%")

    st.write("**ROC-AUC:** 91.98%")

    st.divider()

    st.subheader("📌 Model Details")

    st.write(
        """
        The model uses:

        - Academic Pressure
        - CGPA
        - Study Satisfaction
        - Work/Study Hours
        - Financial Stress
        - Sleep Duration
        - Dietary Habits
        - Family History
        - City
        - Degree
        - Gender
        - Suicidal-thought history
        """
    )

    st.divider()

    st.caption(
        "For educational and research purposes only. "
        "This system does not provide medical diagnosis."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("👤 Student Information")

st.info(
    "Enter the student's information below and click "
    "**Predict Mental Health Risk**."
)


# ============================================================
# THREE-COLUMN INPUT LAYOUT
# ============================================================

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    st.markdown("#### 👨‍🎓 Personal Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=22,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        options=categories["gender"]
    )

    city = st.selectbox(
        "City",
        options=categories["cities"]
    )

    degree = st.selectbox(
        "Degree",
        options=categories["degrees"]
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    st.markdown("#### 📚 Academic Information")

    academic_pressure = st.slider(
        "Academic Pressure",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=1.0
    )

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.50,
        step=0.01,
        format="%.2f"
    )

    study_satisfaction = st.slider(
        "Study Satisfaction",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=1.0
    )

    work_study_hours = st.slider(
        "Work / Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
        step=1.0
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    st.markdown("#### 🏠 Lifestyle & Financial Information")

    financial_stress = st.slider(
        "Financial Stress",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=1.0
    )

    sleep_duration = st.selectbox(
        "Sleep Duration",
        options=categories["sleep"]
    )

    dietary_habits = st.selectbox(
        "Dietary Habits",
        options=categories["diet"]
    )

    family_history = st.selectbox(
        "Family History of Mental Illness",
        options=categories["family_history"]
    )


# ============================================================
# SENSITIVE FEATURE
# ============================================================

st.divider()

st.subheader("🔒 Sensitive Mental-Health Information")

st.warning(
    """
    This question is sensitive. It is included because the trained
    academic model uses it as one of its input features.

    Your response is used only as an input to the prediction displayed
    in this application.
    """
)

suicidal_thoughts = st.selectbox(
    "Have you ever had suicidal thoughts?",
    options=categories["suicidal"]
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Mental Health Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "Gender": [gender],

            "Age": [float(age)],

            "City": [city],

            "Academic Pressure": [
                float(academic_pressure)
            ],

            "CGPA": [
                float(cgpa)
            ],

            "Study Satisfaction": [
                float(study_satisfaction)
            ],

            "Sleep Duration": [
                sleep_duration
            ],

            "Dietary Habits": [
                dietary_habits
            ],

            "Degree": [
                degree
            ],

            "Have you ever had suicidal thoughts ?": [
                suicidal_thoughts
            ],

            "Work/Study Hours": [
                float(work_study_hours)
            ],

            "Financial Stress": [
                float(financial_stress)
            ],

            "Family History of Mental Illness": [
                family_history
            ]
        }
    )


    # --------------------------------------------------------
    # VALIDATE INPUT COLUMNS
    # --------------------------------------------------------

    expected_columns = [
        "Gender",
        "Age",
        "City",
        "Academic Pressure",
        "CGPA",
        "Study Satisfaction",
        "Sleep Duration",
        "Dietary Habits",
        "Degree",
        "Have you ever had suicidal thoughts ?",
        "Work/Study Hours",
        "Financial Stress",
        "Family History of Mental Illness"
    ]

    if list(input_data.columns) != expected_columns:

        st.error(
            "Input columns do not match the model's expected "
            "feature structure."
        )

        st.write(
            "Expected columns:"
        )

        st.write(expected_columns)

        st.stop()


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        # Probability of class 1 = Depression
        probability = float(
            probabilities[1]
        )

    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)


    # --------------------------------------------------------
    # RESULT CLASS
    # --------------------------------------------------------

    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ Prediction: Higher Depression Risk"
            )

        else:

            st.success(
                "✅ Prediction: Lower Depression Risk"
            )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    with result_col2:

        st.metric(
            "Predicted Depression Probability",
            f"{probability * 100:.2f}%"
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.markdown("#### Model Probability")

    st.progress(
        probability
    )

    st.caption(
        f"The model estimates a {probability * 100:.2f}% "
        "probability for the depression-positive class."
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader("ℹ️ Interpretation")

    if prediction == 1:

        st.warning(
            """
            The machine-learning model classifies this input
            pattern as belonging to the depression-positive class.

            This is a statistical prediction based on patterns
            learned from the dataset. It does **not** establish
            that the student has clinical depression.
            """
        )

    else:

        st.info(
            """
            The machine-learning model classifies this input
            pattern as belonging to the depression-negative class.

            This does **not** guarantee the absence of depression
            or any other mental-health concern.
            """
        )


    # ========================================================
    # PROBABILITY INTERPRETATION
    # ========================================================

    st.subheader("📈 Prediction Confidence")

    if probability >= 0.80:

        st.write(
            "The model assigns a relatively high probability "
            "to the depression-positive class."
        )

    elif probability >= 0.50:

        st.write(
            "The model assigns a moderate probability "
            "to the depression-positive class."
        )

    else:

        st.write(
            "The model assigns a lower probability "
            "to the depression-positive class."
        )

    st.caption(
        "This probability reflects the model's output and "
        "should not be interpreted as a clinical probability "
        "or diagnosis."
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.divider()

    with st.expander(
        "📋 View Submitted Information"
    ):

        display_data = input_data.T.rename(
            columns={0: "Value"}
        )

        st.dataframe(
            display_data,
            use_container_width=True
        )


    # ========================================================
    # MODEL OUTPUT DETAILS
    # ========================================================

    with st.expander(
        "🔎 View Model Output"
    ):

        output_data = pd.DataFrame(
            {
                "Class": [
                    "No Depression",
                    "Depression"
                ],
                "Probability": [
                    float(probabilities[0]),
                    float(probabilities[1])
                ],
                "Percentage": [
                    f"{probabilities[0] * 100:.2f}%",
                    f"{probabilities[1] * 100:.2f}%"
                ]
            }
        )

        st.dataframe(
            output_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Student Mental Health Prediction System | XGBoost |
    Academic Minor Project
    """
)

st.caption(
    """
    ⚠️ This application is for educational and research purposes.
    It is not a medical diagnostic tool.
    """
)
# ============================================================
# STEP 6 - STREAMLIT APPLICATION
# College Sleep and GPA Prediction
# ============================================================

import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="College Sleep Prediction",
    page_icon="😴",
    layout="wide"
)


# ============================================================
# 2. CONSTANTS
# ============================================================

DATA_PATH = "college_sleep_and_gpa.csv"

MODEL_FOLDER = "model"

RESULTS_PATH = "results/final_model_comparison.csv"


# Exact features retained during Step 2
FEATURE_COLUMNS = [
    "study",
    "university",
    "semester",
    "cohort_code",
    "gender",
    "first_generation",
    "underrepresented",
    "avg_sleep_minutes",
    "avg_sleep_hours",
    "daytime_sleep_minutes",
    "sleep_midpoint_minutes",
    "bedtime_variability",
    "nights_tracked_fraction",
    "prior_gpa",
    "term_gpa",
    "gpa_change",
    "term_units",
    "term_load_z"
]


# Categorical features from Step 2
CATEGORICAL_FEATURES = [
    "university",
    "semester",
    "cohort_code",
    "gender"
]


# Numerical features from Step 2
NUMERICAL_FEATURES = [
    "study",
    "first_generation",
    "underrepresented",
    "avg_sleep_minutes",
    "avg_sleep_hours",
    "daytime_sleep_minutes",
    "sleep_midpoint_minutes",
    "bedtime_variability",
    "nights_tracked_fraction",
    "prior_gpa",
    "term_gpa",
    "gpa_change",
    "term_units",
    "term_load_z"
]


# ============================================================
# 3. LOAD DATASET
# ============================================================

DATA_PATH = "college_sleep_and_gpa.csv"

if not os.path.exists(DATA_PATH):

    st.error(
        f"Dataset not found at: {DATA_PATH}"
    )

    st.stop()

data = pd.read_csv(DATA_PATH)

st.sidebar.success(
    "Dataset loaded successfully"
)


# ============================================================
# 4. LOAD BEST MODEL FROM STEP 5
# ============================================================

if not os.path.exists(RESULTS_PATH):

    st.error(
        "Step 5 results file not found: "
        f"{RESULTS_PATH}"
    )

    st.stop()


comparison = pd.read_csv(
    RESULTS_PATH
)


# Determine the best model based on the average
# of the six required evaluation metrics
metric_columns = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]


# Calculate overall mean if not already present
comparison["Overall Mean"] = comparison[
    metric_columns
].mean(axis=1)


best_model_row = comparison.loc[
    comparison["Overall Mean"].idxmax()
]


best_model_name = best_model_row["Model"]


# Map model names to saved model files
MODEL_FILES = {
    "Logistic Regression":
        "logistic_regression.pkl",

    "Decision Tree":
        "decision_tree.pkl",

    "KNN":
        "knn.pkl",

    "Naive Bayes":
        "naive_bayes.pkl",

    "Random Forest":
        "random_forest.pkl"
}


if best_model_name not in MODEL_FILES:

    st.error(
        f"Saved model file is not configured for "
        f"{best_model_name}."
    )

    st.stop()


MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    MODEL_FILES[best_model_name]
)


if not os.path.exists(MODEL_PATH):

    st.error(
        f"Trained model not found: {MODEL_PATH}"
    )

    st.stop()


model = joblib.load(MODEL_PATH)


# ============================================================
# 5. APPLICATION HEADER
# ============================================================

st.title("😴 College Sleep Prediction")

st.markdown(
    """
    ### Predicting Under-6-Hour Sleep

    This application uses a machine learning classification
    model to predict whether a college student belongs to
    the **under-6-hours sleep category** based on the
    student's academic, demographic, workload, and sleep
    related information.
    """
)


st.divider()


# ============================================================
# 6. SIDEBAR - MODEL INFORMATION
# ============================================================

st.sidebar.title("Model Information")

st.sidebar.write(
    f"**Selected Model:** {best_model_name}"
)

st.sidebar.write(
    "**Problem Type:** Binary Classification"
)

st.sidebar.write(
    "**Target:** under_6h_sleep"
)

st.sidebar.write(
    "**Class 0:** Not Under 6 Hours"
)

st.sidebar.write(
    "**Class 1:** Under 6 Hours"
)


# Display model performance
st.sidebar.subheader("Model Performance")

for metric in metric_columns:

    score = best_model_row[metric]

    st.sidebar.write(
        f"**{metric}:** {score:.4f}"
    )


# ============================================================
# 7. STUDENT / ACADEMIC INFORMATION
# ============================================================

st.header("Student and Academic Information")


col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Study
# ------------------------------------------------------------

with col1:

    study = st.number_input(
        "Study",
        min_value=float(data["study"].min()),
        max_value=float(data["study"].max()),
        value=float(data["study"].median()),
        step=0.1
    )


# ------------------------------------------------------------
# University
# ------------------------------------------------------------

with col2:

    university_values = sorted(
        data["university"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    university = st.selectbox(
        "University",
        university_values
    )


# ------------------------------------------------------------
# Semester
# ------------------------------------------------------------

with col3:

    semester_values = sorted(
        data["semester"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    semester = st.selectbox(
        "Semester",
        semester_values
    )


# ------------------------------------------------------------
# Cohort Code
# ------------------------------------------------------------

with col4:

    cohort_values = sorted(
        data["cohort_code"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    cohort_code = st.selectbox(
        "Cohort Code",
        cohort_values
    )


# ------------------------------------------------------------
# Gender
# ------------------------------------------------------------

with col1:

    gender_values = sorted(
        data["gender"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    gender = st.selectbox(
        "Gender",
        gender_values
    )


# ------------------------------------------------------------
# First Generation
# ------------------------------------------------------------

with col2:

    first_generation = st.selectbox(
        "First Generation",
        sorted(
            data["first_generation"]
            .dropna()
            .unique()
            .tolist()
        )
    )


# ------------------------------------------------------------
# Underrepresented
# ------------------------------------------------------------

with col3:

    underrepresented = st.selectbox(
        "Underrepresented",
        sorted(
            data["underrepresented"]
            .dropna()
            .unique()
            .tolist()
        )
    )


# ------------------------------------------------------------
# Term Units
# ------------------------------------------------------------

with col4:

    term_units = st.number_input(
        "Term Units",
        min_value=float(
            data["term_units"].min()
        ),
        max_value=float(
            data["term_units"].max()
        ),
        value=float(
            data["term_units"].median()
        ),
        step=1.0
    )


# ============================================================
# 8. SLEEP INFORMATION
# ============================================================

st.header("Sleep Information")


col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Average Sleep Minutes
# ------------------------------------------------------------

with col1:

    avg_sleep_minutes = st.number_input(
        "Average Sleep Minutes",
        min_value=float(
            data["avg_sleep_minutes"].min()
        ),
        max_value=float(
            data["avg_sleep_minutes"].max()
        ),
        value=float(
            data["avg_sleep_minutes"].median()
        ),
        step=1.0
    )


# ------------------------------------------------------------
# Average Sleep Hours
# ------------------------------------------------------------

with col2:

    avg_sleep_hours = st.number_input(
        "Average Sleep Hours",
        min_value=float(
            data["avg_sleep_hours"].min()
        ),
        max_value=float(
            data["avg_sleep_hours"].max()
        ),
        value=float(
            data["avg_sleep_hours"].median()
        ),
        step=0.1
    )


# ------------------------------------------------------------
# Daytime Sleep Minutes
# ------------------------------------------------------------

with col3:

    daytime_sleep_minutes = st.number_input(
        "Daytime Sleep Minutes",
        min_value=float(
            data["daytime_sleep_minutes"].min()
        ),
        max_value=float(
            data["daytime_sleep_minutes"].max()
        ),
        value=float(
            data["daytime_sleep_minutes"].median()
        ),
        step=1.0
    )


# ------------------------------------------------------------
# Sleep Midpoint Minutes
# ------------------------------------------------------------

with col4:

    sleep_midpoint_minutes = st.number_input(
        "Sleep Midpoint Minutes",
        min_value=float(
            data["sleep_midpoint_minutes"].min()
        ),
        max_value=float(
            data["sleep_midpoint_minutes"].max()
        ),
        value=float(
            data["sleep_midpoint_minutes"].median()
        ),
        step=1.0
    )


col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Bedtime Variability
# ------------------------------------------------------------

with col1:

    bedtime_variability = st.number_input(
        "Bedtime Variability",
        min_value=float(
            data["bedtime_variability"].min()
        ),
        max_value=float(
            data["bedtime_variability"].max()
        ),
        value=float(
            data["bedtime_variability"].median()
        ),
        step=0.1
    )


# ------------------------------------------------------------
# Nights Tracked Fraction
# ------------------------------------------------------------

with col2:

    nights_tracked_fraction = st.number_input(
        "Nights Tracked Fraction",
        min_value=float(
            data["nights_tracked_fraction"].min()
        ),
        max_value=float(
            data["nights_tracked_fraction"].max()
        ),
        value=float(
            data["nights_tracked_fraction"].median()
        ),
        step=0.01
    )


# ------------------------------------------------------------
# Prior GPA
# ------------------------------------------------------------

with col3:

    prior_gpa = st.number_input(
        "Prior GPA",
        min_value=float(
            data["prior_gpa"].min()
        ),
        max_value=float(
            data["prior_gpa"].max()
        ),
        value=float(
            data["prior_gpa"].median()
        ),
        step=0.01
    )


# ------------------------------------------------------------
# Term GPA
# ------------------------------------------------------------

with col4:

    term_gpa = st.number_input(
        "Term GPA",
        min_value=float(
            data["term_gpa"].min()
        ),
        max_value=float(
            data["term_gpa"].max()
        ),
        value=float(
            data["term_gpa"].median()
        ),
        step=0.01
    )


# ============================================================
# 9. ACADEMIC PERFORMANCE INFORMATION
# ============================================================

st.header("Academic Performance")


col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# GPA Change
# ------------------------------------------------------------

with col1:

    gpa_change = st.number_input(
        "GPA Change",
        min_value=float(
            data["gpa_change"].min()
        ),
        max_value=float(
            data["gpa_change"].max()
        ),
        value=float(
            data["gpa_change"].median()
        ),
        step=0.01
    )


# ------------------------------------------------------------
# Term Load Z
# ------------------------------------------------------------

with col2:

    term_load_z = st.number_input(
        "Term Load Z",
        min_value=float(
            data["term_load_z"].min()
        ),
        max_value=float(
            data["term_load_z"].max()
        ),
        value=float(
            data["term_load_z"].median()
        ),
        step=0.01
    )


# ============================================================
# 10. CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "study": [study],

    "university": [university],

    "semester": [semester],

    "cohort_code": [cohort_code],

    "gender": [gender],

    "first_generation": [first_generation],

    "underrepresented": [underrepresented],

    "avg_sleep_minutes": [
        avg_sleep_minutes
    ],

    "avg_sleep_hours": [
        avg_sleep_hours
    ],

    "daytime_sleep_minutes": [
        daytime_sleep_minutes
    ],

    "sleep_midpoint_minutes": [
        sleep_midpoint_minutes
    ],

    "bedtime_variability": [
        bedtime_variability
    ],

    "nights_tracked_fraction": [
        nights_tracked_fraction
    ],

    "prior_gpa": [
        prior_gpa
    ],

    "term_gpa": [
        term_gpa
    ],

    "gpa_change": [
        gpa_change
    ],

    "term_units": [
        term_units
    ],

    "term_load_z": [
        term_load_z
    ]
})


# Ensure exact feature order
input_data = input_data[
    FEATURE_COLUMNS
]


# ============================================================
# 11. DISPLAY INPUT SUMMARY
# ============================================================

with st.expander("View Input Data"):

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ============================================================
# 12. PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Sleep Category",
    type="primary",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        input_data
    )[0]


    probability_not_under_6 = (
        probabilities[0] * 100
    )


    probability_under_6 = (
        probabilities[1] * 100
    )


    confidence = (
        probabilities[int(prediction)] * 100
    )


    # ========================================================
    # 13. DISPLAY PREDICTION
    # ========================================================

    st.header("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Prediction: Under 6 Hours of Sleep"
        )

        interpretation = (
            "The model predicts that this student "
            "is likely to belong to the under-6-hours "
            "sleep category."
        )

    else:

        st.success(
            "✅ Prediction: Not Under 6 Hours of Sleep"
        )

        interpretation = (
            "The model predicts that this student "
            "is unlikely to belong to the under-6-hours "
            "sleep category."
        )


    # ========================================================
    # 14. CONFIDENCE
    # ========================================================

    st.subheader("Prediction Confidence")


    st.metric(
        label="Model Confidence",
        value=f"{confidence:.2f}%"
    )


    # ========================================================
    # 15. PROBABILITY BREAKDOWN
    # ========================================================

    st.subheader("Prediction Probability")


    probability_col1, probability_col2 = st.columns(2)


    with probability_col1:

        st.metric(
            "Not Under 6 Hours",
            f"{probability_not_under_6:.2f}%"
        )


    with probability_col2:

        st.metric(
            "Under 6 Hours",
            f"{probability_under_6:.2f}%"
        )


    probability_df = pd.DataFrame({

        "Sleep Category": [
            "Not Under 6 Hours",
            "Under 6 Hours"
        ],

        "Probability (%)": [
            probability_not_under_6,
            probability_under_6
        ]
    })


    st.bar_chart(
        probability_df.set_index(
            "Sleep Category"
        )
    )


    # ========================================================
    # 16. INTERPRETATION
    # ========================================================

    st.subheader("Interpretation")


    if prediction == 1:

        st.warning(
            interpretation
            + " The predicted probability for this "
            "category is "
            + f"{probability_under_6:.2f}%."
        )

    else:

        st.info(
            interpretation
            + " The predicted probability for this "
            "category is "
            + f"{probability_not_under_6:.2f}%."
        )


    # ========================================================
    # 17. MODEL INFORMATION
    # ========================================================

    st.caption(
        f"Prediction generated using the "
        f"{best_model_name} model."
    )


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.caption(
    "College Sleep and GPA Classification Project"
)
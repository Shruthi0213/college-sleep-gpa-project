# ============================================================
# COLLEGE SLEEP AND GPA PREDICTION
# STREAMLIT APPLICATION
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
# 2. FILE PATHS
# ============================================================

DATA_PATH = "test_data.csv"

MODEL_FOLDER = "model"

RESULTS_PATH = "model_comparison_results.csv"


# ============================================================
# 3. FEATURES REQUIRED BY THE TRAINED MODEL
# ============================================================

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
    "sleep_midpoint_clock",
    "sleep_bracket",
    "bedtime_variability",
    "nights_tracked_fraction",
    "prior_gpa",
    "term_gpa",
    "gpa_change",
    "term_units",
    "term_load_z"
]


# ============================================================
# 4. MODEL FILES
# ============================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}


# ============================================================
# 5. LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):

    st.error(
        f"Dataset not found: {DATA_PATH}"
    )

    st.stop()


data = pd.read_csv(DATA_PATH)


st.sidebar.success(
    "Dataset loaded successfully"
)


# ============================================================
# 6. CHECK DATASET COLUMNS
# ============================================================

missing_columns = [
    column
    for column in FEATURE_COLUMNS
    if column not in data.columns
]


if missing_columns:

    st.error(
        "The test dataset is missing these required columns: "
        f"{missing_columns}"
    )

    st.stop()


# ============================================================
# 7. LOAD MODEL COMPARISON RESULTS
# ============================================================

if not os.path.exists(RESULTS_PATH):

    st.error(
        f"Model comparison file not found: {RESULTS_PATH}"
    )

    st.stop()


comparison = pd.read_csv(
    RESULTS_PATH
)


# ============================================================
# 8. MODEL METRICS
# ============================================================

# Your CSV uses:
# ML Model Name, Accuracy, AUC, Precision, Recall, F1, MCC

metric_columns = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]


required_result_columns = [
    "ML Model Name"
] + metric_columns


missing_result_columns = [
    column
    for column in required_result_columns
    if column not in comparison.columns
]


if missing_result_columns:

    st.error(
        "The model comparison file is missing these columns: "
        f"{missing_result_columns}"
    )

    st.stop()


# ============================================================
# 9. FIND BEST MODEL
# ============================================================

comparison["Overall Mean"] = comparison[
    metric_columns
].mean(axis=1)


best_model_row = comparison.loc[
    comparison["Overall Mean"].idxmax()
]


best_model_name = best_model_row[
    "ML Model Name"
]


if best_model_name not in MODEL_FILES:

    st.error(
        f"Model '{best_model_name}' is not configured."
    )

    st.stop()


# ============================================================
# 10. LOAD BEST MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    MODEL_FILES[best_model_name]
)


if not os.path.exists(MODEL_PATH):

    st.error(
        f"Trained model not found: {MODEL_PATH}"
    )

    st.stop()


try:

    model = joblib.load(
        MODEL_PATH
    )

except Exception as error:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# 11. APPLICATION HEADER
# ============================================================

st.title(
    "😴 College Sleep Prediction"
)


st.markdown(
    """
    ### Predicting Under-6-Hour Sleep

    This application uses a machine learning classification
    model to predict whether a college student belongs to the
    **under-6-hours sleep category** based on academic,
    demographic, workload, and sleep-related information.
    """
)


st.divider()


# ============================================================
# 12. SIDEBAR MODEL INFORMATION
# ============================================================

st.sidebar.title(
    "Model Information"
)


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


# ============================================================
# 13. MODEL PERFORMANCE
# ============================================================

st.sidebar.subheader(
    "Model Performance"
)


for metric in metric_columns:

    score = best_model_row[metric]

    st.sidebar.write(
        f"**{metric}:** {score:.4f}"
    )


# ============================================================
# 14. STUDENT AND ACADEMIC INFORMATION
# ============================================================

st.header(
    "Student and Academic Information"
)


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

    first_generation_values = sorted(
        data["first_generation"]
        .dropna()
        .unique()
        .tolist()
    )


    first_generation = st.selectbox(
        "First Generation",
        first_generation_values
    )


# ------------------------------------------------------------
# Underrepresented
# ------------------------------------------------------------

with col3:

    underrepresented_values = sorted(
        data["underrepresented"]
        .dropna()
        .unique()
        .tolist()
    )


    underrepresented = st.selectbox(
        "Underrepresented",
        underrepresented_values
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
# 15. SLEEP INFORMATION
# ============================================================

st.header(
    "Sleep Information"
)


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


# ============================================================
# 16. ADDITIONAL SLEEP FEATURES
# ============================================================

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Sleep Midpoint Clock
# ------------------------------------------------------------

with col1:

    sleep_midpoint_clock_values = sorted(
        data["sleep_midpoint_clock"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    sleep_midpoint_clock = st.selectbox(
        "Sleep Midpoint Clock",
        sleep_midpoint_clock_values
    )


# ------------------------------------------------------------
# Sleep Bracket
# ------------------------------------------------------------

with col2:

    sleep_bracket_values = sorted(
        data["sleep_bracket"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    sleep_bracket = st.selectbox(
        "Sleep Bracket",
        sleep_bracket_values
    )


# ------------------------------------------------------------
# Bedtime Variability
# ------------------------------------------------------------

with col3:

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

with col4:

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


# ============================================================
# 17. GPA INFORMATION
# ============================================================

st.header(
    "Academic Performance"
)


col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Prior GPA
# ------------------------------------------------------------

with col1:

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

with col2:

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


# ------------------------------------------------------------
# GPA Change
# ------------------------------------------------------------

with col3:

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

with col4:

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
# 18. CREATE INPUT DATAFRAME
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

    "sleep_midpoint_clock": [
        sleep_midpoint_clock
    ],

    "sleep_bracket": [
        sleep_bracket
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


# ============================================================
# 19. ENSURE CORRECT FEATURE ORDER
# ============================================================

input_data = input_data[
    FEATURE_COLUMNS
]


# ============================================================
# 20. VIEW INPUT DATA
# ============================================================

with st.expander(
    "View Input Data"
):

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ============================================================
# 21. PREDICTION BUTTON
# ============================================================

st.divider()


predict_button = st.button(
    "🔮 Predict Sleep Category",
    type="primary",
    use_container_width=True
)


# ============================================================
# 22. GENERATE PREDICTION
# ============================================================

if predict_button:

    try:

        # IMPORTANT:
        # The saved model already contains the preprocessing
        # pipeline. Therefore, input_data is passed directly.

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

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
                probabilities[
                    int(prediction)
                ] * 100
            )

        else:

            if int(prediction) == 1:

                probability_not_under_6 = 0.0
                probability_under_6 = 100.0

            else:

                probability_not_under_6 = 100.0
                probability_under_6 = 0.0


            confidence = 100.0


        # ====================================================
        # 23. DISPLAY RESULT
        # ====================================================

        st.header(
            "Prediction Result"
        )


        if int(prediction) == 1:

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


        # ====================================================
        # 24. CONFIDENCE
        # ====================================================

        st.subheader(
            "Prediction Confidence"
        )


        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        # ====================================================
        # 25. PROBABILITY BREAKDOWN
        # ====================================================

        st.subheader(
            "Prediction Probability"
        )


        probability_col1, probability_col2 = (
            st.columns(2)
        )


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


        # ====================================================
        # 26. INTERPRETATION
        # ====================================================

        st.subheader(
            "Interpretation"
        )


        if int(prediction) == 1:

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


        # ====================================================
        # 27. MODEL INFORMATION
        # ====================================================

        st.caption(
            f"Prediction generated using the "
            f"{best_model_name} model."
        )


    except Exception as error:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(
            error
        )


# ============================================================
# 28. FOOTER
# ============================================================

st.divider()


st.caption(
    "College Sleep and GPA Classification Project"
)
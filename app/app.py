
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"


st.set_page_config(
    page_title="Student Mental Health Prediction",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


st.sidebar.title("🎓 Student Mental Health")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Student Prediction"]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Educational Project")


if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


if page == "Dashboard":

    st.title("Student Mental Health Prediction")
    st.caption("Logistic Regression — Model Performance Dashboard")

    st.warning(
        "Educational prototype only. Predictions are not medical "
        "diagnoses or validated clinical risk assessments."
    )

    if not METRICS_PATH.exists():
        st.info("Export metrics.json from the notebook first.")
        st.stop()

    metrics = load_metrics()

    st.subheader("Overall Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}"
    )

    col2.metric(
        "Precision Macro",
        f"{metrics['precision']:.2%}"
    )

    col3.metric(
        "Recall Macro",
        f"{metrics['recall']:.2%}"
    )

    col4.metric(
        "F1 Macro",
        f"{metrics['f1_macro']:.2%}"
    )

    st.divider()

    st.subheader("Overall Metrics Comparison")

    overall_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Macro"
        ],
        "Score": [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_macro"]
        ]
    })

    st.bar_chart(
        overall_df.set_index("Metric"),
        y="Score",
        horizontal=True,
        x_label="Score",
        color="#3283CE"
    )

    st.divider()

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=metrics["labels"],
        yticklabels=metrics["labels"],
        ax=ax
    )

    ax.set_title("Logistic Regression Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    if "classification_report" in metrics:

        st.subheader("Classification Performance by Class")

        report = metrics["classification_report"]
        classes = metrics["labels"]

        class_metrics = pd.DataFrame({
            "Class": classes,
            "Precision": [
                report[c]["precision"] for c in classes
            ],
            "Recall": [
                report[c]["recall"] for c in classes
            ],
            "F1 Score": [
                report[c]["f1-score"] for c in classes
            ]
        })

        st.bar_chart(
            class_metrics.set_index("Class"),
            y=["Precision", "Recall", "F1 Score"]
        )

        st.dataframe(
            class_metrics,
            hide_index=True,
            use_container_width=True
        )

    else:
        st.info(
            "Classification chart unavailable. "
            "Export classification_report into metrics.json."
        )

    st.divider()

    if "roc_curves" in metrics:

        st.subheader("ROC Curves — One vs Rest")

        fig, ax = plt.subplots(figsize=(9, 6))

        for class_name, values in metrics["roc_curves"].items():

            ax.plot(
                values["fpr"],
                values["tpr"],
                linewidth=2,
                label=f"{class_name} (AUC = {values['auc']:.3f})"
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Random Classifier"
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Multiclass ROC Curves")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        roc_results = pd.DataFrame([
            {
                "Class": class_name,
                "ROC-AUC": values["auc"]
            }
            for class_name, values in metrics["roc_curves"].items()
        ])

        st.dataframe(
            roc_results,
            hide_index=True,
            use_container_width=True
        )

    else:
        st.info(
            "ROC curves unavailable. "
            "Export roc_curves into metrics.json."
        )

    st.divider()

    st.subheader("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Algorithm:** Logistic Regression")
        st.write("**Task:** Multiclass Classification")
        st.write("**Classes:** High, Low, Medium")

    with col2:
        st.write("**Preprocessing:** Full Pipeline")
        st.write("**Evaluation Metric:** F1 Macro")
        st.write("**Evaluation Dataset:** Independent Test Set")


elif page == "Student Prediction":

    st.title("Student Prediction")

    st.write(
        "Enter student information to demonstrate "
        "the trained machine learning model."
    )

    st.warning(
        "This prediction is for educational demonstration only. "
        "It must not be used for diagnosis, screening, "
        "admission, discipline, or decisions about a student."
    )

    with st.form("student_form"):

        st.subheader("Student Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=17,
                max_value=29,
                value=21
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Non-binary"]
            )

            year_of_study = st.selectbox(
                "Year of Study",
                [1, 2, 3, 4, 5]
            )

            sleep_hours = st.number_input(
                "Sleep Hours",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.1
            )

            study_hours = st.number_input(
                "Study Hours Per Week",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=0.5
            )

            gpa = st.number_input(
                "GPA",
                min_value=0.0,
                max_value=4.0,
                value=3.0,
                step=0.01
            )

            social_media_hours = st.number_input(
                "Social Media Hours",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.1
            )

            screen_time_hours = st.number_input(
                "Screen Time Hours",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.1
            )

        with col2:

            physical_activity = st.selectbox(
                "Physical Activity",
                ["Low", "Moderate", "High"]
            )

            financial_stress = st.slider(
                "Financial Stress",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )

            family_support = st.slider(
                "Family Support",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )

            attendance_rate = st.slider(
                "Attendance Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=0.5
            )

            part_time_job = st.selectbox(
                "Part Time Job",
                ["No", "Yes"]
            )

            diet_quality = st.selectbox(
                "Diet Quality",
                ["Poor", "Average", "Good"]
            )

            mental_health_history = st.selectbox(
                "Previous Mental Health History",
                ["No", "Yes"]
            )

        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True
        )

    if submitted:

        if social_media_hours > screen_time_hours:

            st.error(
                "Social Media Hours cannot exceed Screen Time Hours."
            )

        else:

            student_data = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Year_of_Study": year_of_study,
                "Sleep_Hours": sleep_hours,
                "Study_Hours_Per_Week": study_hours,
                "GPA": gpa,
                "Social_Media_Hours": social_media_hours,
                "Screen_Time_Hours": screen_time_hours,
                "Physical_Activity": physical_activity,
                "Financial_Stress": financial_stress,
                "Family_Support": family_support,
                "Attendance_Rate": attendance_rate,
                "Part_Time_Job": part_time_job,
                "Diet_Quality": diet_quality,
                "Previous_Mental_Health_History": mental_health_history
            }])

            if hasattr(model, "feature_names_in_"):
                student_data = student_data[
                    list(model.feature_names_in_)
                ]

            prediction = model.predict(student_data)[0]

            probabilities = model.predict_proba(student_data)[0]

            st.divider()

            st.subheader("Prediction Result")

            st.metric(
                "Predicted Class",
                prediction
            )

            st.subheader("Class Probabilities")

            probability_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probabilities
            })

            st.bar_chart(
                probability_df.set_index("Class"),
                y="Probability"
            )

            probability_df["Probability (%)"] = (
                probability_df["Probability"] * 100
            ).round(2)

            st.dataframe(
                probability_df[
                    ["Class", "Probability (%)"]
                ],
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Probabilities are model outputs, "
                "not validated clinical risk estimates."
            )

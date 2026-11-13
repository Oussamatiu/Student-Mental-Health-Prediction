
from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


TARGET = "Mental_Health_Risk"


def load_data(csv_path):
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    print(f"Dataset loaded: {df.shape}")

    return str(path)


def validate_and_clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    required_columns = [
        "Student_ID",
        "Age",
        "Gender",
        "Year_of_Study",
        "Sleep_Hours",
        "Study_Hours_Per_Week",
        "GPA",
        "Social_Media_Hours",
        "Screen_Time_Hours",
        "Physical_Activity",
        "Financial_Stress",
        "Family_Support",
        "Attendance_Rate",
        "Part_Time_Job",
        "Diet_Quality",
        "Previous_Mental_Health_History",
        TARGET
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values")

    invalid_rows = (
        df["Social_Media_Hours"].notna()
        & (df["Social_Media_Hours"] > df["Screen_Time_Hours"])
    )

    print("Invalid rows:", invalid_rows.sum())

    df = df.loc[~invalid_rows].copy()

    df = df.drop_duplicates()

    if df.empty:
        raise ValueError("No valid data remaining")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("Cleaned dataset:", df.shape)

    return str(output_path)


def split_data(input_path, train_path, test_path):
    df = pd.read_csv(input_path)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET]
    )

    train_path = Path(train_path)
    test_path = Path(test_path)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Train:", train_df.shape)
    print("Test:", test_df.shape)

    return str(train_path), str(test_path)


def build_preprocessor(X):
    numerical_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    nominal_features = [
        "Gender",
        "Part_Time_Job",
        "Previous_Mental_Health_History"
    ]

    ordinal_features = [
        "Physical_Activity",
        "Diet_Quality"
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                ["Low", "Moderate", "High"],
                ["Poor", "Average", "Good"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("nom", nominal_pipeline, nominal_features),
        ("ord", ordinal_pipeline, ordinal_features)
    ])

    return preprocessor


def train_model(train_path, model_path, C=1.0):
    df = pd.read_csv(train_path)

    X_train = df.drop(columns=["Student_ID", TARGET])
    y_train = df[TARGET]

    preprocessor = build_preprocessor(X_train)

    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=42
        ))
    ])

    model_pipeline.fit(X_train, y_train)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, model_path)

    print("Model trained and saved:", model_path)

    return str(model_path)


def evaluate_model(model_path, test_path, metrics_path):
    model = joblib.load(model_path)

    df = pd.read_csv(test_path)

    X_test = df.drop(columns=["Student_ID", TARGET])
    y_test = df[TARGET]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    classes = list(model.classes_)

    roc_data = {}

    for i, class_name in enumerate(classes):
        y_binary = (y_test == class_name).astype(int)

        fpr, tpr, _ = roc_curve(
            y_binary,
            probabilities[:, i]
        )

        auc = roc_auc_score(
            y_binary,
            probabilities[:, i]
        )

        roc_data[class_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc)
        }

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, average="macro"
        )),
        "recall": float(recall_score(
            y_test, predictions, average="macro"
        )),
        "f1_macro": float(f1_score(
            y_test, predictions, average="macro"
        )),
        "roc_auc": float(roc_auc_score(
            y_test,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="macro"
        )),
        "labels": classes,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=classes
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True
        ),
        "roc_curves": roc_data
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Accuracy:", metrics["accuracy"])
    print("F1 Macro:", metrics["f1_macro"])
    print("ROC-AUC:", metrics["roc_auc"])

    return str(metrics_path)


def save_artifacts(model_path, metrics_path, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    if metrics["f1_macro"] < 0.70:
        raise ValueError("Model did not pass the F1 quality threshold")

    shutil.copyfile(
        model_path,
        output_dir / "logistic_regression.joblib"
    )

    shutil.copyfile(
        metrics_path,
        output_dir / "metrics.json"
    )

    print("Model and metrics saved successfully")

    return str(output_dir)

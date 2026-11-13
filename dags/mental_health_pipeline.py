
from datetime import datetime, timedelta
from airflow.decorators import dag, task

from src.ml_pipeline import (
    load_data,
    validate_and_clean_data,
    split_data,
    build_preprocessor,
    train_model,
    evaluate_model,
    save_artifacts
)

BASE = "/opt/airflow"


@dag(
    dag_id="student_mental_health",
    start_date=datetime(2026, 11, 12),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2)
    }
)
def mental_health_dag():

    @task
    def prepare_data():
        raw = load_data(
            f"{BASE}/data/raw/student_mental_health.csv"
        )

        clean = validate_and_clean_data(
            raw,
            f"{BASE}/data/processed/clean.csv"
        )

        return split_data(
            clean,
            f"{BASE}/data/processed/train.csv",
            f"{BASE}/data/processed/test.csv"
        )

    @task
    def preprocess_data(paths):
        import pandas as pd

        train_df = pd.read_csv(paths[0])

        X_train = train_df.drop(
            columns=["Student_ID", "Mental_Health_Risk"]
        )

        preprocessor = build_preprocessor(X_train)

        print("Preprocessing pipeline created")

        return paths

    @task
    def train(paths):
        return train_model(
            paths[0],
            f"{BASE}/artifacts/model.joblib",
            C=1.0
        )

    @task
    def evaluate(model_path):
        metrics_path = evaluate_model(
            model_path,
            f"{BASE}/data/processed/test.csv",
            f"{BASE}/artifacts/metrics.json"
        )

        save_artifacts(
            model_path,
            metrics_path,
            f"{BASE}/models"
        )

        return metrics_path

    paths = prepare_data()
    processed = preprocess_data(paths)
    model = train(processed)
    evaluate(model)


mental_health_dag()

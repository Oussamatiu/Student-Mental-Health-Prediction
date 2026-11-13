# Student Mental Health Risk Prediction

An end-to-end **educational machine learning project** that explores student-related factors and predicts one of three dataset-defined labels: **Low**, **Medium**, or **High**. It includes exploratory data analysis, preprocessing, model comparison, a Streamlit dashboard, and an Apache Airflow workflow.

> **Disclaimer:** This is a learning prototype, **not a medical diagnosis, validated mental-health screening tool, or clinical risk assessment**. It must not be used to make medical, academic, admissions, or disciplinary decisions about students.

## Project overview

The project covers five stages:

1. **Exploratory Data Analysis (EDA):** inspect dataset structure, missing values, duplicates, unusual observations, class balance, and relationships among features.
2. **Data preprocessing:** apply a stratified 80/20 train/test split, missing-value imputation, categorical encoding, and numeric scaling using a scikit-learn pipeline to reduce data leakage.
3. **Model training:** train and compare Logistic Regression, Random Forest, and XGBoost using cross-validation.
4. **Optimization and evaluation:** tune hyperparameters, evaluate on a held-out test set, inspect classification errors, and save the complete fitted pipeline.
5. **Application and orchestration:** display results and demo predictions with Streamlit; orchestrate preparation, preprocessing, training, and evaluation with Airflow.

## Dataset

**Source:** [Mental Health Risk Prediction Dataset – Students (Kaggle)](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)

- Original dataset: **2,000 rows**, **16 input features** and **1 target column** (`Mental_Health_Risk`).
- Target classes: `Low`, `Medium`, `High` (approximately balanced).
- Cleaning decision used in the notebook: **37 records removed** where `Social_Media_Hours > Screen_Time_Hours`, leaving **1,963 rows**. This rule assumes screen time represents total screen use including social media; confirm this interpretation against the dataset documentation before applying it to other data.
- Evaluation split after cleaning: approximately **80% training / 20% test**, with **393 test records**.
- Identified numerical outliers were retained rather than removed automatically.

### Preprocessing

| Feature type | Transformation |
| --- | --- |
| Numerical | Median imputation and `StandardScaler` |
| Nominal categorical | Most-frequent imputation and `OneHotEncoder` |
| Ordinal categorical | Most-frequent imputation and `OrdinalEncoder` |

The saved model contains **both the fitted preprocessing pipeline and the classifier**. No manual encoding or scaling is required when using its `predict` method with the original input feature columns.

## Models and results

The following scores were reported for the independent test split (**393 rows**):

| Model | Accuracy | Macro F1 | Multiclass ROC-AUC (OvR) |
| --- | ---: | ---: | ---: |
| **Logistic Regression** | **0.75** | **0.75** | **0.9047** |
| XGBoost | 0.73 | 0.73 | 0.8832 |
| Random Forest | 0.67 | 0.66 | 0.8627 |

**Selected model:** Logistic Regression, based on the reported test-set comparison. Its per-class F1 scores were approximately **0.80 (High)**, **0.80 (Low)** and **0.63 (Medium)**. The `Medium` class was the most difficult to distinguish.

The reported Logistic Regression confusion matrix uses class order `High`, `Low`, `Medium` for both rows (actual) and columns (predicted):

| Actual \\ Predicted | High | Low | Medium |
| --- | ---: | ---: | ---: |
| High | 105 | 2 | 27 |
| Low | 0 | 103 | 27 |
| Medium | 23 | 21 | 85 |

This corresponds to **293 correct predictions out of 393** (about **74.6%** accuracy). Results depend on the exact data cleaning, split, and saved model version. Do not repeatedly optimize models against the held-out test set.

## Project structure

```text
Student-Mental-Health-Prediction/
├── app/
│   └── app.py
├── dags/
│   └── mental_health_pipeline.py
├── src/
│   └── ml_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── logistic_regression.joblib
│   └── metrics.json
├── artifacts/
├── notebooks/
│   └── mental_health_prediction.ipynb
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

*This is the intended layout; some generated directories or artifacts may only appear after you run the notebook or Airflow workflow.*

## Run locally

### Prerequisites

- Python with dependencies such as `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `matplotlib`, `seaborn`, `streamlit`, and `joblib`.
- Docker Desktop with the Linux engine enabled for the Airflow setup.
- The Kaggle dataset downloaded into `data/raw/`.

### Install dependencies

From the repository root:

```bash
python -m pip install -r requirements.txt
```

Make sure `requirements.txt` contains the libraries imported by the notebook and app. The Airflow Docker image also needs the packages imported by `src/ml_pipeline.py`.

### Explore and train

Open `notebooks/mental_health_prediction.ipynb` in VS Code or Jupyter. Run its cells to perform EDA, compare models and export:

- `models/logistic_regression.joblib` — full fitted pipeline;
- `models/metrics.json` — test-set metrics, confusion matrix, per-class classification report and ROC curve data.

### Run the Streamlit app

```bash
python -m streamlit run app/app.py
```

Open the local URL printed in your terminal (normally `http://localhost:8501`). The app has:

- **Dashboard:** evaluation metrics, confusion matrix, ROC curves and per-class charts.
- **Student Prediction:** an input form for all 15 features and model-generated class probabilities.

The model and metric files must exist in `models/` before starting the dashboard.

### Run Apache Airflow

From the project root, with Docker Desktop running:

```bash
docker compose up -d --build
docker compose ps
```

Open the Airflow web UI at the port configured in `docker-compose.yml` (commonly `http://localhost:8080`). Find and manually trigger the DAG named **`student_mental_health`**.

The intended **four-task workflow** is:

```text
prepare_data -> preprocess_data -> train_model -> evaluate_model
```

The DAG is configured for a **weekly schedule**, with **two task retries**. Follow execution in Airflow's Graph/Grid view and inspect Task Logs for failures. In this implementation, preprocessing is fitted as part of the full training pipeline rather than being fitted on the entire dataset in advance.

For command-line diagnostics (when your Compose service is named `airflow-scheduler`):

```bash
docker compose exec airflow-scheduler airflow dags list-import-errors
docker compose exec airflow-scheduler airflow dags list
docker compose exec airflow-scheduler airflow dags trigger student_mental_health
```

The Compose configuration must mount `dags/`, `src/`, `data/`, `artifacts/` and `models/` into the Airflow containers and make `src` importable.

To stop services:

```bash
docker compose down
```

## Tech stack

**Python**, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, XGBoost, joblib, Streamlit, Apache Airflow, PostgreSQL, and Docker Compose.

## Limitations and responsible use

- Dataset labels and predicted probabilities are **not validated clinical judgments**.
- Dataset-specific relationships may not generalize to other student populations.
- `Medium` was the weakest class in the reported evaluation.
- The demo should use fictional or appropriately consented data, and should avoid collecting or publishing sensitive personal information.
- Repeated automated training on an unchanged dataset does not necessarily improve the model. Deployment of updated artifacts should be gated by appropriate validation rather than automatic test-set tuning.

## Presentation

[Project presentation on Canva](https://canva.link/gg6tg43l4q1tohr)

## Author

**Oussama Chentouf** — YouCode, AI Development.

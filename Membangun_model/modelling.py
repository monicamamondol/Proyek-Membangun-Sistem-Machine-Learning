import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Ganti nama eksperimen
mlflow.set_experiment("Breast Cancer")

# Load data dari file CSV yang terpisah
x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

input_example = x_train.iloc[0:5]

with mlflow.start_run():
    # Aktifkan autolog
    mlflow.autolog()

    # Inisialisasi model dengan parameter tetap
    n_estimators = 505
    max_depth = 37
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # Logging manual model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    # Latih model
    model.fit(x_train, y_train)

    # Hitung akurasi dan log
    accuracy = model.score(x_test, y_test)
    mlflow.log_metric("accuracy", accuracy)


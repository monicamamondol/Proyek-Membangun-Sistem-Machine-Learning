import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Ganti nama eksperimen
mlflow.set_experiment("Breast Cancer")

# Load data dari file CSV yang terpisah
x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

input_example = x_train.iloc[0:5]

# Mendifinisikan model menggunakan hyperparameter tuning.
# Define Elastic Search parameters
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)  # 5 evenly spaced values
max_depth_range = np.linspace(1, 50, 5, dtype=int)  # 5 evenly spaced values

best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"elastic_search_{n_estimators}_{max_depth}"):
            # Logging manual
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(x_train, y_train)

            accuracy = model.score(x_test, y_test)
            mlflow.log_metric("accuracy", accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example
                )


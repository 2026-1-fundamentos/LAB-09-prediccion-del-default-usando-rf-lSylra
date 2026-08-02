# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
# flake8: noqa: E501

# flake8: noqa: E501
# flake8: noqa: E501

import os
import gzip
import json
import pickle
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, balanced_accuracy_score,
    recall_score, f1_score, confusion_matrix
)

# ---------------------------
# Utilidades de carga
# ---------------------------

def load_csv_zip(path):
    """Carga un archivo CSV comprimido."""
    return pd.read_csv(path)


def unify_education_levels(value):
    """Agrupa niveles superiores de educación en la categoría 4 (others)."""
    return value if value <= 4 else 4


def prepare_dataset(frame):
    """Limpieza general del dataset según los pasos requeridos."""
    df = frame.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.dropna()
    df["EDUCATION"] = df["EDUCATION"].apply(unify_education_levels)
    return df


# ---------------------------
# Carga y preparación
# ---------------------------

training_raw = load_csv_zip("files/input/train_data.csv.zip")
testing_raw = load_csv_zip("files/input/test_data.csv.zip")

training = prepare_dataset(training_raw)
testing = prepare_dataset(testing_raw)

X_tr = training.drop(columns=["default"])
y_tr = training["default"]

X_te = testing.drop(columns=["default"])
y_te = testing["default"]


# ---------------------------
# Pipeline
# ---------------------------

def build_pipeline(cat_cols):
    """Crea el pipeline de transformación + modelo."""
    transformer = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )

    model = RandomForestClassifier(random_state=42)

    return Pipeline([
        ("transform", transformer),
        ("model", model)
    ])


categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
pipeline = build_pipeline(categorical_cols)


# ---------------------------
# Grid Search
# ---------------------------

def run_grid_search(pipe, x, y):
    """Optimización de hiperparámetros con validación cruzada."""
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    search.fit(x, y)
    return search


best_estimator = run_grid_search(pipeline, X_tr, y_tr)


# ---------------------------
# Guardado del modelo gzip
# ---------------------------

def save_gzip_model(model, filepath):
    """Guarda el modelo comprimido en formato gzip."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, "wb") as f:
        pickle.dump(model, f)


save_gzip_model(best_estimator, "files/models/model.pkl.gz")


# ---------------------------
# Métricas
# ---------------------------

def build_metric(model, x, y, dataset_name):
    preds = model.predict(x)
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, preds, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, preds),
        "recall": recall_score(y, preds, zero_division=0),
        "f1_score": f1_score(y, preds, zero_division=0)
    }


def build_confusion_dict(model, x, y, dataset_name):
    preds = model.predict(x)
    cm = confusion_matrix(y, preds)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


results = [
    build_metric(best_estimator, X_tr, y_tr, "train"),
    build_metric(best_estimator, X_te, y_te, "test"),
    build_confusion_dict(best_estimator, X_tr, y_tr, "train"),
    build_confusion_dict(best_estimator, X_te, y_te, "test"),
]


# ---------------------------
# Guardado de métricas
# ---------------------------

def write_metrics_to_file(entries, out_path):
    """Escribe cada métrica como una línea JSON independiente."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


write_metrics_to_file(results, "files/output/metrics.json")
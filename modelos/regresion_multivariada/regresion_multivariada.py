#!/usr/bin/env python3
"""Clasificador de Regresión Multivariada (Logística Multinomial) para Vocales.
Responsable: Carlos Pinto.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

# Agregar raíz del repositorio a sys.path
RAIZ_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ_REPO))

from src import cargar_datos, evaluar_clasificador

def main():
    print("1. Cargando datos preparados (PCA 95% de varianza)...")
    X_train, y_train, X_test, y_test, grupos_train = cargar_datos(espacio="pca")
    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")

    # Configuración de validación cruzada agrupada (sin fuga)
    cv_agrupado = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Definición de la grilla de hiperparámetros
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [500]
    }

    modelo_base = LogisticRegression(random_state=42)

    print("\n2. Ejecutando búsqueda de hiperparámetros con GridSearchCV...")
    grid = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        cv=cv_agrupado.split(X_train, y_train, grupos_train),
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    print("\n3. Mejores resultados de validación cruzada:")
    print("   Mejores hiperparámetros:", grid.best_params_)
    print(f"   Mejor Accuracy CV:      {grid.best_score_*100:.2f}%")

    print("\n4. Evaluación final sobre el conjunto de prueba (test set):")
    mejor_modelo = grid.best_estimator_
    resultados = evaluar_clasificador(
        modelo=mejor_modelo,
        X_test=X_test,
        y_test=y_test,
        nombre_modelo="Regresión Logística Multinomial"
    )

if __name__ == "__main__":
    main()

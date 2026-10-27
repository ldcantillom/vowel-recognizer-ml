#!/usr/bin/env python3
"""Clasificador Random Forest para Reconocimiento de Vocales.
Responsable: Jorge García.
"""

import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

# Agregar raíz del repositorio a sys.path
RAIZ_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ_REPO))

from src import cargar_datos, evaluar_clasificador

def main():
    print("1. Cargando datos preparados (PCA 95% de varianza)...")
    X_train, y_train, X_test, y_test, grupos_train = cargar_datos(espacio="pca")
    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")

    # Configuración de validación cruzada agrupada
    cv_agrupado = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Grilla de hiperparámetros para Random Forest
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 15, 25],
        "min_samples_split": [2, 5],
        "criterion": ["gini", "entropy"]
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    print("\n2. Ejecutando búsqueda de hiperparámetros con GridSearchCV...")
    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv_agrupado.split(X_train, y_train, grupos_train),
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    print("\n3. Resultados de validación cruzada:")
    print("   Mejores hiperparámetros:", grid.best_params_)
    print(f"   Mejor Accuracy CV:      {grid.best_score_*100:.2f}%")

    print("\n4. Evaluación sobre el conjunto de prueba (test set):")
    mejor_modelo = grid.best_estimator_
    resultados = evaluar_clasificador(
        modelo=mejor_modelo,
        X_test=X_test,
        y_test=y_test,
        nombre_modelo="Random Forest"
    )

if __name__ == "__main__":
    main()

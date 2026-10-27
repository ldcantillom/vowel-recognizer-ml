"""Módulo de evaluación estandarizada de modelos de clasificación supervisada.
Genera métricas homogéneas (Accuracy, Balanced Accuracy, Precision, Recall, F1)
y matriz de confusión para cumplir con los ítems 5 y 6 de la Guía del trabajo.
"""

from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from .carga_datos import VOCALES

def evaluar_clasificador(
    modelo: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    nombre_modelo: str = "Clasificador",
    mostrar_grafica: bool = True
) -> Dict[str, Any]:
    """Evalúa un modelo entrenado sobre el conjunto de prueba y genera métricas y matriz de confusión.
    
    Retorna un diccionario con las métricas calculadas para la tabla comparativa final.
    """
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"==================================================")
    print(f"  RESULTADOS DE EVALUACIÓN: {nombre_modelo}")
    print(f"==================================================")
    print(f"  Accuracy (Exactitud):         {acc*100:.2f}%")
    print(f"  Balanced Accuracy:            {bacc*100:.2f}%")
    print(f"  Precisión Macro:              {prec*100:.2f}%")
    print(f"  Exhaustividad (Recall) Macro: {rec*100:.2f}%")
    print(f"  F1-Score Macro:               {f1*100:.2f}%")
    print(f"--------------------------------------------------")
    print("\nReporte detallado por clase:")
    print(classification_report(y_test, y_pred, labels=VOCALES, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=VOCALES)

    if mostrar_grafica:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=VOCALES)
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        ax.set_title(f"Matriz de Confusión — {nombre_modelo}", fontsize=11)
        plt.tight_layout()
        plt.show()

    return {
        "Modelo": nombre_modelo,
        "Accuracy": acc,
        "Balanced_Accuracy": bacc,
        "Precision_Macro": prec,
        "Recall_Macro": rec,
        "F1_Macro": f1,
        "y_pred": y_pred,
        "matriz_confusion": cm
    }

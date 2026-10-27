#!/usr/bin/env python3
"""Script para generar y exportar las matrices de datos preparadas (X, y, PCA, grupos).

Uso:
    python generar_datos.py
"""

import sys
from pathlib import Path

# Agregar directorio actual al sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocesamiento import preparar_y_guardar_datos, RUTA_CACHE_POR_DEFECTO

if __name__ == "__main__":
    print("==========================================================")
    print("Generación de Datos Preparados para el Reconocedor de Vocales")
    print("==========================================================")
    ruta_guardada = preparar_y_guardar_datos()
    print(f"\n¡Datos listos en: {ruta_guardada}!")
    print("Todos los miembros del equipo ya pueden importar los datos con:")
    print("    from src import cargar_datos")
    print("    X_train, y_train, X_test, y_test, grupos_train = cargar_datos(espacio='pca')")

# Random Forest (Bosques Aleatorios)

**Responsable:** Jorge García  
**Asignatura:** Inteligencia Artificial — Universidad del Magdalena

---

## 🎯 Objetivo
Entrenar y evaluar un ensamble de árboles de decisión mediante **Random Forest** para clasificar las imágenes de vocales (`A`, `E`, `I`, `O`, `U`).

## 📁 Archivos
- `random_forest.py`: Script de ajuste de hiperparámetros con `GridSearchCV`.
- `random_forest.ipynb`: Cuaderno Jupyter de experimentación.

## 🚀 Cómo ejecutar
```bash
python random_forest.py
```

## 🔍 Hiperparámetros a explorar con `GridSearchCV`
- `n_estimators`: Número de árboles en el bosque (ej. `[50, 100, 200]`).
- `max_depth`: Profundidad máxima de los árboles (ej. `[None, 10, 20, 30]`).
- `min_samples_split`: Mínimo de muestras para dividir un nodo (ej. `[2, 5, 10]`).
- `criterion`: Criterio de división (`'gini'`, `'entropy'`).

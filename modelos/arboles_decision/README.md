# Árboles de Decisión (CART / C4.5)

**Responsable:** Santiago Gonzales  
**Asignatura:** Inteligencia Artificial — Universidad del Magdalena

---

## 🎯 Objetivo
Entrenar y evaluar un clasificador individual de **Árbol de Decisión** para predecir las cinco vocales (`A`, `E`, `I`, `O`, `U`), analizando la interpretabilidad y el riesgo de sobreajuste.

## 📁 Archivos
- `arbol_decision.py`: Script de ajuste de hiperparámetros con `GridSearchCV`.
- `arbol_decision.ipynb`: Cuaderno Jupyter de experimentación.

## 🚀 Cómo ejecutar
```bash
python arbol_decision.py
```

## 🔍 Hiperparámetros a explorar con `GridSearchCV`
- `max_depth`: Profundidad máxima del árbol (ej. `[5, 10, 15, 20, None]`).
- `min_samples_split`: Muestras mínimas para bifurcar (ej. `[2, 5, 10, 20]`).
- `min_samples_leaf`: Muestras mínimas en hojas (ej. `[1, 2, 4]`).
- `criterion`: Criterio de impureza (`'gini'`, `'entropy'`).

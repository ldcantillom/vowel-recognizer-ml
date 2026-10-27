# Regresión Multivariada (Logística Multinomial / Softmax)

**Responsable:** Carlos Pinto  
**Asignatura:** Inteligencia Artificial — Universidad del Magdalena

---

## 🎯 Objetivo
Entrenar y evaluar un clasificador multiclase basado en **Regresión Logística Multinomial** (regresión multivariada para clasificación) para predecir las cinco vocales (`A`, `E`, `I`, `O`, `U`).

## 📁 Archivos
- `regresion_multivariada.py`: Script base de experimentación con `GridSearchCV`.
- `regresion_multivariada.ipynb`: Cuaderno Jupyter interactivo.

## 🚀 Cómo ejecutar
```bash
python regresion_multivariada.py
```

## 🔍 Hiperparámetros a explorar con `GridSearchCV`
- `C`: Inverso de la fuerza de regularización (ej. `[0.01, 0.1, 1.0, 10.0]`).
- `penalty`: Norma de penalización (`'l2'`, `'l1'`).
- `solver`: Algoritmo de optimización (`'lbfgs'`, `'saga'`).
- `max_iter`: Número de iteraciones (ej. `[500, 1000]`).

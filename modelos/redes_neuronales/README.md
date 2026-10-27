# Redes Neuronales (MLP y DNN)

**Responsable:** León Cantillo  
**Asignatura:** Inteligencia Artificial — Universidad del Magdalena

---

## 🎯 Objetivo
Diseñar, entrenar y evaluar modelos de redes neuronales artificiales:
1. **Perceptrón Multicapa (MLP)** mediante `sklearn.neural_network.MLPClassifier`.
2. **Red Neuronal Profunda (DNN)** con múltiples capas densas y regularización.

## 📁 Archivos
- `redes_neuronales.py`: Script de ajuste de hiperparámetros con `GridSearchCV`.
- `redes_neuronales.ipynb`: Cuaderno Jupyter de experimentación y curvas de aprendizaje.

## 🚀 Cómo ejecutar
```bash
python redes_neuronales.py
```

## 🔍 Hiperparámetros a explorar con `GridSearchCV`
- `hidden_layer_sizes`: Arquitectura de capas ocultas (ej. `[(128,), (128, 64), (256, 128)]`).
- `activation`: Función de activación (`'relu'`, `'tanh'`).
- `alpha`: Término de regularización $L_2$ (ej. `[0.0001, 0.001, 0.01]`).
- `learning_rate_init`: Tasa de aprendizaje inicial (ej. `[0.001, 0.01]`).
- `max_iter`: Épocas máximas de entrenamiento (ej. `[300, 500]`).

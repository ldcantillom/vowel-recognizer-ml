# Módulos de Clasificación Supervisada por Integrante

En este directorio cada integrante del equipo cuenta con su propio espacio de trabajo aislado para implementar, ajustar con `GridSearchCV` y evaluar el método asignado, evitando conflictos de Git y manteniendo el proyecto modular.

---

## 👥 Reparto de Métodos y Asignación

| Carpeta | Integrante | Método Asignado |
| :--- | :--- | :--- |
| [`regresion_multivariada/`](./regresion_multivariada/) | **Carlos Pinto** | Regresión Multivariada (Logística Multinomial / Softmax) |
| [`random_forest/`](./random_forest/) | **Jorge García** | Random Forest (Bosques Aleatorios) |
| [`arboles_decision/`](./arboles_decision/) | **Santiago Gonzales** | Árboles de Decisión (CART / C4.5) |
| [`redes_neuronales/`](./redes_neuronales/) | **León Cantillo** | Redes Neuronales (MLP y DNN) |

---

## 🚀 ¿Cómo importar los datos en tu cuaderno o script?

No necesitas preprocesar las imágenes ni volver a calcular PCA. Todo el pipeline de homogenización, escala de grises, normalización, partición sin fuga y PCA está empaquetado en el módulo [`src`](../src/).

### En tu script o notebook:

```python
import sys
from pathlib import Path

# Permite importar desde la raíz del proyecto
sys.path.append(str(Path.cwd().parent.parent))

from src import cargar_datos, evaluar_clasificador, ver_vocal

# 1. Cargar datos listos en el espacio reducido de PCA (135 componentes, 95% de varianza)
X_train, y_train, X_test, y_test, grupos_train = cargar_datos(espacio="pca")

# (Opcional) Si tu método requiere el espacio de píxeles original (7.500 variables normalizadas):
# X_train, y_train, X_test, y_test, grupos_train = cargar_datos(espacio="original")

print("Datos listos:")
print("  Entrenamiento:", X_train.shape, y_train.shape)
print("  Prueba:       ", X_test.shape, y_test.shape)
```

---

## ⚙️ Configuración obligatoria de `GridSearchCV` con Validación Cruzada Agrupada

Para cumplir con la **Rúbrica de Evaluación** y evitar optimismo artificial (*data leakage* debido a data augmentation):

```python
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

# Particionador agrupado por trazo base
cv_agrupado = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Definir la búsqueda de hiperparámetros
grid = GridSearchCV(
    estimator=tu_modelo,
    param_grid=parametros,
    cv=cv_agrupado.split(X_train, y_train, grupos_train),
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True
)

grid.fit(X_train, y_train)
```

---

## 📊 Evaluación Final Estandarizada

Para generar automáticamente las métricas del literal 5 de la guía (Accuracy, Balanced Accuracy, Precision Macro, Recall Macro, F1 Macro y Matriz de Confusión):

```python
# Evaluar tu mejor modelo sobre el conjunto de prueba
resultados = evaluar_clasificador(
    modelo=grid.best_estimator_,
    X_test=X_test,
    y_test=y_test,
    nombre_modelo="Nombre de Tu Modelo"
)
```

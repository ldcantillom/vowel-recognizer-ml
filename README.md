# vowel-recognizer-ml

Trabajo experimental de Machine Learning para la asignatura de **Inteligencia Artificial** de la **Universidad del Magdalena** (IA-2026-II).

**Integrantes y Asignación de Métodos:**
- **Carlos Pinto:** Regresión Multivariada (Logística Multinomial) — [`modelos/regresion_multivariada/`](modelos/regresion_multivariada/)
- **Jorge García:** Random Forest — [`modelos/random_forest/`](modelos/random_forest/)
- **Santiago Gonzales:** Árboles de Decisión — [`modelos/arboles_decision/`](modelos/arboles_decision/)
- **León Cantillo:** Redes Neuronales (MLP y DNN) — [`modelos/redes_neuronales/`](modelos/redes_neuronales/)

---

## 🎯 Tema del Proyecto
Reconocimiento óptico de caracteres (OCR) aplicado a vocales manuscritas: **A, E, I, O, U**.  
La vocal **O** se representa con la carpeta **`0`** (por similitud visual y ausencia de letra O aislada en el dataset original).

---

## 📂 Estructura Modular del Repositorio

Para evitar conflictos de Git y archivos monolíticos gigantes, el repositorio se organiza de forma modular:

```text
vowel-recognizer-ml/
├── src/                               # 📦 Módulo común de datos y evaluación
│   ├── __init__.py
│   ├── carga_datos.py                 # Lectura y homogenización de imágenes (75x100)
│   ├── preprocesamiento.py            # Normalización [0, 1], partición y PCA (95% varianza)
│   └── evaluacion.py                  # Evaluación con métricas estándar y matriz de confusión
├── modelos/                           # 🚀 Espacios de trabajo individuales por integrante
│   ├── README.md                      # Instrucciones de uso para el equipo
│   ├── regresion_multivariada/        # Carlos Pinto
│   ├── random_forest/                 # Jorge García
│   ├── arboles_decision/              # Santiago Gonzales
│   └── redes_neuronales/              # León Cantillo
├── data_processed/                    # Caché local binario de matrices (vocales_preparadas.npz)
├── generar_datos.py                   # Script ejecutable para reconstruir datos en 1 comando
├── Taller_Vocales_ML.ipynb            # Cuaderno unificado de inspección y preprocesamiento base
└── README.md
```

---

## ⚡ Guía Rápida para los Integrantes: ¿Cómo importar los datos?

Cualquier integrante puede cargar instantáneamente los datos listos para entrenar su modelo sin necesidad de preprocesar imágenes manualmente:

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent))  # Para scripts dentro de modelos/<metodo>/

from src import cargar_datos, evaluar_clasificador

# Opción recomendada: Espacio reducido con PCA (135 componentes, 95% de varianza explicada)
X_train, y_train, X_test, y_test, grupos_train = cargar_datos(espacio="pca")

# Opción alternativa: Espacio original completo (7.500 píxeles normalizados [0.0, 1.0])
# X_train, y_train, X_test, y_test, grupos_train = cargar_datos(espacio="original")
```

### Configuración obligatoria en `GridSearchCV`:
Para respetar la regla contra la fuga de información (*data leakage* por aumento de datos):

```python
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

cv_agrupado = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=tu_modelo,
    param_grid=param_grid,
    cv=cv_agrupado.split(X_train, y_train, grupos_train),
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)

# Evaluación sobre el conjunto de prueba
evaluar_clasificador(grid.best_estimator_, X_test, y_test, nombre_modelo="Mi Modelo")
```

---

## 📥 Descarga del Dataset Original
El conjunto de datos completo se puede descargar desde Kaggle:  
https://www.kaggle.com/datasets/straighttoheaven/handwriting-letter-and-number-dataset

> `CNN letter Dataset/` y archivos `.zip` o `.npz` están excluidos mediante `.gitignore` para no sobrecargar el repositorio.

---

## 📜 Licencia
Uso académico — Inteligencia Artificial, Universidad del Magdalena.

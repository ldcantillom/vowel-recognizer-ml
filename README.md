# vowel-recognizer-ml

Trabajo experimental de Machine Learning para la asignatura de **Inteligencia Artificial** de la **Universidad del Magdalena** (IA-2026-II).

**Integrantes:** Carlos Pinto, León Cantillo, Santiago Gonzales, Jorge García.

## Tema

Reconocimiento óptico de caracteres (OCR) aplicado a vocales manuscritas: **A, E, I, O, U**. La vocal **O** se representa con la carpeta **`0`** (no existe un directorio propio para la letra O, por similitud visual con el dígito cero).

## Contenido del notebook

`Taller_Vocales_ML.ipynb` contiene el flujo completo del trabajo:

- Descripción del problema, variables, número de casos y objetivo.
- Inspección del dataset: conteos por clase, muestras, dimensiones, calidad, duplicados, imagen promedio, histogramas y matriz de correlación.
- **Preprocesamiento:** conversión de cada imagen a blanco y negro con el método de Otsu, guardado en `dataset/blanco y negro/` (carpeta derivada, no versionada).
- **Partición agrupada** por identificador de aumento (`aug<id>_<sufijo>`) para evitar fuga de información entre entrenamiento y prueba.
- **Árbol de decisión** ajustado con `GridSearchCV` y validación cruzada agrupada de 5 pliegues.
- Evaluación con accuracy, balanced accuracy, precisión, recall, F1 y matrices de confusión.
- Comparación entre la línea base, el árbol sin ajuste, el árbol ajustado y una versión reducida a 500 píxeles.
- Conclusiones, limitaciones, trabajo futuro e implicaciones éticas.

## Resultados principales

| Modelo | Accuracy en prueba |
|---|---|
| Línea base (clase mayoritaria) | 20,32 % |
| Árbol sin ajuste | 99,41 % |
| Árbol ajustado con `GridSearchCV` | 99,01 % |
| Árbol ajustado con 500 píxeles | 99,21 % |

El mejor modelo según validación cruzada usa `criterion="entropy"`, `max_depth=10`, `min_samples_leaf=1` y todas las variables. El árbol final tiene 9 niveles, 41 hojas y solo 40 píxeles con importancia mayor que cero.

## Requisitos

- Python 3.11 o superior
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pillow`, `jupyter`

Instalación sugerida:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib pillow jupyter
jupyter notebook Taller_Vocales_ML.ipynb
```

## Descarga del dataset

El conjunto de datos completo se puede descargar desde Kaggle:

https://www.kaggle.com/datasets/straighttoheaven/handwriting-letter-and-number-dataset

Ubicación esperada:

```text
dataset/normal/CNN letter Dataset/{A,E,I,O,U,0,...}/
```

> El dataset y la carpeta `dataset/blanco y negro/` no se incluyen en el repositorio por su tamaño; están excluidos mediante `.gitignore`.

## Notas

- Las imágenes de muestra (`aug*.jpg`) se incluyen únicamente con fines ilustrativos.
- Conteo normalizado por vocal: A: 1010, E: 1010, I: 1010, O: 1030, U: 1010 (total 5070).
- La comparación con regresión multivariada, random forest, MLP y DNN corresponde a la integración final del equipo; esta rama desarrolla árboles de decisión.

## Licencia

Uso académico — Universidad del Magdalena.

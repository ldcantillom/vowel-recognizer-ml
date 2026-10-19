# vowel-recognizer-ml

Trabajo experimental de Machine Learning para la asignatura de **Inteligencia Artificial** de la **Universidad del Magdalena** (IA-2026-II).

**Integrantes:** Carlos Pinto, León Cantillo, Santiago Gonzales.

## Tema

Reconocimiento óptico de caracteres (OCR) aplicado a vocales manuscritas: **A, E, I, O, U**. La vocal **O** se representa con la carpeta **`0`** (no existe directorio propio para la letra O, por similitud visual con el dígito cero).

## Entregable

- `Taller_Vocales_ML.ipynb` — Cuaderno de Jupyter unificado y normalizado con la inspección del dataset de vocales:
  - Configuración y carga del dataset (solo vocales; se excluyen números del 1 al 9 y otras letras).
  - Organización de archivos y carpetas por clase vocal.
  - Cantidad de imágenes por vocal y diagrama de barras.
  - Muestras aleatorias por clase en cuadrícula.
  - Dimensiones, canales, formato y consistencia de tamaño.
  - Calidad de datos: corruptas, duplicados (MD5), contraste, patrón de nombres.
  - Exploración gráfica: imagen promedio por vocal y distribución de intensidades.
  - Resumen cuantitativo y tratamientos propuestos.

## Descarga del dataset

El conjunto de datos completo se puede descargar desde Kaggle:

https://www.kaggle.com/datasets/straighttoheaven/handwriting-letter-and-number-dataset

> El dataset no se incluye en este repositorio por su tamaño (`CNN letter Dataset/` y `archive (1).zip` están excluidos mediante `.gitignore`).

## Notas

- Las imágenes de muestra (`aug*.jpg`) se incluyen únicamente con fines ilustrativos.
- Conteo normalizado por vocal: A: 1010, E: 1010, I: 1010, O: 1030, U: 1010 (total 5070).

## Licencia

Uso académico — Universidad del Magdalena.
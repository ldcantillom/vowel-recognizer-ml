## Taller formativo – Trabajo de la unidad ML Inteligencia Artificial IA-2026-II

El objetivo de esta actividad es inspeccionar y verificar la calidad del dataset asignado, y hacer una primera exploración de las variables, especialmente la variable a predecir. Con esta actividad se espera que cada equipo de trabajo:

- Comprenda el problema y la estructura del dataset asignado.

- Identifique tamaño, tipos de datos, valores faltantes, duplicados y posibles inconsistencias.

- Construya al menos una visualización exploratoria apropiada al tipo de datos.

## I. Tareas comunes para todos los equipos

- 1. Buscar documentación del problema a resolver para entender el problema. Con base en la documentación describa con sus propias palabras en que consiste, de manera contextualizada.

- 2. Cargar el dataset asignado y verifiquen que pueden acceder correctamente a sus archivos o registros.

- 3. Inspeccionar el tamaño del dataset y describir su estructura, las variables con las que cuenta tipo de variables.

- 4. Revisar calidad de datos: valores faltantes, duplicados, valores inválidos o inconsistentes y posibles clases o categorías poco representadas.

- 5. Realizar una primera exploración gráfica adecuada al tipo de datos. No se espera todavía entrenamiento de modelos ni una limpieza completa.

En esta primera etapa se espera que cuantifiquen el problema, muestren ejemplos y justifiquen qué tratamiento podría ser necesario más adelante.

## II. Tareas específicas de acuerdo con cada datase

Cada dataset representa un reto diferente y requerirá tareas específicas de acuerdo con el objetivo y tipo de datos que trataran. Realice las tareas que se listan a continuación según el dataset asignado. En cada sección encontrarán recursos recomendados para operaciones que todavía no necesariamente se han trabajado en clase. No es necesario estudiar los tutoriales completos: consulten únicamente las partes relacionadas con las tareas solicitadas.

## Imágenes de caracteres: vocales

En esta etapa interesa comprender cómo está organizado el conjunto de datos y verificar cantidad de imágenes, dimensiones, formato, calidad visual, distribución de los ejemplos por clase.


- 1. Identifiquen cómo están organizados los archivos y carpetas, y como pueden leer las imágenes.

- 2. Cuenten cuántas imágenes existen para cada vocal y construyan un diagrama de barras con esos conteos.

- 3. Seleccionen varias imágenes aleatorias de cada clase y muéstrenlas en una cuadrícula. Observen variaciones de escritura, fondo, iluminación, grosor de trazo, rotación o ruido.

- 4. Verifiquen dimensiones, número de canales y formato de archivo de una muestra de imágenes. Determinen si todas tienen el mismo tamaño.

Para verificar canales, pueden observar la forma del arreglo de una imagen: una imagen en escala de grises suele tener dos dimensiones, mientras que una imagen de color suele tener 3 dimensiones. Estas herramientas pueden ser de ayuda para realizar estas tareas:

Pillow – tutorial básico — abrir una imagen con Image.open(), consultar su tamaño (size), modo

(mode) y realizar conversiones sencillas.

OpenCV – lectura y visualización de imágenes — leer imágenes con cv2.imread() y comprender cómo se representa una imagen como arreglo.

Matplotlib – imshow() — mostrar imágenes y construir las visualizaciones necesarias para inspeccionarlas.

https://docs.opencv.org/3.4/db/deb/tutorial_display_image.html
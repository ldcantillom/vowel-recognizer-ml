# Resultados — Árbol de decisión

**Método:** árbol de decisión para clasificar las vocales A, E, I, O y U.

**Evidencia:** `Taller_Vocales_ML.ipynb` (secciones 12 y 13).

---

## 1. Datos y preprocesamiento

| Aspecto | Valor |
|---|---:|
| Imágenes de vocales | 5.070 |
| Tamaño original | 75 x 100 píxeles |
| Canales | 1 (escala de grises) |
| Clases | 5 (A, E, I, O, U) |
| Variable de entrada | 7.500 píxeles binarios por imagen |
| Umbral | Otsu calculado por imagen |
| Salida binaria | Tinta = 0, fondo = 255 |

La vocal **O** se toma de la carpeta `0`. Las imágenes binarizadas se guardan en `dataset/blanco y negro/`, que no se versiona.

---

## 2. Diseño del experimento

- **Partición:** 80 % entrenamiento (4.056 imágenes) y 20 % prueba (1.014 imágenes).
- **Agrupamiento:** `StratifiedGroupKFold` por identificador `aug<id>_<sufijo>`, para que variantes del mismo trazo no queden repartidas entre entrenamiento y prueba.
- **Validación cruzada:** 5 pliegues agrupados y estratificados.
- **Búsqueda:** `GridSearchCV` con 32 combinaciones de hiperparámetros.
- **Métrica de selección:** accuracy.
- **Conjunto de prueba:** reservado hasta la evaluación final.

### Rejilla evaluada

| Hiperparámetro | Valores |
|---|---|
| `criterion` | `gini`, `entropy` |
| `max_depth` | 6, 10, 14, sin límite |
| `min_samples_leaf` | 1, 3 |
| `max_features` | todas, `sqrt` |

---

## 3. Mejor modelo

| Aspecto | Valor |
|---|---:|
| `criterion` | `entropy` |
| `max_depth` | 10 |
| `min_samples_leaf` | 1 |
| `max_features` | todas |
| Accuracy de validación cruzada | 99,11 % |
| Accuracy de entrenamiento | 100,00 % |
| Accuracy de prueba | 99,01 % |
| Balanced accuracy | 99,01 % |
| F1 macro | 99,01 % |
| Profundidad real | 9 |
| Hojas | 41 |
| Píxeles con importancia mayor que cero | 40 de 7.500 |

Las profundidades 10, 14 y sin límite empataron en 99,11 % de validación cruzada; `GridSearchCV` conservó la primera combinación empatada y el modelo final quedó con profundidad 10.

---

## 4. Comparación de variantes

| Modelo | Accuracy entrenamiento | Accuracy prueba | Balanced accuracy | F1 macro | Profundidad | Hojas | Tiempo de ajuste |
|---|---:|---:|---:|---:|---:|---:|---:|
| Línea base (clase mayoritaria) | 20,32 % | 20,32 % | 20,00 % | 6,75 % | — | — | 0,0013 s |
| Árbol sin ajuste | 100,00 % | 99,41 % | 99,41 % | 99,41 % | 9 | 40 | 2,20 s |
| Árbol ajustado (7.500 píxeles) | 100,00 % | 99,01 % | 99,01 % | 99,01 % | 9 | 41 | 1,76 s |
| Árbol ajustado (500 píxeles) | 100,00 % | 99,21 % | 99,22 % | 99,21 % | 8 | 38 | 0,08 s |

La diferencia entre el árbol sin ajuste y el ajustado es de 4 imágenes de 1.014; está dentro del error de muestreo y el modelo ajustado se eligió por validación cruzada, sin usar el conjunto de prueba.

La versión reducida a 20 x 25 píxeles (500 variables) conserva un desempeño similar y reduce el tiempo de ajuste aproximadamente 21 veces.

---

## 5. Matriz de confusión

Filas: clase real. Columnas: clase predicha.

| Real \ Predicha | A | E | I | O | U | Total |
|---|---:|---:|---:|---:|---:|---:|
| A | 200 | 0 | 2 | 0 | 0 | 202 |
| E | 0 | 202 | 0 | 0 | 0 | 202 |
| I | 0 | 0 | 202 | 0 | 0 | 202 |
| O | 0 | 0 | 0 | 204 | 2 | 206 |
| U | 0 | 0 | 2 | 4 | 196 | 202 |
| **Total** | 200 | 202 | 206 | 208 | 198 | 1.014 |

**Errores (10 en total):**

- 2 casos de **A** predichos como **I**.
- 2 casos de **O** predichos como **U**.
- 2 casos de **U** predichos como **I**.
- 4 casos de **U** predichos como **O**.

**Recall por clase:** A 99,01 %, E 100,00 %, I 100,00 %, O 99,03 %, U 97,03 %.

---

## 6. Píxeles más importantes

| Fila | Columna | Importancia |
|---:|---:|---:|
| 38 | 38 | 0,3528 |
| 51 | 37 | 0,2416 |
| 14 | 39 | 0,0881 |
| 57 | 38 | 0,0809 |
| 66 | 37 | 0,0405 |

El árbol usa solo **40 píxeles** de los 7.500. Las reglas se concentran en la zona central de la imagen, donde se diferencian la abertura de la A, los trazos de la E, la barra de la I, el contorno de la O y la forma abierta de la U.

---

## 7. Interpretación

- El árbol supera a la línea base por más de 78 puntos porcentuales; aprende la forma de las vocales y no depende del desbalance de clases.
- La binarización con Otsu eliminó la variación de iluminación y permitió que reglas simples separaran las clases.
- Los errores se concentran en vocales de forma parecida: I, O y U.
- El accuracy es alto porque el dataset es relativamente sencillo: imágenes centradas, trazo grueso y fondo uniforme. No debe interpretarse como desempeño garantizado en escritura real.
- El modelo es interpretable: 9 niveles, 41 hojas y 40 píxeles relevantes.

---

## 8. Limitaciones

- La **O** se aproxima con el dígito **0**.
- El dataset no cubre todos los estilos, edades, culturas ni condiciones motrices.
- La evaluación usa una sola partición agrupada; no se hizo validación por escritor.
- El árbol es inestable ante pequeños cambios en los datos y no aprovecha relaciones espaciales complejas.
- No se comparó todavía con regresión multivariada, random forest, MLP ni DNN; esa comparación corresponde a la integración final del equipo.

---

## 9. Reproducción

1. Descargar el dataset y ubicarlo en `dataset/normal/CNN letter Dataset/`.
2. Ejecutar `Taller_Vocales_ML.ipynb` de principio a fin.
3. El notebook regenera `dataset/blanco y negro/` y calcula todos los resultados con semilla 42.

Los tiempos pueden variar según la máquina; las métricas no, porque la partición y los modelos usan semillas fijas.

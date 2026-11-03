"""Árbol de decisión para clasificar las vocales A, E, I, O y U.

Reproduce el experimento del notebook Taller_Vocales_ML.ipynb:
binarización con Otsu, partición agrupada y GridSearchCV.

Uso:
    python "algoritmos/arboles de decision/arbol_decision.py"
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

VOCALES = ["A", "E", "I", "O", "U"]
CARPETAS = {"0": "O", "A": "A", "E": "E", "I": "I", "U": "U"}
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

RAIZ_REPO = Path(__file__).resolve().parents[2]
DATASET_POR_DEFECTO = RAIZ_REPO / "dataset" / "normal" / "CNN letter Dataset"
SALIDA_BN_POR_DEFECTO = RAIZ_REPO / "dataset" / "blanco y negro"


def umbral_otsu(imagen: np.ndarray) -> int:
    """Calcula el umbral de Otsu de una imagen en escala de grises."""
    histograma = np.bincount(imagen.ravel(), minlength=256).astype(np.float64)
    probabilidad = histograma / histograma.sum()
    omega = np.cumsum(probabilidad)
    media_acumulada = np.cumsum(probabilidad * np.arange(256))
    media_total = media_acumulada[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        varianza_entre = (media_total * omega - media_acumulada) ** 2 / (omega * (1 - omega))
    varianza_entre[~np.isfinite(varianza_entre)] = -1
    return int(np.argmax(varianza_entre))


def binarizar(ruta: Path) -> tuple[np.ndarray, int]:
    """Devuelve la imagen binaria (tinta=0, fondo=255) y el umbral usado."""
    with Image.open(ruta) as imagen:
        gris = np.asarray(imagen.convert("L"))
    umbral = umbral_otsu(gris)
    return np.where(gris > umbral, 255, 0).astype(np.uint8), umbral


def resolver_dataset(ruta: Path | None) -> Path:
    """Busca el dataset en las ubicaciones habituales."""
    candidatos = [
        ruta,
        DATASET_POR_DEFECTO,
        Path("dataset/normal/CNN letter Dataset"),
    ]
    for candidato in candidatos:
        if candidato is not None and candidato.is_dir():
            if (candidato / "A").is_dir() and (candidato / "0").is_dir():
                return candidato
    raise FileNotFoundError(
        "No se encontró el dataset. Use --dataset o ubíquelo en "
        "dataset/normal/CNN letter Dataset/."
    )


def listar_imagenes(raiz: Path) -> pd.DataFrame:
    """Construye la tabla de imágenes de vocales."""
    registros = []
    for carpeta, clase in CARPETAS.items():
        directorio = raiz / carpeta
        if not directorio.is_dir():
            raise FileNotFoundError(f"No existe la carpeta {directorio}")
        for ruta in sorted(directorio.iterdir()):
            if ruta.suffix.lower() in EXTENSIONES:
                registros.append({"ruta_original": ruta, "clase": clase})
    return pd.DataFrame(registros)


def generar_binarizadas(df: pd.DataFrame, salida: Path) -> pd.DataFrame:
    """Convierte las imágenes a blanco y negro y las guarda en disco."""
    salida.mkdir(parents=True, exist_ok=True)
    registros = []
    for fila in df.itertuples(index=False):
        binaria, umbral = binarizar(fila.ruta_original)
        destino = salida / fila.clase / f"{Path(fila.ruta_original).stem}.png"
        destino.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binaria).save(destino)
        registros.append(
            {
                "ruta_original": fila.ruta_original,
                "ruta_bn": destino,
                "clase": fila.clase,
                "umbral": umbral,
            }
        )
    return pd.DataFrame(registros)


def cargar_matriz(df_bn: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga los píxeles binarios y construye X, y y los grupos."""
    X = np.stack(
        [
            (np.asarray(Image.open(ruta)) == 0).astype(np.uint8).ravel()
            for ruta in df_bn["ruta_bn"]
        ]
    )
    y = df_bn["clase"].to_numpy()
    base = df_bn["ruta_original"].apply(
        lambda ruta: re.match(r"aug(\d+)_", Path(ruta).stem).group(1)
    )
    grupos = (df_bn["clase"] + "_" + base).to_numpy()
    return X, y, grupos


def partir_datos(
    X: np.ndarray, y: np.ndarray, grupos: np.ndarray, semilla: int
) -> tuple[np.ndarray, ...]:
    """Separa entrenamiento y prueba respetando los grupos de aumento."""
    particionador = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)
    idx_entrenamiento, idx_prueba = next(particionador.split(X, y, grupos))
    return (
        X[idx_entrenamiento],
        X[idx_prueba],
        y[idx_entrenamiento],
        y[idx_prueba],
        grupos[idx_entrenamiento],
        grupos[idx_prueba],
    )


def buscar_arbol(
    X_ent: np.ndarray,
    y_ent: np.ndarray,
    g_ent: np.ndarray,
    semilla: int,
    n_jobs: int,
) -> GridSearchCV:
    """Ajusta un árbol de decisión con GridSearchCV."""
    parametros = {
        "criterion": ["gini", "entropy"],
        "max_depth": [6, 10, 14, None],
        "min_samples_leaf": [1, 3],
        "max_features": [None, "sqrt"],
    }
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)
    busqueda = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=semilla),
        param_grid=parametros,
        scoring="accuracy",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        refit=True,
    )
    busqueda.fit(X_ent, y_ent, groups=g_ent)
    return busqueda


def reducir_binarizada(
    matriz: np.ndarray, ancho: int = 20, alto: int = 25
) -> np.ndarray:
    """Reduce cada imagen binaria a una cuadrícula de ancho x alto."""
    return np.stack(
        [
            (
                np.asarray(
                    Image.fromarray(fila.reshape(100, 75) * 255).resize(
                        (ancho, alto), Image.Resampling.BOX
                    )
                )
                > 127
            ).astype(np.uint8).ravel()
            for fila in matriz
        ]
    )


def evaluar(nombre: str, modelo, X_ent, X_prueba, y_ent, y_prueba) -> dict:
    """Ajusta y evalúa un modelo sobre el conjunto de prueba."""
    inicio = time.time()
    modelo.fit(X_ent, y_ent)
    duracion = time.time() - inicio
    prediccion = modelo.predict(X_prueba)
    return {
        "Modelo": nombre,
        "Accuracy entrenamiento": accuracy_score(y_ent, modelo.predict(X_ent)),
        "Accuracy prueba": accuracy_score(y_prueba, prediccion),
        "Balanced accuracy": balanced_accuracy_score(y_prueba, prediccion),
        "F1 macro": f1_score(y_prueba, prediccion, average="macro"),
        "Profundidad": getattr(modelo, "get_depth", lambda: None)(),
        "Hojas": getattr(modelo, "get_n_leaves", lambda: None)(),
        "Tiempo (s)": duracion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Árbol de decisión para vocales.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--salida-bn", type=Path, default=SALIDA_BN_POR_DEFECTO)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    raiz = resolver_dataset(args.dataset)
    print(f"Dataset: {raiz}")

    df = listar_imagenes(raiz)
    print(f"Imágenes de vocales: {len(df)}")

    inicio = time.time()
    df_bn = generar_binarizadas(df, args.salida_bn)
    print(f"Binarización: {len(df_bn)} imágenes en {time.time() - inicio:.1f} s")
    print(f"Salida: {args.salida_bn}")

    X, y, grupos = cargar_matriz(df_bn)
    X_ent, X_prueba, y_ent, y_prueba, g_ent, g_prueba = partir_datos(
        X, y, grupos, args.semilla
    )
    print(f"Entrenamiento: {len(y_ent)} | Prueba: {len(y_prueba)}")
    print(f"Grupos compartidos: {len(set(g_ent) & set(g_prueba))}")

    inicio = time.time()
    busqueda = buscar_arbol(X_ent, y_ent, g_ent, args.semilla, args.n_jobs)
    print(f"Búsqueda: {time.time() - inicio:.1f} s")
    print("Mejores hiperparámetros:")
    for clave, valor in busqueda.best_params_.items():
        print(f"  {clave}: {valor}")
    print(f"Accuracy de validación cruzada: {busqueda.best_score_:.4f}")

    modelo = busqueda.best_estimator_
    prediccion = modelo.predict(X_prueba)
    print(f"Accuracy de entrenamiento: {accuracy_score(y_ent, modelo.predict(X_ent)):.4f}")
    print(f"Accuracy de prueba: {accuracy_score(y_prueba, prediccion):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_prueba, prediccion):.4f}")
    print(f"Profundidad: {modelo.get_depth()} | Hojas: {modelo.get_n_leaves()}")
    print("\nReporte por clase:")
    print(classification_report(y_prueba, prediccion, labels=VOCALES, digits=4))
    print("Matriz de confusión:")
    print(confusion_matrix(y_prueba, prediccion, labels=VOCALES))

    filas = [
        evaluar(
            "Línea base (clase mayoritaria)",
            DummyClassifier(strategy="most_frequent"),
            X_ent,
            X_prueba,
            y_ent,
            y_prueba,
        ),
        evaluar(
            "Árbol sin ajuste",
            DecisionTreeClassifier(random_state=args.semilla),
            X_ent,
            X_prueba,
            y_ent,
            y_prueba,
        ),
        evaluar(
            "Árbol ajustado (7.500 píxeles)",
            DecisionTreeClassifier(random_state=args.semilla, **busqueda.best_params_),
            X_ent,
            X_prueba,
            y_ent,
            y_prueba,
        ),
        evaluar(
            "Árbol ajustado (500 píxeles)",
            DecisionTreeClassifier(random_state=args.semilla, **busqueda.best_params_),
            reducir_binarizada(X_ent),
            reducir_binarizada(X_prueba),
            y_ent,
            y_prueba,
        ),
    ]
    comparacion = pd.DataFrame(filas)
    print("\nComparación:")
    print(comparacion.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

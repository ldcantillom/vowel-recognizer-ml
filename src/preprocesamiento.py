"""Módulo de preprocesamiento, escalado, partición agrupada y reducción con PCA.
Provee funciones listas para ser consumidas por los modelos de cada integrante.
"""

from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA

from .carga_datos import leer_imagenes_vocales, ANCHO, ALTO, VOCALES

# Ruta por defecto del archivo binario comprimido con datos preparados
RUTA_CACHE_POR_DEFECTO = Path(__file__).resolve().parent.parent / "data_processed" / "vocales_preparadas.npz"

def preparar_y_guardar_datos(
    ruta_salida: Union[str, Path] = RUTA_CACHE_POR_DEFECTO,
    n_splits: int = 5,
    random_state: int = 42,
    varianza_pca: float = 0.95
) -> Path:
    """Ejecuta el pipeline completo de preparación y guarda los arreglos comprimidos en disco.
    
    Ajusta la normalización, la partición agrupada sin fuga de información
    y el Análisis de Componentes Principales (PCA) al 95% de varianza explicada.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    print("1/4. Leyendo y homogenizando imágenes del dataset...")
    X_img, y, grupos, _ = leer_imagenes_vocales()

    print(f"2/4. Aplanando y normalizando al rango [0.0, 1.0] ({X_img.shape[0]} imágenes)...")
    # Vectorización: (N, 7500) en float32
    X = (X_img.reshape(len(X_img), -1) / 255.0).astype(np.float32)

    print("3/4. Partición agrupada (StratifiedGroupKFold 80/20)...")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(X, y, grupos))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    grupos_train, grupos_test = grupos[train_idx], grupos[test_idx]

    print(f"4/4. Ajustando PCA ({int(varianza_pca*100)}% de varianza) estrictamente sobre entrenamiento...")
    # PCA exploratorio para seleccionar componentes
    pca_exp = PCA(n_components=300, svd_solver="randomized", random_state=random_state)
    pca_exp.fit(X_train)
    var_acum = np.cumsum(pca_exp.explained_variance_ratio_)
    k_sel = int(np.searchsorted(var_acum, varianza_pca) + 1)

    # PCA definitivo al 95% de varianza
    pca = PCA(n_components=k_sel, svd_solver="randomized", random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)  # Sin fit: estrictamente transform

    print(f"Guardando matrices preparadas en: {ruta_salida}...")
    np.savez_compressed(
        ruta_salida,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        grupos_train=grupos_train,
        grupos_test=grupos_test,
        X_train_pca=X_train_pca,
        X_test_pca=X_test_pca,
        k_pca=np.array([k_sel]),
        ancho=np.array([ANCHO]),
        alto=np.array([ALTO])
    )
    print("✓ Preparación completada con éxito.")
    return ruta_salida

def cargar_datos(
    espacio: str = "pca",
    ruta_cache: Union[str, Path] = RUTA_CACHE_POR_DEFECTO,
    regenerar_si_falta: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    """Carga de forma instantánea los datos listos para el entrenamiento y prueba.
    
    Parámetros:
        espacio: 
            - 'pca': Retorna (X_train_pca, y_train, X_test_pca, y_test, grupos_train) [k=135 columnas]
            - 'original': Retorna (X_train, y_train, X_test, y_test, grupos_train) [7.500 columnas]
            - 'ambos': Retorna un diccionario con todas las matrices y metadatos.
        ruta_cache: Ruta del archivo .npz precomputado.
        regenerar_si_falta: Si True y el archivo no existe, lo construye automáticamente.
        
    Retorna:
        Tupla (X_train, y_train, X_test, y_test, grupos_train) o Diccionario completo.
    """
    ruta_cache = Path(ruta_cache)
    if not ruta_cache.is_file():
        if regenerar_si_falta:
            print(f"Archivo de caché no encontrado en {ruta_cache}. Generando datos preparados...")
            preparar_y_guardar_datos(ruta_salida=ruta_cache)
        else:
            raise FileNotFoundError(f"No existe el archivo de caché en: {ruta_cache}")

    data = np.load(ruta_cache, allow_pickle=True)

    y_train = data["y_train"]
    y_test = data["y_test"]
    grupos_train = data["grupos_train"]

    if espacio == "pca":
        return data["X_train_pca"], y_train, data["X_test_pca"], y_test, grupos_train
    elif espacio == "original":
        return data["X_train"], y_train, data["X_test"], y_test, grupos_train
    elif espacio == "ambos":
        return {
            "X_train": data["X_train"],
            "X_test": data["X_test"],
            "X_train_pca": data["X_train_pca"],
            "X_test_pca": data["X_test_pca"],
            "y_train": y_train,
            "y_test": y_test,
            "grupos_train": grupos_train,
            "grupos_test": data["grupos_test"],
            "k_pca": int(data["k_pca"][0]),
            "ancho": int(data["ancho"][0]),
            "alto": int(data["alto"][0])
        }
    else:
        raise ValueError(f"Espacio no reconocido: '{espacio}'. Use 'pca', 'original' o 'ambos'.")

def ver_vocal(
    vector: np.ndarray,
    etiqueta: Optional[str] = None,
    prediccion: Optional[str] = None,
    titulo: Optional[str] = None,
    ancho: int = ANCHO,
    alto: int = ALTO
) -> None:
    """Reconstruye y visualiza una imagen de vocal a partir de su vector unidimensional de características.
    
    Compatible con vectores originales de 7.500 posiciones o matrices (100, 75).
    """
    if vector.ndim == 1:
        if len(vector) != ancho * alto:
            raise ValueError(f"El vector tiene longitud {len(vector)}, pero se esperaban {ancho*alto} (espacio original).")
        img = vector.reshape((alto, ancho))
    else:
        img = vector

    plt.figure(figsize=(2.6, 3.2))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0 if img.max() <= 1.0 else 255)
    
    subtitulos = []
    if etiqueta is not None:
        subtitulos.append(f"Real: {etiqueta}")
    if prediccion is not None:
        subtitulos.append(f"Pred: {prediccion}")
    
    t = titulo if titulo else (" | ".join(subtitulos) if subtitulos else "Vocal")
    plt.title(t, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

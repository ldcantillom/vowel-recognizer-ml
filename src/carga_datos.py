"""Módulo de carga y localización del conjunto de datos de vocales manuscritas.
Filtra exclusivamente las 5 vocales: A, E, I, O (carpeta '0'), U.
"""

import os
import re
import zipfile
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np

# Configuración base del conjunto de datos
ANCHO = 75
ALTO = 100
VOCALES = ["A", "E", "I", "O", "U"]
EXT_VALIDAS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}

def _tiene_clases(carpeta: Path) -> bool:
    """Verifica si la carpeta contiene al menos dos subdirectorios con imágenes válidas."""
    if not carpeta.is_dir():
        return False
    subs = [d for d in carpeta.iterdir() if d.is_dir() and not d.name.startswith(("__", "."))]
    con_imgs = [d for d in subs if any(f.suffix.lower() in EXT_VALIDAS for f in d.iterdir() if f.is_file())]
    return len(con_imgs) >= 2

def encontrar_raiz_dataset(
    rutas_candidatas: List[str] = None,
    ruta_zip: str = "archive (1).zip"
) -> Path:
    """Busca y resuelve la ruta raíz donde se encuentran las carpetas de las clases.
    
    Si el dataset está comprimido en zip, lo extrae automáticamente en 'dataset_extraido'.
    """
    if rutas_candidatas is None:
        rutas_candidatas = [
            "CNN letter Dataset",
            "../CNN letter Dataset",
            "../../CNN letter Dataset",
            "dataset_extraido/CNN letter Dataset",
            "dataset_extraido"
        ]

    for cand in rutas_candidatas:
        p = Path(cand).resolve()
        if p.is_dir() and _tiene_clases(p):
            return p

    # Buscar archivos zip para descomprimir
    candidatos_zip = [
        Path(ruta_zip),
        Path("..") / ruta_zip,
        Path("../..") / ruta_zip,
        Path("/mnt/user-data/uploads") / ruta_zip
    ]
    for z in candidatos_zip:
        if z.is_file():
            destino = Path("dataset_extraido")
            if not destino.exists():
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(destino)
            break

    for base in [Path("dataset_extraido"), Path(".")]:
        if base.is_dir():
            for d in [base] + [x for x in base.rglob("*") if x.is_dir()]:
                if "__MACOSX" in d.parts:
                    continue
                if _tiene_clases(d):
                    return d.resolve()

    raise FileNotFoundError(
        "No se encontró el directorio del dataset ni el archivo ZIP. "
        "Asegúrese de colocar 'CNN letter Dataset/' en la raíz del proyecto."
    )

def leer_imagenes_vocales(raiz: Path = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Path]]:
    """Lee y homogeniza todas las imágenes de vocales.
    
    Retorna:
        X_img: Arreglo (N, 100, 75) uint8 en escala de grises.
        y: Arreglo (N,) con las etiquetas ['A', 'E', 'I', 'O', 'U'].
        grupos: Arreglo (N,) con los identificadores de trazo base 'CLASE_ID'.
        rutas: Lista con las rutas Path de origen de cada imagen.
    """
    if raiz is None:
        raiz = encontrar_raiz_dataset()

    registros = []
    for p in raiz.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXT_VALIDAS and "__MACOSX" not in p.parts:
            clase_dir = p.parent.name
            # Mapeo de la carpeta '0' a la vocal 'O'
            clase = "O" if clase_dir == "0" else clase_dir
            if clase in VOCALES:
                registros.append((p, clase))

    # Ordenar para garantizar determinismo absoluto
    registros.sort(key=lambda r: (r[1], str(r[0])))

    imgs_list = []
    y_list = []
    grupos_list = []
    rutas_list = []

    for ruta, clase in registros:
        with Image.open(ruta) as im:
            # Homogenización a 75x100 y modo escala de grises 'L'
            if im.size != (ANCHO, ALTO):
                im = im.resize((ANCHO, ALTO), Image.BILINEAR)
            if im.mode != "L":
                im = im.convert("L")
            imgs_list.append(np.array(im, dtype=np.uint8))
            y_list.append(clase)
            rutas_list.append(ruta)

            # Extraer id base (ej. 'aug1001_...' -> '1001')
            match = re.match(r"aug(\d+)_", ruta.name)
            base_id = match.group(1) if match else ruta.stem
            grupos_list.append(f"{clase}_{base_id}")

    X_img = np.stack(imgs_list)
    y = np.array(y_list)
    grupos = np.array(grupos_list)

    return X_img, y, grupos, rutas_list

"""Paquete base de utilidades de preprocesamiento, datos y evaluación para el clasificador de vocales."""

from .carga_datos import leer_imagenes_vocales, ANCHO, ALTO, VOCALES
from .preprocesamiento import cargar_datos, preparar_y_guardar_datos, ver_vocal
from .evaluacion import evaluar_clasificador

__all__ = [
    "leer_imagenes_vocales",
    "ANCHO",
    "ALTO",
    "VOCALES",
    "cargar_datos",
    "preparar_y_guardar_datos",
    "ver_vocal",
    "evaluar_clasificador"
]

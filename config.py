"""
Configuración central del clasificador binario Fruta / Verdura.

Igual que en Temperatura/config.py: un solo lugar con rutas, hiperparámetros
y la tabla de mapeo clase-original -> fruta/verdura, para que entrenar.py,
dataset.py y predecir.py no se desincronicen.
"""

import os
from pathlib import Path
import torch

# =========================================================
# RUTAS
# =========================================================

RAIZ = Path(__file__).resolve().parent

RUTA_DATOS = RAIZ / "data"
RUTA_FOTOS_PROPIAS = RUTA_DATOS / "fotos_propias"  # fotos tomadas por el equipo (ruido real)
# El dataset de Kaggle NO se guarda dentro del proyecto: kagglehub lo descarga
# a su propio caché global (~/.cache/kagglehub/), compartido entre prácticas.

RUTA_SALIDAS = RAIZ / "salidas"
RUTA_MODELO = RUTA_SALIDAS / "modelo_frutas_verduras.pt"
RUTA_HISTORICO_CSV = RUTA_SALIDAS / "historico_entrenamiento.csv"
RUTA_METRICAS_CSV = RUTA_SALIDAS / "resultados_frutas_verduras.csv"
RUTA_MATRIZ_CONFUSION = RUTA_SALIDAS / "matriz_confusion.png"

# Identificador del dataset en Kaggle (kagglehub.dataset_download lo descarga aquí).
# https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition
KAGGLE_DATASET_ID = "kritikseth/fruit-and-vegetable-image-recognition"

# =========================================================
# CLASES: fruta vs. verdura (criterio culinario/de verdulería, no botánico)
# =========================================================
# El dataset de Kaggle trae 36 carpetas con nombres de clase específicos
# (manzana, zanahoria, etc.). Aquí se decide, UNA sola vez, a qué lado del
# problema binario pertenece cada una. jitomate/pepino se clasifican como
# verdura (uso culinario), aunque botánicamente sean frutos.

CLASES_FRUTA = {
    "apple", "banana", "grapes", "kiwi", "lemon", "mango",
    "orange", "pear", "pineapple", "pomegranate", "watermelon",
}

CLASES_VERDURA = {
    "beetroot", "bell pepper", "cabbage", "capsicum", "carrot",
    "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant",
    "garlic", "ginger", "jalepeno", "lettuce", "onion", "paprika",
    "peas", "potato", "raddish", "soy beans", "spinach", "sweetcorn",
    "sweetpotato", "tomato", "turnip",
}

ETIQUETAS = ["verdura", "fruta"]  # índice 0 = verdura, 1 = fruta (orden alfabético)


def normalizar_nombre_clase(nombre: str) -> str:
    """'Bell Pepper', 'bell_pepper', 'bell-pepper' -> 'bell pepper'."""
    return nombre.strip().lower().replace("_", " ").replace("-", " ")


def mapear_clase(nombre_carpeta: str) -> str:
    """
    Traduce el nombre de una carpeta de clase original del dataset de Kaggle
    a 'fruta' o 'verdura'. Lanza un error explícito si aparece una clase no
    catalogada (mejor fallar ruidosamente que etiquetar mal en silencio).
    """
    clave = normalizar_nombre_clase(nombre_carpeta)
    if clave in CLASES_FRUTA:
        return "fruta"
    if clave in CLASES_VERDURA:
        return "verdura"
    raise ValueError(
        f"Clase sin mapear: '{nombre_carpeta}' (normalizada: '{clave}'). "
        "Agrégala a CLASES_FRUTA o CLASES_VERDURA en config.py."
    )


# =========================================================
# HIPERPARÁMETROS
# =========================================================

TAMANO_IMAGEN = 224          # tamaño de entrada esperado por MobileNetV2
BATCH_SIZE = 32
NUM_EPOCAS = 15
TASA_APRENDIZAJE = 1e-3
PACIENCIA_EARLY_STOPPING = 4  # épocas sin mejorar F1 de validación antes de parar
RANDOM_STATE = 42

# Fracción de las fotos propias que se reserva como prueba "de ruido real"
# (el resto se mezcla al entrenamiento). El dataset de Kaggle ya trae su
# propio train/validation/test, así que esto solo aplica a fotos_propias/.
FRACCION_PRUEBA_FOTOS_PROPIAS = 0.2

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carga de datos en paralelo: en CPU local (Windows) la decodificación de
# imágenes y el augmentation (blur, perspectiva, etc.) corrían en un solo
# hilo y eran el cuello de botella real, no el modelo. En Colab (Linux)
# el fork de procesos es más barato, así que ahí se usan más workers.
NUM_WORKERS = min(4, os.cpu_count() or 2)
PIN_MEMORY = DISPOSITIVO.type == "cuda"

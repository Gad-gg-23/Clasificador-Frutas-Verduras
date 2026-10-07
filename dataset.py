"""
Construcción del dataset binario (fruta / verdura) a partir de dos fuentes:

1. El dataset de Kaggle (36 clases originales, carpetas train/validation/test)
   -> cada imagen se re-etiqueta a 'fruta' o 'verdura' con config.mapear_clase().
2. data/fotos_propias/{fruta,verdura}/ -> fotos tomadas por el equipo con el
   "ruido" que pidió el profesor (cortadas, sobre una superficie, con fondo
   distinto, etc.). Esta es la parte que "aplasta el ruido con datos": se
   combina con Kaggle y además se le aplica mucha más data augmentation.

El resultado son 3 listas de (ruta_imagen, etiqueta): train / val / test,
listas para envolver en ImagenesFrutaVerdura (Dataset de PyTorch).
"""

import random
import time
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

import config

EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MAX_REINTENTOS_DESCARGA = 6
ESPERA_BASE_SEGUNDOS = 5  # backoff: 5, 10, 20, 40, 80, 160 s


# =========================================================
# DESCARGA DEL DATASET DE KAGGLE
# =========================================================

def descargar_dataset_kaggle() -> Path:
    """
    Descarga (o reutiliza el cache local de) el dataset de Kaggle con
    kagglehub. Requiere credenciales de Kaggle configuradas una sola vez
    (ver README_Frutas_Verduras.md): archivo kaggle.json o variables de
    entorno KAGGLE_USERNAME / KAGGLE_KEY.

    El dataset pesa ~2 GB, así que con una conexión inestable es normal que
    se corte a medio camino (ConnectionResetError / ChunkedEncodingError).
    kagglehub sí soporta reanudar la descarga (usa HTTP Range) siempre que
    el archivo parcial en el cache NO se borre entre intentos, así que aquí
    simplemente se reintenta varias veces con backoff, sin tocar el cache.
    """
    import kagglehub
    from requests.exceptions import ChunkedEncodingError, ConnectionError as RequestsConnectionError

    for intento in range(1, MAX_REINTENTOS_DESCARGA + 1):
        try:
            ruta = kagglehub.dataset_download(config.KAGGLE_DATASET_ID)
            return Path(ruta)
        except (ChunkedEncodingError, RequestsConnectionError, ConnectionError) as error:
            if intento == MAX_REINTENTOS_DESCARGA:
                raise
            espera = ESPERA_BASE_SEGUNDOS * (2 ** (intento - 1))
            print(
                f"\nConexión interrumpida durante la descarga (intento {intento}/{MAX_REINTENTOS_DESCARGA}): {error}"
                f"\nReintentando en {espera}s (se reanuda desde donde se cortó, no desde cero)...\n"
            )
            time.sleep(espera)

    raise RuntimeError("No se pudo descargar el dataset de Kaggle tras varios reintentos.")


def _buscar_carpeta_split(raiz_kaggle: Path, nombre_split: str) -> Path | None:
    """
    La estructura exacta que entrega kagglehub puede venir anidada en una
    carpeta extra (p. ej. .../fruit-and-vegetable-image-recognition/train).
    Se busca 'train', 'validation' o 'test' en cualquier nivel razonable.
    """
    candidatos = list(raiz_kaggle.rglob(nombre_split))
    candidatos = [c for c in candidatos if c.is_dir()]
    return candidatos[0] if candidatos else None


# =========================================================
# ÍNDICE DE (ruta_imagen, etiqueta) A PARTIR DE CARPETAS
# =========================================================

def _listar_imagenes(carpeta: Path) -> list[Path]:
    return [p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES_VALIDAS]


def indexar_split_kaggle(carpeta_split: Path) -> list[tuple[Path, str]]:
    """Recorre train/ (o validation/ o test/): una subcarpeta por clase original."""
    indice = []
    clases_vistas = set()
    for carpeta_clase in sorted(carpeta_split.iterdir()):
        if not carpeta_clase.is_dir():
            continue
        etiqueta = config.mapear_clase(carpeta_clase.name)
        clases_vistas.add(carpeta_clase.name)
        for ruta_img in _listar_imagenes(carpeta_clase):
            indice.append((ruta_img, etiqueta))
    return indice


def indexar_fotos_propias() -> list[tuple[Path, str]]:
    """data/fotos_propias/fruta/*.jpg y data/fotos_propias/verdura/*.jpg."""
    indice = []
    for etiqueta in config.ETIQUETAS:
        carpeta = config.RUTA_FOTOS_PROPIAS / etiqueta
        if not carpeta.exists():
            continue
        for ruta_img in _listar_imagenes(carpeta):
            indice.append((ruta_img, etiqueta))
    return indice


# =========================================================
# CONSTRUCCIÓN DE LOS 3 SPLITS (train / val / test)
# =========================================================

def construir_splits() -> dict[str, list[tuple[Path, str]]]:
    """
    - Usa los splits train/validation/test que ya trae Kaggle (evita fugas:
      nunca se mezclan imágenes de train con las de test).
    - Las fotos propias (ruido real del equipo) se reparten aparte con
      train_test_split estratificado: una parte a train, una parte a test,
      para medir específicamente qué tan bien generaliza al ruido real.
    """
    raiz_kaggle = descargar_dataset_kaggle()

    splits: dict[str, list[tuple[Path, str]]] = {"train": [], "val": [], "test": []}

    mapa_nombres = {"train": "train", "val": "validation", "test": "test"}
    for clave, nombre_kaggle in mapa_nombres.items():
        carpeta = _buscar_carpeta_split(raiz_kaggle, nombre_kaggle)
        if carpeta is None:
            print(f"Aviso: no se encontró la carpeta '{nombre_kaggle}' en el dataset de Kaggle.")
            continue
        splits[clave].extend(indexar_split_kaggle(carpeta))

    fotos_propias = indexar_fotos_propias()
    if fotos_propias:
        etiquetas = [etq for _, etq in fotos_propias]
        train_fp, test_fp = train_test_split(
            fotos_propias,
            test_size=config.FRACCION_PRUEBA_FOTOS_PROPIAS,
            random_state=config.RANDOM_STATE,
            stratify=etiquetas,
        )
        splits["train"].extend(train_fp)
        splits["test"].extend(test_fp)
        print(f"Fotos propias encontradas: {len(fotos_propias)} "
              f"({len(train_fp)} a train, {len(test_fp)} a test).")
    else:
        print("Aviso: no hay fotos propias en data/fotos_propias/{fruta,verdura}/ todavía. "
              "Se entrena solo con el dataset de Kaggle (menos robusto al ruido real).")

    for clave, lista in splits.items():
        random.Random(config.RANDOM_STATE).shuffle(lista)
    return splits


# =========================================================
# TRANSFORMACIONES (aquí vive el "aplastar el ruido con datos")
# =========================================================
# ImageNet mean/std porque el modelo base (MobileNetV2) se preentrenó con eso.
_MEDIA_IMAGENET = [0.485, 0.456, 0.406]
_STD_IMAGENET = [0.229, 0.224, 0.225]


def transform_entrenamiento() -> transforms.Compose:
    """
    Data augmentation agresiva: simula los mismos tipos de 'ruido' que
    describió el profesor (recortes, fondos distintos, ángulos, luz, oclusión
    parcial) para que el modelo no aprenda el fondo sino la fruta/verdura.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(config.TAMANO_IMAGEN, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEDIA_IMAGENET, _STD_IMAGENET),
        # RandomErasing simula pedazos tapados/cortados; va después de ToTensor.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def transform_evaluacion() -> transforms.Compose:
    """Sin augmentation: solo redimensionar y normalizar (val/test/predicción)."""
    return transforms.Compose([
        transforms.Resize((config.TAMANO_IMAGEN, config.TAMANO_IMAGEN)),
        transforms.ToTensor(),
        transforms.Normalize(_MEDIA_IMAGENET, _STD_IMAGENET),
    ])


# =========================================================
# DATASET DE PYTORCH
# =========================================================

class ImagenesFrutaVerdura(Dataset):
    """Envuelve una lista de (ruta_imagen, etiqueta_texto) para PyTorch."""

    def __init__(self, items: list[tuple[Path, str]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ruta, etiqueta_texto = self.items[idx]
        imagen = Image.open(ruta).convert("RGB")
        imagen = self.transform(imagen)
        etiqueta = config.ETIQUETAS.index(etiqueta_texto)  # 0 = verdura, 1 = fruta
        return imagen, etiqueta

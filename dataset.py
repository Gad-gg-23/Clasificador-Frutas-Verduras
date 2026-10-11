"""
Construcción del dataset binario (fruta / verdura) a partir de dos fuentes:

1. El dataset de Kaggle (36 clases originales, carpetas train/validation/test)
   -> cada imagen se re-etiqueta a 'fruta' o 'verdura' con config.mapear_clase().
2. data/fotos_propias/{fruta,verdura}/ -> fotos tomadas por el equipo con el
   "ruido" que pidió el profesor (cortadas, sobre una superficie, con fondo
   distinto, etc.). Esta es la parte que "aplasta el ruido con datos": se
   combina con Kaggle y además se le aplica mucha más data augmentation.

El resultado son 4 listas de (ruta_imagen, etiqueta): train / val /
test_kaggle / test_propias,
listas para envolver en ImagenesFrutaVerdura (Dataset de PyTorch).
"""

import hashlib
import random
import time
from pathlib import Path

from PIL import Image, ImageOps
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

def _hash_archivo(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def _quitar_duplicados_internos(items: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """Si la misma imagen (bytes idénticos) se subió dos veces, se queda solo una."""
    vistos, unicos = set(), []
    for ruta, etiqueta in items:
        h = _hash_archivo(ruta)
        if h not in vistos:
            vistos.add(h)
            unicos.append((ruta, etiqueta))
    return unicos


def _quitar_fugas(items: list[tuple[Path, str]], hashes_prohibidos: set[str]) -> list[tuple[Path, str]]:
    """Quita de `items` toda imagen idéntica a alguna de otro split (fuga de datos)."""
    return [(ruta, etq) for ruta, etq in items if _hash_archivo(ruta) not in hashes_prohibidos]


def _hashes(items: list[tuple[Path, str]]) -> set[str]:
    return {_hash_archivo(ruta) for ruta, _ in items}


def _separar_estratificado(items, fraccion):
    """train_test_split estratificado por fruta/verdura; devuelve (resto, separado)."""
    if not items or fraccion <= 0:
        return items, []
    return train_test_split(
        items,
        test_size=fraccion,
        random_state=config.RANDOM_STATE,
        stratify=[etq for _, etq in items],
    )


def construir_splits() -> dict[str, list[tuple[Path, str]]]:
    """
    Devuelve 4 listas: train, val, test_kaggle y test_propias.

    - Kaggle: se usan sus carpetas train/validation/test, pero se verifica
      por hash que no haya la MISMA imagen en dos splits (en datasets
      armados con imágenes de internet es común que validation/ y test/
      compartan imágenes, o que alguna de train se repita en test). Lo
      repetido se quita de train/val, nunca de test.
      Si al hacerlo validation se queda casi vacío, la validación de Kaggle
      se toma de un 10% de train.
    - Fotos propias: se quitan duplicados exactos y se reparten
      estratificadas en train / val / test. Que haya fotos propias en val
      hace que el early stopping elija el modelo que mejor maneja el ruido.
    - Los dos tests se reportan por separado: Kaggle (fotos "limpias") y
      fotos propias (ruido real, la métrica que más importa).
    """
    raiz_kaggle = descargar_dataset_kaggle()

    kaggle: dict[str, list[tuple[Path, str]]] = {"train": [], "val": [], "test": []}
    mapa_nombres = {"train": "train", "val": "validation", "test": "test"}
    for clave, nombre_kaggle in mapa_nombres.items():
        carpeta = _buscar_carpeta_split(raiz_kaggle, nombre_kaggle)
        if carpeta is None:
            print(f"Aviso: no se encontró la carpeta '{nombre_kaggle}' en el dataset de Kaggle.")
            continue
        kaggle[clave].extend(indexar_split_kaggle(carpeta))

    # --- Limpieza de fugas en Kaggle (test manda, train/val ceden) ---
    hashes_test = _hashes(kaggle["test"])
    val_original = len(kaggle["val"])
    kaggle["val"] = _quitar_fugas(kaggle["val"], hashes_test)
    train_original = len(kaggle["train"])
    kaggle["train"] = _quitar_fugas(kaggle["train"], hashes_test | _hashes(kaggle["val"]))
    print(f"Kaggle: {val_original - len(kaggle['val'])} imágenes de validation y "
          f"{train_original - len(kaggle['train'])} de train eran idénticas a otro split y se quitaron.")

    if len(kaggle["val"]) < 0.5 * val_original:
        print("Kaggle: validation quedó casi vacío (era copia de test); "
              "se toma un 10% de train como validación.")
        kaggle["train"], val_extra = _separar_estratificado(kaggle["train"], 0.10)
        kaggle["val"].extend(val_extra)

    splits = {
        "train": kaggle["train"],
        "val": kaggle["val"],
        "test_kaggle": kaggle["test"],
        "test_propias": [],
    }

    fotos_propias = _quitar_duplicados_internos(indexar_fotos_propias())
    if fotos_propias:
        resto, test_fp = _separar_estratificado(fotos_propias, config.FRACCION_PRUEBA_FOTOS_PROPIAS)
        # la fracción de val se expresa sobre el total, no sobre el resto
        fraccion_val = config.FRACCION_VAL_FOTOS_PROPIAS / (1 - config.FRACCION_PRUEBA_FOTOS_PROPIAS)
        train_fp, val_fp = _separar_estratificado(resto, fraccion_val)
        splits["train"].extend(train_fp)
        splits["val"].extend(val_fp)
        splits["test_propias"].extend(test_fp)
        print(f"Fotos propias (sin duplicados): {len(fotos_propias)} "
              f"({len(train_fp)} a train, {len(val_fp)} a val, {len(test_fp)} a test).")
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
    """
    Sin augmentation (val/test/predicción). Se redimensiona el lado corto y
    se recorta el centro, como en ImageNet: Resize((224, 224)) directo
    aplastaba las fotos rectangulares y deformaba la fruta/verdura.
    """
    return transforms.Compose([
        transforms.Resize(int(config.TAMANO_IMAGEN * 256 / 224)),
        transforms.CenterCrop(config.TAMANO_IMAGEN),
        transforms.ToTensor(),
        transforms.Normalize(_MEDIA_IMAGENET, _STD_IMAGENET),
    ])


# =========================================================
# DATASET DE PYTORCH
# =========================================================

def abrir_imagen(ruta: Path) -> Image.Image:
    """
    Abre en RGB respetando la orientación EXIF: las fotos de celular se
    guardan "acostadas" con una etiqueta que dice cómo rotarlas, y PIL no
    la aplica sola.
    """
    with Image.open(ruta) as imagen:
        return ImageOps.exif_transpose(imagen).convert("RGB")


class ImagenesFrutaVerdura(Dataset):
    """Envuelve una lista de (ruta_imagen, etiqueta_texto) para PyTorch."""

    def __init__(self, items: list[tuple[Path, str]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ruta, etiqueta_texto = self.items[idx]
        imagen = abrir_imagen(ruta)
        imagen = self.transform(imagen)
        etiqueta = config.ETIQUETAS.index(etiqueta_texto)  # 0 = verdura, 1 = fruta
        return imagen, etiqueta

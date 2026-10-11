"""
Descarga automática de imágenes de internet (DuckDuckGo) para ampliar el dataset.
Requiere:  python -m pip install ddgs requests

(No se usa Bing: sin sesión ignora las palabras extra de la búsqueda y
"zucchini whole" / "zucchini market" devuelven las mismas imágenes.)

Flujo en 3 pasos, con revisión humana en medio (internet trae basura: dibujos,
recetas, fotos mezcladas de fruta con verdura...):

    python herramientas/descargar_imagenes.py descargar   # baja y limpia a descargas_revisar/
    (abrir descargas_revisar/ y BORRAR a mano las imágenes malas)
    python herramientas/descargar_imagenes.py mover       # pasa las buenas a data/fotos_propias/

Opcional: --solo Calabacita Ruibarbo  para bajar solo esas especies.

La limpieza automática hace: descartar imágenes rotas o chicas (<200 px),
quitar casi-duplicados (contra las otras descargadas Y contra las que ya
están en data/fotos_propias/), convertir a JPG RGB y reducir a 800 px máx.
"""

import argparse
import io
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402

CARPETA_REVISION = RAIZ / "descargas_revisar"
LADO_MAXIMO = 800
LADO_MINIMO = 200
IMAGENES_POR_BUSQUEDA = 6     # válidas por búsqueda; 5 búsquedas -> ~20-30 por especie
RESULTADOS_A_PROBAR = 15      # muchos links fallan o son repetidos, se prueban de más
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
CABECERAS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

# Especie (nombre de archivo) -> (clase, nombre en español, nombre en inglés).
# Se busca en los dos idiomas: en inglés el buscador devuelve muchas más fotos buenas.
ESPECIES = {
    # ---- FRUTA (botánica): parecen verdura ----
    "Calabacita": ("fruta", "calabacita", "zucchini"),
    "Calabaza": ("fruta", "calabaza de castilla", "pumpkin squash"),
    "Ejote": ("fruta", "ejotes", "green beans"),
    "Tomate_verde": ("fruta", "tomate verde", "tomatillo"),
    "Chile_poblano": ("fruta", "chile poblano", "poblano pepper"),
    "Chile_serrano": ("fruta", "chile serrano", "serrano pepper"),
    "Chile_habanero": ("fruta", "chile habanero", "habanero pepper"),
    "Chile_guero": ("fruta", "chile güero", "yellow wax pepper"),
    "Okra": ("fruta", "okra", "okra"),
    "Aceituna": ("fruta", "aceitunas", "olives fresh"),
    "Chilacayote": ("fruta", "chilacayote", "fig leaf gourd"),
    "Haba": ("fruta", "habas verdes en vaina", "fava beans pods"),
    "Chicharo": ("fruta", "chícharos en vaina", "pea pods"),
    # ---- FRUTA: comunes que faltan ----
    "Lima": ("fruta", "lima dulce", "sweet lime"),
    "Xoconostle": ("fruta", "xoconostle", "xoconostle"),
    "Capulin": ("fruta", "capulín fruta", "capulin cherry"),
    "Cacao": ("fruta", "mazorca de cacao", "cacao pod"),
    "Cafe": ("fruta", "cereza de café", "coffee cherries"),
    "Platano_macho": ("fruta", "plátano macho", "plantain"),
    # ---- FRUTA: complementar ----
    "Pepino": ("fruta", "pepino", "cucumber"),
    "Jitomate": ("fruta", "jitomate", "tomato"),
    "Berenjena": ("fruta", "berenjena", "eggplant"),
    "Elote": ("fruta", "elote", "corn on the cob"),
    "Pimiento": ("fruta", "pimiento morrón", "bell pepper"),
    # ---- VERDURA: parecen fruta ----
    "Ruibarbo": ("verdura", "ruibarbo", "rhubarb"),
    "Jicama": ("verdura", "jícama", "jicama"),
    "Camote": ("verdura", "camote", "sweet potato"),
    "Rabano_rojo": ("verdura", "rábanos", "radish"),
    "Cebolla_morada": ("verdura", "cebolla morada", "red onion"),
    "Jengibre": ("verdura", "jengibre raíz", "ginger root"),
    "Curcuma": ("verdura", "cúrcuma raíz", "turmeric root"),
    "Yuca": ("verdura", "yuca raíz", "cassava root"),
    "Betabel": ("verdura", "betabel", "beetroot"),
    # ---- VERDURA: comunes que faltan ----
    "Cilantro": ("verdura", "cilantro manojo", "cilantro bunch"),
    "Perejil": ("verdura", "perejil manojo", "parsley bunch"),
    "Quelites": ("verdura", "quelites", "quelites greens"),
    "Romeritos": ("verdura", "romeritos", "romeritos seepweed"),
    "Hierbabuena": ("verdura", "hierbabuena", "spearmint"),
    "Hoja_santa": ("verdura", "hoja santa", "hoja santa leaf"),
    "Chaya": ("verdura", "chaya planta", "chaya leaves"),
    "Tupinambo": ("verdura", "tupinambo", "jerusalem artichoke"),
    "Palmito": ("verdura", "palmito", "hearts of palm"),
    # ---- VERDURA: complementar ----
    "Zanahoria": ("verdura", "zanahoria", "carrot"),
    "Brocoli": ("verdura", "brócoli", "broccoli"),
    "Coliflor": ("verdura", "coliflor", "cauliflower"),
    "Papa": ("verdura", "papa", "potato"),
}


def _busquedas(es: str, en: str) -> list[str]:
    """
    Varias búsquedas por especie para que salgan enteras, cortadas y en
    montón. Casi todas en inglés y con "raw"/"fresh": en español "cortada"
    trae sobre todo recetas ya cocinadas.
    """
    return [
        es,
        f"{en} fresh",
        f"{en} cut in half raw",
        f"{en} sliced raw cutting board",
        f"{en} pile market stall",
    ]


def _buscar_urls(busqueda: str) -> list[str]:
    from ddgs import DDGS

    for intento in range(3):
        try:
            return [r["image"] for r in DDGS().images(busqueda, max_results=RESULTADOS_A_PROBAR)]
        except Exception as error:  # límite de peticiones u otro fallo temporal
            print(f"  (búsqueda '{busqueda}' falló: {error}; reintentando)")
            time.sleep(5 * (intento + 1))
    return []


def _bajar_imagen(url: str) -> Image.Image | None:
    import requests

    try:
        r = requests.get(url, headers=CABECERAS, timeout=10)
        r.raise_for_status()
        with Image.open(io.BytesIO(r.content)) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return None


# =========================================================
# HASH PERCEPTUAL (para detectar casi-duplicados)
# =========================================================

def _dhash(imagen: Image.Image) -> int:
    """Difference hash de 64 bits: dos imágenes casi iguales dan hashes cercanos."""
    gris = np.asarray(imagen.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (gris[:, 1:] > gris[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _es_casi_duplicado(h: int, conocidos: list[int], umbral: int = 6) -> bool:
    return any(bin(h ^ otro).count("1") <= umbral for otro in conocidos)


def _hashes_existentes() -> list[int]:
    hashes = []
    for ruta in config.RUTA_FOTOS_PROPIAS.rglob("*"):
        if ruta.suffix.lower() in EXTENSIONES:
            try:
                with Image.open(ruta) as im:
                    hashes.append(_dhash(im))
            except Exception:
                pass
    return hashes


# =========================================================
# PASO 1: DESCARGAR + LIMPIAR
# =========================================================

def descargar(especies: list[str]):
    print("Calculando hashes de las fotos que ya están en el dataset...")
    conocidos = _hashes_existentes()

    for especie in especies:
        clase, es, en = ESPECIES[especie]
        destino = CARPETA_REVISION / clase / especie
        destino.mkdir(parents=True, exist_ok=True)
        n = len(list(destino.glob("*.jpg")))
        aceptadas = descartadas = 0

        print(f"\n=== {especie} ({clase}) ===")
        for busqueda in _busquedas(es, en):
            urls = _buscar_urls(busqueda)
            with ThreadPoolExecutor(max_workers=8) as pool:
                imagenes = list(pool.map(_bajar_imagen, urls))

            guardadas_busqueda = 0
            for im in imagenes:
                if guardadas_busqueda >= IMAGENES_POR_BUSQUEDA:
                    break
                if im is None or min(im.size) < LADO_MINIMO:
                    descartadas += 1
                    continue
                h = _dhash(im)
                if _es_casi_duplicado(h, conocidos):
                    descartadas += 1
                    continue
                conocidos.append(h)
                im.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
                n += 1
                im.save(destino / f"{especie}_{n}.jpg", "JPEG", quality=90)  # sin EXIF
                guardadas_busqueda += 1
            aceptadas += guardadas_busqueda
            time.sleep(1)  # no saturar el buscador

        print(f"  {aceptadas} guardadas en {destino.relative_to(RAIZ)} ({descartadas} descartadas)")

    print(f"\nListo. Ahora abre '{CARPETA_REVISION}' y BORRA las imágenes malas:")
    print("  - dibujos, íconos, recetas/platillos, jugos, cosas procesadas")
    print("  - fotos que mezclen fruta con verdura")
    print("  - fotos que no sean de esa especie")
    print("Después corre:  python herramientas/descargar_imagenes.py mover")


# =========================================================
# PASO 2: MOVER LAS REVISADAS AL DATASET
# =========================================================

def mover():
    if not CARPETA_REVISION.exists():
        print("No hay nada en descargas_revisar/. Corre primero 'descargar'.")
        return

    total = 0
    for carpeta_especie in sorted(CARPETA_REVISION.glob("*/*")):
        if not carpeta_especie.is_dir() or carpeta_especie.parent.name not in config.ETIQUETAS:
            continue
        clase, especie = carpeta_especie.parent.name, carpeta_especie.name
        destino = config.RUTA_FOTOS_PROPIAS / clase
        # continuar la numeración si ya hay fotos de esa especie
        n = len(list(destino.glob(f"{especie}_*.jpg")))
        movidas = 0
        for ruta in sorted(carpeta_especie.glob("*.jpg")):
            n += 1
            while (destino / f"{especie}_{n}.jpg").exists():
                n += 1
            shutil.move(str(ruta), destino / f"{especie}_{n}.jpg")
            movidas += 1
        print(f"  {clase}/{especie}: {movidas} imágenes")
        total += movidas

    shutil.rmtree(CARPETA_REVISION, ignore_errors=True)
    print(f"\n{total} imágenes movidas a {config.RUTA_FOTOS_PROPIAS.relative_to(RAIZ)}/")


def main():
    parser = argparse.ArgumentParser(description="Descarga imágenes de internet para el dataset.")
    parser.add_argument("paso", choices=["descargar", "mover"])
    parser.add_argument("--solo", nargs="+", metavar="ESPECIE",
                        help=f"Solo estas especies. Disponibles: {', '.join(ESPECIES)}")
    args = parser.parse_args()

    if args.paso == "descargar":
        especies = args.solo or list(ESPECIES)
        desconocidas = [e for e in especies if e not in ESPECIES]
        if desconocidas:
            parser.error(f"Especies no reconocidas: {desconocidas}")
        descargar(especies)
    else:
        mover()


if __name__ == "__main__":
    main()

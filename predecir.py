"""
Predicción sobre una imagen (o una carpeta de imágenes) con el modelo ya
entrenado. Uso:

    python predecir.py --imagen ruta/foto.jpg
    python predecir.py --carpeta ruta/con/varias/fotos/
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import config
import dataset
import modelo as modelo_mod

EXTENSIONES_VALIDAS = dataset.EXTENSIONES_VALIDAS


def cargar_modelo() -> torch.nn.Module:
    if not config.RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró '{config.RUTA_MODELO}'. Corre primero 'python Frutas_Verduras.py entrenar'."
        )
    modelo = modelo_mod.crear_modelo()
    modelo.load_state_dict(torch.load(config.RUTA_MODELO, map_location=config.DISPOSITIVO))
    modelo.eval()
    return modelo


def predecir_imagen(modelo: torch.nn.Module, ruta_imagen: Path) -> dict:
    transform = dataset.transform_evaluacion()
    imagen = Image.open(ruta_imagen).convert("RGB")
    tensor = transform(imagen).unsqueeze(0).to(config.DISPOSITIVO)

    with torch.no_grad():
        salida = modelo(tensor)
        probabilidades = F.softmax(salida, dim=1).squeeze(0)

    idx_predicho = int(probabilidades.argmax())
    return {
        "archivo": ruta_imagen.name,
        "prediccion": config.ETIQUETAS[idx_predicho],
        "confianza": float(probabilidades[idx_predicho]),
        "prob_verdura": float(probabilidades[0]),
        "prob_fruta": float(probabilidades[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Predice fruta/verdura para una imagen o carpeta.")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--imagen", type=str, help="Ruta a una sola imagen.")
    grupo.add_argument("--carpeta", type=str, help="Ruta a una carpeta con varias imágenes.")
    args = parser.parse_args()

    modelo = cargar_modelo()

    if args.imagen:
        rutas = [Path(args.imagen)]
    else:
        carpeta = Path(args.carpeta)
        rutas = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES_VALIDAS)
        if not rutas:
            print(f"No se encontraron imágenes en '{carpeta}'.")
            return

    for ruta in rutas:
        resultado = predecir_imagen(modelo, ruta)
        print(
            f"{resultado['archivo']}: {resultado['prediccion'].upper()} "
            f"(confianza={resultado['confianza']:.3f}, "
            f"prob_fruta={resultado['prob_fruta']:.3f}, prob_verdura={resultado['prob_verdura']:.3f})"
        )


if __name__ == "__main__":
    main()

"""
Punto de entrada único del clasificador fruta/verdura.

    python Frutas_Verduras.py datos        -> descarga/verifica el dataset de Kaggle
    python Frutas_Verduras.py entrenar     -> entrena y guarda el modelo
    python Frutas_Verduras.py predecir --imagen ruta.jpg
    python Frutas_Verduras.py predecir --carpeta ruta/
"""

import argparse
import sys


def comando_datos():
    import dataset

    print(f"Descargando/verificando dataset de Kaggle ({__import__('config').KAGGLE_DATASET_ID})...")
    raiz = dataset.descargar_dataset_kaggle()
    print(f"Dataset disponible en: {raiz}")

    splits = dataset.construir_splits()
    for nombre, lista in splits.items():
        print(f"  {nombre}: {len(lista)} imágenes")


def comando_entrenar():
    import entrenar

    entrenar.entrenar()


def comando_predecir(args_restantes):
    import predecir

    sys.argv = [sys.argv[0]] + args_restantes
    predecir.main()


def main():
    parser = argparse.ArgumentParser(description="Clasificador binario Fruta / Verdura.")
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser("datos", help="Descarga y verifica el dataset de Kaggle.")
    subparsers.add_parser("entrenar", help="Entrena el modelo.")
    subparsers.add_parser("predecir", help="Predice sobre una imagen o carpeta.")

    args, restantes = parser.parse_known_args()

    if args.comando == "datos":
        comando_datos()
    elif args.comando == "entrenar":
        comando_entrenar()
    elif args.comando == "predecir":
        comando_predecir(restantes)


if __name__ == "__main__":
    main()

"""
Entrenamiento del clasificador binario fruta / verdura.

Flujo (mismo espíritu que las otras prácticas del curso, adaptado a imágenes):
  1) Construir los splits train/val/test (dataset.construir_splits)
  2) Preparar los DataLoaders con augmentation en train
  3) Crear el modelo (transfer learning, modelo.crear_modelo)
  4) Entrenar con early stopping sobre F1 de validación
  5) Evaluar en test: Accuracy, Precision, Recall, F1, Matriz de Confusión
  6) Guardar modelo, histórico de entrenamiento, métricas y matriz de confusión
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

import config
import dataset
import modelo as modelo_mod


def _crear_dataloaders():
    splits = dataset.construir_splits()

    for nombre in ("train", "val", "test"):
        print(f"  {nombre}: {len(splits[nombre])} imágenes")

    ds_train = dataset.ImagenesFrutaVerdura(splits["train"], dataset.transform_entrenamiento())
    ds_val = dataset.ImagenesFrutaVerdura(splits["val"], dataset.transform_evaluacion())
    ds_test = dataset.ImagenesFrutaVerdura(splits["test"], dataset.transform_evaluacion())

    dl_train = DataLoader(
        ds_train, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, persistent_workers=config.NUM_WORKERS > 0,
    )
    dl_val = DataLoader(
        ds_val, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, persistent_workers=config.NUM_WORKERS > 0,
    )
    dl_test = DataLoader(
        ds_test, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, persistent_workers=config.NUM_WORKERS > 0,
    )
    return dl_train, dl_val, dl_test


def _correr_epoca(modelo, dataloader, criterio, optimizador=None):
    """Una pasada completa. Si optimizador es None, es modo evaluación (sin backprop)."""
    entrenando = optimizador is not None
    modelo.train() if entrenando else modelo.eval()

    perdida_total = 0.0
    y_true, y_pred = [], []

    contexto = torch.enable_grad() if entrenando else torch.no_grad()
    with contexto:
        for imagenes, etiquetas in dataloader:
            imagenes = imagenes.to(config.DISPOSITIVO)
            etiquetas = etiquetas.to(config.DISPOSITIVO)

            if entrenando:
                optimizador.zero_grad()

            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)

            if entrenando:
                perdida.backward()
                optimizador.step()

            perdida_total += perdida.item() * imagenes.size(0)
            y_pred.extend(salidas.argmax(dim=1).cpu().tolist())
            y_true.extend(etiquetas.cpu().tolist())

    n = len(y_true)
    metricas = {
        "perdida": perdida_total / n,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }
    return metricas, y_true, y_pred


def entrenar():
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)

    print("Construyendo splits (Kaggle + fotos propias)...")
    dl_train, dl_val, dl_test = _crear_dataloaders()

    print(f"\nDispositivo: {config.DISPOSITIVO}")
    modelo = modelo_mod.crear_modelo()
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()),
        lr=config.TASA_APRENDIZAJE,
    )

    mejor_f1_val = -1.0
    epocas_sin_mejorar = 0
    historico = []

    print("\nEntrenando...")
    for epoca in range(1, config.NUM_EPOCAS + 1):
        inicio = time.time()

        metricas_train, _, _ = _correr_epoca(modelo, dl_train, criterio, optimizador)
        metricas_val, _, _ = _correr_epoca(modelo, dl_val, criterio)

        duracion = time.time() - inicio
        print(
            f"Época {epoca:02d}/{config.NUM_EPOCAS} "
            f"- train: perdida={metricas_train['perdida']:.4f} acc={metricas_train['accuracy']:.4f} "
            f"- val: perdida={metricas_val['perdida']:.4f} acc={metricas_val['accuracy']:.4f} f1={metricas_val['f1']:.4f} "
            f"({duracion:.1f}s)"
        )

        historico.append({
            "epoca": epoca,
            "perdida_train": metricas_train["perdida"],
            "accuracy_train": metricas_train["accuracy"],
            "perdida_val": metricas_val["perdida"],
            "accuracy_val": metricas_val["accuracy"],
            "f1_val": metricas_val["f1"],
        })

        if metricas_val["f1"] > mejor_f1_val:
            mejor_f1_val = metricas_val["f1"]
            epocas_sin_mejorar = 0
            torch.save(modelo.state_dict(), config.RUTA_MODELO)
            print(f"  -> Nuevo mejor modelo guardado (F1 val = {mejor_f1_val:.4f})")
        else:
            epocas_sin_mejorar += 1
            if epocas_sin_mejorar >= config.PACIENCIA_EARLY_STOPPING:
                print(f"\nEarly stopping: {config.PACIENCIA_EARLY_STOPPING} épocas sin mejorar F1 de validación.")
                break

    pd.DataFrame(historico).to_csv(config.RUTA_HISTORICO_CSV, index=False)
    print(f"\nHistórico de entrenamiento guardado en '{config.RUTA_HISTORICO_CSV}'")

    # --- Evaluación final en test, con el MEJOR modelo (no el último) ---
    print("\nCargando mejor checkpoint para evaluar en test...")
    modelo.load_state_dict(torch.load(config.RUTA_MODELO, map_location=config.DISPOSITIVO))
    _, y_true, y_pred = _correr_epoca(modelo, dl_test, criterio)

    resultados = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }

    print("\n=== Resultados en test ===")
    for nombre, valor in resultados.items():
        print(f"{nombre}: {valor:.4f}")

    pd.DataFrame([resultados]).to_csv(config.RUTA_METRICAS_CSV, index=False)
    print(f"\nMétricas guardadas en '{config.RUTA_METRICAS_CSV}'")

    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusión:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.ETIQUETAS)
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión - Fruta vs. Verdura (test)")
    plt.tight_layout()
    plt.savefig(config.RUTA_MATRIZ_CONFUSION, dpi=150)
    plt.close()
    print(f"Matriz de confusión guardada en '{config.RUTA_MATRIZ_CONFUSION}'")

    print(f"\nModelo final en '{config.RUTA_MODELO}'")


if __name__ == "__main__":
    entrenar()

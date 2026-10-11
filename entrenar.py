"""
Entrenamiento del clasificador binario fruta / verdura.

Flujo (mismo espíritu que las otras prácticas del curso, adaptado a imágenes):
  1) Construir los splits train/val/test (dataset.construir_splits)
  2) Preparar los DataLoaders con augmentation en train
  3) Crear el modelo (transfer learning, modelo.crear_modelo)
  4) Fase 1: entrenar solo la cabeza, con early stopping sobre F1 macro de validación
  5) Fase 2: descongelar los últimos bloques y afinar con lr bajo (mismo early stopping)
  6) Evaluar en test, por separado para Kaggle y para fotos propias:
     Accuracy, Balanced Accuracy, Precision, Recall, F1, Matriz de Confusión
  7) Guardar modelo, histórico, métricas, matrices de confusión y lista de errores
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

import config
import dataset
import modelo as modelo_mod

NOMBRES_TEST = {
    "test_propias": "Fotos propias (ruido real)",
    "test_kaggle": "Kaggle (fotos limpias)",
}


def _crear_dataloaders():
    splits = dataset.construir_splits()

    for nombre, lista in splits.items():
        n_fruta = sum(1 for _, etq in lista if etq == "fruta")
        print(f"  {nombre}: {len(lista)} imágenes ({n_fruta} fruta, {len(lista) - n_fruta} verdura)")

    def _loader(items, transform, barajar):
        return DataLoader(
            dataset.ImagenesFrutaVerdura(items, transform),
            batch_size=config.BATCH_SIZE, shuffle=barajar,
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
            persistent_workers=config.NUM_WORKERS > 0,
        )

    dl_train = _loader(splits["train"], dataset.transform_entrenamiento(), True)
    dl_val = _loader(splits["val"], dataset.transform_evaluacion(), False)
    dls_test = {
        nombre: _loader(splits[nombre], dataset.transform_evaluacion(), False)
        for nombre in NOMBRES_TEST
        if splits[nombre]
    }
    return dl_train, dl_val, dls_test, splits["train"]


def _pesos_clase(items_train) -> torch.Tensor:
    """Peso inverso a la frecuencia: la clase con menos imágenes pesa más en la pérdida."""
    conteos = torch.tensor(
        [sum(1 for _, etq in items_train if etq == clase) for clase in config.ETIQUETAS],
        dtype=torch.float,
    )
    pesos = conteos.sum() / (len(config.ETIQUETAS) * conteos.clamp(min=1))
    return pesos.to(config.DISPOSITIVO)


def _correr_epoca(modelo, dataloader, criterio, optimizador=None):
    """Una pasada completa. Si optimizador es None, es modo evaluación (sin backprop)."""
    entrenando = optimizador is not None
    modelo.train() if entrenando else modelo.eval()

    perdida_total = 0.0
    y_true, y_pred, prob_fruta = [], [], []

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
            prob_fruta.extend(torch.softmax(salidas, dim=1)[:, 1].detach().cpu().tolist())

    n = len(y_true)
    metricas = {
        "perdida": perdida_total / n,
        "accuracy": accuracy_score(y_true, y_pred),
        # F1 macro (promedio de F1 de fruta y de verdura): con clases
        # desbalanceadas no premia ignorar la clase minoritaria.
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return metricas, y_true, y_pred, prob_fruta


def _entrenar_fase(nombre_fase, modelo, optimizador, num_epocas, dl_train, dl_val,
                   criterio, estado, historico):
    """
    Corre una fase con early stopping. `estado["mejor_f1_val"]` se comparte
    entre fases: la fase 2 solo sobrescribe el checkpoint si de verdad
    mejora lo que ya había logrado la fase 1.
    """
    epocas_sin_mejorar = 0
    for epoca in range(1, num_epocas + 1):
        inicio = time.time()

        metricas_train, *_ = _correr_epoca(modelo, dl_train, criterio, optimizador)
        metricas_val, *_ = _correr_epoca(modelo, dl_val, criterio)

        duracion = time.time() - inicio
        print(
            f"[{nombre_fase}] Época {epoca:02d}/{num_epocas} "
            f"- train: perdida={metricas_train['perdida']:.4f} acc={metricas_train['accuracy']:.4f} "
            f"- val: perdida={metricas_val['perdida']:.4f} acc={metricas_val['accuracy']:.4f} "
            f"f1_macro={metricas_val['f1_macro']:.4f} ({duracion:.1f}s)"
        )

        historico.append({
            "fase": nombre_fase,
            "epoca": epoca,
            "perdida_train": metricas_train["perdida"],
            "accuracy_train": metricas_train["accuracy"],
            "perdida_val": metricas_val["perdida"],
            "accuracy_val": metricas_val["accuracy"],
            "f1_macro_val": metricas_val["f1_macro"],
        })

        if metricas_val["f1_macro"] > estado["mejor_f1_val"]:
            estado["mejor_f1_val"] = metricas_val["f1_macro"]
            epocas_sin_mejorar = 0
            torch.save(modelo.state_dict(), config.RUTA_MODELO)
            print(f"  -> Nuevo mejor modelo guardado (F1 macro val = {estado['mejor_f1_val']:.4f})")
        else:
            epocas_sin_mejorar += 1
            if epocas_sin_mejorar >= config.PACIENCIA_EARLY_STOPPING:
                print(f"\n[{nombre_fase}] Early stopping: {config.PACIENCIA_EARLY_STOPPING} "
                      "épocas sin mejorar F1 macro de validación.")
                break


def _metricas_test(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "F1 macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _evaluar_tests(modelo, dls_test, criterio):
    """Evalúa cada test por separado y además el total combinado."""
    filas, matrices, errores = [], {}, []
    y_true_total, y_pred_total = [], []

    for nombre, dl in dls_test.items():
        _, y_true, y_pred, prob_fruta = _correr_epoca(modelo, dl, criterio)
        y_true_total += y_true
        y_pred_total += y_pred

        filas.append({"Conjunto": NOMBRES_TEST[nombre], "Imágenes": len(y_true),
                      **_metricas_test(y_true, y_pred)})
        matrices[NOMBRES_TEST[nombre]] = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # dl no baraja, así que el orden coincide con dl.dataset.items
        for (ruta, _), real, pred, p in zip(dl.dataset.items, y_true, y_pred, prob_fruta):
            if real != pred:
                errores.append({
                    "conjunto": nombre,
                    "archivo": str(ruta),
                    "real": config.ETIQUETAS[real],
                    "predicho": config.ETIQUETAS[pred],
                    "prob_fruta": round(p, 4),
                })

    if len(dls_test) > 1:
        filas.append({"Conjunto": "Total", "Imágenes": len(y_true_total),
                      **_metricas_test(y_true_total, y_pred_total)})

    return pd.DataFrame(filas), matrices, pd.DataFrame(errores)


def _guardar_matrices(matrices):
    fig, ejes = plt.subplots(1, len(matrices), figsize=(5.5 * len(matrices), 4.5), squeeze=False)
    for eje, (titulo, cm) in zip(ejes[0], matrices.items()):
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.ETIQUETAS).plot(
            cmap="Blues", ax=eje, colorbar=False
        )
        eje.set_title(titulo)
    fig.suptitle("Matriz de Confusión - Fruta vs. Verdura (test)")
    fig.tight_layout()
    fig.savefig(config.RUTA_MATRIZ_CONFUSION, dpi=150)
    plt.close(fig)


def entrenar():
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)

    print("Construyendo splits (Kaggle + fotos propias)...")
    dl_train, dl_val, dls_test, items_train = _crear_dataloaders()

    print(f"\nDispositivo: {config.DISPOSITIVO}")
    modelo = modelo_mod.crear_modelo()

    pesos = _pesos_clase(items_train) if config.USAR_PESOS_CLASE else None
    if pesos is not None:
        print("Pesos por clase en la pérdida: "
              + ", ".join(f"{c}={p:.3f}" for c, p in zip(config.ETIQUETAS, pesos.tolist())))
    criterio = nn.CrossEntropyLoss(weight=pesos)

    estado = {"mejor_f1_val": -1.0}
    historico = []

    # --- Fase 1: solo la cabeza ---
    print("\nFase 1: entrenando solo la cabeza (backbone congelado)...")
    optimizador = torch.optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()),
        lr=config.TASA_APRENDIZAJE,
    )
    _entrenar_fase("cabeza", modelo, optimizador, config.NUM_EPOCAS,
                   dl_train, dl_val, criterio, estado, historico)

    # --- Fase 2: afinado de los últimos bloques, partiendo del mejor de la fase 1 ---
    if config.NUM_EPOCAS_AFINADO > 0:
        print(f"\nFase 2: afinando los últimos {config.BLOQUES_A_DESCONGELAR} bloques "
              f"(lr={config.TASA_APRENDIZAJE_AFINADO})...")
        modelo.load_state_dict(torch.load(config.RUTA_MODELO, map_location=config.DISPOSITIVO))
        modelo_mod.descongelar_ultimos_bloques(modelo, config.BLOQUES_A_DESCONGELAR)
        optimizador = torch.optim.Adam(
            filter(lambda p: p.requires_grad, modelo.parameters()),
            lr=config.TASA_APRENDIZAJE_AFINADO,
        )
        _entrenar_fase("afinado", modelo, optimizador, config.NUM_EPOCAS_AFINADO,
                       dl_train, dl_val, criterio, estado, historico)

    pd.DataFrame(historico).to_csv(config.RUTA_HISTORICO_CSV, index=False)
    print(f"\nHistórico de entrenamiento guardado en '{config.RUTA_HISTORICO_CSV}'")

    # --- Evaluación final en test, con el MEJOR modelo (no el último) ---
    print("\nCargando mejor checkpoint para evaluar en test...")
    modelo.load_state_dict(torch.load(config.RUTA_MODELO, map_location=config.DISPOSITIVO))
    resultados, matrices, errores = _evaluar_tests(modelo, dls_test, criterio)

    print("\n=== Resultados en test ===")
    print(resultados.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    resultados.to_csv(config.RUTA_METRICAS_CSV, index=False)
    print(f"\nMétricas guardadas en '{config.RUTA_METRICAS_CSV}'")

    for titulo, cm in matrices.items():
        print(f"\nMatriz de Confusión - {titulo} (filas = real, columnas = predicho; {config.ETIQUETAS}):")
        print(cm)
    _guardar_matrices(matrices)
    print(f"Matriz de confusión guardada en '{config.RUTA_MATRIZ_CONFUSION}'")

    errores.to_csv(config.RUTA_ERRORES_CSV, index=False)
    print(f"{len(errores)} imágenes mal clasificadas listadas en '{config.RUTA_ERRORES_CSV}' "
          "(útil para ver qué tipo de fotos le cuestan y cuáles agregar)")

    print(f"\nModelo final en '{config.RUTA_MODELO}'")


if __name__ == "__main__":
    entrenar()

"""
Modelo: transfer learning con MobileNetV2 preentrenada en ImageNet.

Se congela el 'backbone' (las capas convolucionales que ya saben extraer
bordes/texturas/formas) y solo se entrena una cabeza nueva de clasificación
binaria (fruta vs. verdura). Esto es justo lo que recomendó el profesor:
con pocas fotos propias + mucha augmentation, un modelo preentrenado
generaliza mucho mejor al ruido que una CNN entrenada desde cero.
"""

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import config


def crear_modelo(afinar_completo: bool = False) -> nn.Module:
    """
    afinar_completo=False (por defecto): solo se entrena la cabeza nueva,
        el resto de la red queda congelada (rápido, pocos datos necesarios).
    afinar_completo=True: además se descongelan las últimas capas
        convolucionales para un ajuste más fino (más lento, usar solo si
        ya hay bastantes fotos propias acumuladas).
    """
    modelo = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

    for parametro in modelo.parameters():
        parametro.requires_grad = False

    # Cabeza original: Sequential(Dropout, Linear(1280, 1000)) -> la cambiamos
    # por una cabeza binaria. classifier[1] siempre queda entrenable.
    num_caracteristicas = modelo.classifier[1].in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_caracteristicas, 2),  # 2 clases: verdura (0), fruta (1)
    )

    if afinar_completo:
        # Descongela el último bloque de 'features' para ajuste fino.
        for parametro in modelo.features[-3:].parameters():
            parametro.requires_grad = True

    return modelo.to(config.DISPOSITIVO)

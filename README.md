# Clasificador binario Fruta / Verdura (con ruido)

Clasificador de imágenes que recibe una foto y decide si contiene una **fruta** o una **verdura**. El profesor pidió que fuera robusto a "ruido": fotos cortadas, sobre distintas superficies, con fondos variados, etc. La estrategia que recomendó ("aplastar el ruido con muchos datos") se implementa con **transfer learning + data augmentation agresiva**, combinando un dataset público con fotos propias del equipo.

Es un trabajo en equipo: el código está pensado para que cada integrante corra el mismo pipeline (local o en Colab) y aporte sus propias fotos.

## Idea general

1. **Transfer learning**: en vez de entrenar una CNN desde cero (necesitaría muchísimos datos para ser robusta), se parte de **MobileNetV2** preentrenada en ImageNet y solo se reentrena la última capa para el problema binario. Generaliza mucho mejor con pocos datos propios.
2. **Dataset base**: [Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) (Kaggle), 36 clases que se re-etiquetan a fruta/verdura (ver `CLASES_FRUTA` / `CLASES_VERDURA` en `config.py`).
3. **Fotos propias del equipo** (`data/fotos_propias/fruta/` y `.../verdura/`): aquí va el "ruido real" que pidió el profesor — fruta/verdura cortada, sobre una mesa, con distintos fondos y luces. Se mezclan con el dataset de Kaggle al entrenar.
4. **Data augmentation agresiva** (`dataset.transform_entrenamiento`): recorte aleatorio, rotación, cambios de color, distorsión de perspectiva, blur y "recorte" de pedazos de la imagen (`RandomErasing`, simula oclusión). Esto multiplica la variedad de ruido que ve el modelo sin necesitar fotos infinitas.

## Archivos

| Archivo | Descripción |
|---|---|
| `config.py` | Rutas, hiperparámetros y la tabla de mapeo clase original → fruta/verdura. |
| `dataset.py` | Descarga el dataset de Kaggle, indexa las imágenes (Kaggle + fotos propias), construye los splits train/val/test y define las transformaciones (augmentation). |
| `modelo.py` | Define el modelo de transfer learning (MobileNetV2 + cabeza binaria nueva). |
| `entrenar.py` | Entrena con early stopping sobre F1 de validación, evalúa en test y guarda modelo + métricas + matriz de confusión. |
| `predecir.py` | Predice fruta/verdura para una imagen o una carpeta de imágenes. |
| `Frutas_Verduras.py` | Punto de entrada único (`datos` / `entrenar` / `predecir`). |
| `data/fotos_propias/{fruta,verdura}/` | Aquí cada integrante agrega sus fotos. |
| `salidas/` | Modelo entrenado, histórico, métricas (CSV) y matriz de confusión (PNG) — se genera al entrenar. |
| `colab/Frutas_Verduras_Colab.ipynb` | Notebook para entrenar en Google Colab con GPU gratuita. |

## Cómo agregar fotos propias (todo el equipo)

1. Tomar fotos de frutas o verduras con el "ruido" que pidió el profesor: cortadas, sobre distintas superficies, ángulos y fondos variados.
2. Guardarlas en `data/fotos_propias/fruta/` o `data/fotos_propias/verdura/` según corresponda (jpg/png/webp, cualquier nombre de archivo).
3. No hace falta avisar nada más: `entrenar.py` las detecta automáticamente y las mezcla con el dataset de Kaggle (80% a entrenamiento, 20% a una prueba específica de "ruido real", ver `FRACCION_PRUEBA_FOTOS_PROPIAS` en `config.py`).
4. `git add data/fotos_propias && git commit -m "Fotos de <nombre>" && git push`, y que el resto haga `git pull` antes de entrenar.
5. Mientras más personas agreguen fotos variadas, más robusto se vuelve el modelo — ese es el punto de "aplastar el ruido con datos".

## Credenciales de Kaggle (una sola vez por persona)

El dataset se descarga con `kagglehub`, que necesita credenciales de Kaggle. Kaggle usa ahora un **token único** (ya no el `kaggle.json` de usuario+key del método viejo):

1. Crear cuenta en [kaggle.com](https://www.kaggle.com) si no se tiene.
2. Ir a **kaggle.com/settings → API → Create New Token**. Esto da un token que empieza con `KGAT_...`.
3. **Aceptar las reglas del dataset una vez**: abrir la página del [dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) logueado y aceptar/descargar — si no, la API puede rechazar la descarga aunque el token sea válido.
4. **Local**: guardar el token (sin comillas, sin saltos de línea extra) en un archivo de texto plano en `C:\Users\<usuario>\.kaggle\access_token` (crear la carpeta `.kaggle` si no existe).
5. **Colab**: guardarlo en **Secrets** (icono de llave 🔑 a la izquierda del notebook) con el nombre `KAGGLE_API_TOKEN` — cada integrante usa su propio token, nunca el de otra persona.

Verificar que quedó bien conectado:

```powershell
python -c "from kagglehub import auth; print(auth.whoami())"
```

⚠️ El archivo de token es una credencial: nunca se sube al repo (vive fuera de esta carpeta, el `.gitignore` ni lo toca) y no se comparte entre compañeros.

## Cómo ejecutar (local)

```powershell
python -m pip install -r requirements.txt

python Frutas_Verduras.py datos       # descarga el dataset y muestra cuántas imágenes hay por split
python Frutas_Verduras.py entrenar    # entrena, evalúa y guarda el modelo en salidas/
python Frutas_Verduras.py predecir --imagen ruta\a\una\foto.jpg
python Frutas_Verduras.py predecir --carpeta ruta\a\una\carpeta\
```

Entrenar en CPU funciona (ver resultados abajo) pero es más lento que en GPU. Para entrenar en serio se recomienda GPU — ver la sección de Google Colab.

## Cómo ejecutar (Google Colab, con GPU)

1. Ir a [colab.research.google.com](https://colab.research.google.com) → **Archivo → Abrir cuaderno → GitHub** → pegar `Gad-gg-23/Clasificador-Frutas-Verduras` → abrir `colab/Frutas_Verduras_Colab.ipynb`.
   - Alternativa directa: https://colab.research.google.com/github/Gad-gg-23/Clasificador-Frutas-Verduras/blob/main/colab/Frutas_Verduras_Colab.ipynb
2. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU** (T4, la gratuita).
3. Configurar el token de Kaggle en Secrets (ver sección anterior).
4. Ejecutar las celdas en orden: clona el repo, instala `kagglehub`, descarga el dataset, entrena, y al final se puede descargar el modelo o subirlo de vuelta al repositorio.

## Métricas y validación

- División train/val/test: se usan los splits que ya trae el dataset de Kaggle (sin mezclar train con test) y las fotos propias se reparten con `train_test_split` estratificado (`random_state=42`), igual que el resto de las prácticas del curso.
- Entrenamiento con **early stopping** sobre el F1 de validación (`PACIENCIA_EARLY_STOPPING` épocas sin mejorar).
- Evaluación final en test: Accuracy, Precision, Recall, F1-Score y Matriz de Confusión.

## Resultado actual (solo dataset de Kaggle, sin fotos propias todavía)

Entrenado en CPU local, 13 épocas con early stopping (~1-2 min/época):

| Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|
| 0.9610 | 0.9439 | 0.9266 | 0.9352 |

Esto es con imágenes "limpias" del dataset público. El verdadero reto (fotos cortadas, con fondos y superficies distintas) se prueba agregando fotos propias del equipo y reentrenando — ver sección de arriba.

## Salidas

- `salidas/modelo_frutas_verduras.pt` — pesos del mejor modelo (según F1 de validación).
- `salidas/historico_entrenamiento.csv` — pérdida/accuracy/F1 por época.
- `salidas/resultados_frutas_verduras.csv` — métricas finales en test.
- `salidas/matriz_confusion.png`.

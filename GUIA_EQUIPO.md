# Guía para el equipo — Clasificador Fruta / Verdura

Repositorio: **https://github.com/Gad-gg-23/Clasificador-Frutas-Verduras**
Notebook de Colab: **https://colab.research.google.com/github/Gad-gg-23/Clasificador-Frutas-Verduras/blob/main/colab/Frutas_Verduras_Colab.ipynb**

---

## Parte 0 — Lo que hace el admin del repo (Gad-gg-23), una sola vez

1. Entra a https://github.com/Gad-gg-23/Clasificador-Frutas-Verduras → **Settings** (pestaña del repo) → **Collaborators** (menú izquierdo) → **Add people** → escribe el usuario o correo de GitHub de cada compañero → **Add**.
2. Cada compañero recibe una invitación (correo o notificación en GitHub) — **tiene que aceptarla** antes de poder subir cambios al repo.
3. Les pasas esta guía y el link del notebook de arriba.

Nadie más necesita hacer este paso. El resto de la guía es para **cada compañero**.

---

## Parte 1 — La primera vez que abres el notebook (cada persona, una sola vez)

### 1.1 Crear tu cuenta de Kaggle y tu token

1. Entra a https://www.kaggle.com y crea una cuenta si no tienes.
2. Ve a **https://www.kaggle.com/settings** → sección **API** → **Create New Token**. Esto te da un token que empieza con `KGAT_...` — cópialo, lo vas a necesitar en un momento.
3. Abre la página del dataset una vez, logueado, para aceptar sus reglas: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition (no hace falta descargar nada ahí manualmente, solo con abrirla logueado basta).

### 1.2 Abrir el notebook

1. Abre este link: https://colab.research.google.com/github/Gad-gg-23/Clasificador-Frutas-Verduras/blob/main/colab/Frutas_Verduras_Colab.ipynb
2. Inicia sesión con tu cuenta de Google si te lo pide.

### 1.3 Activar GPU

`Entorno de ejecución` (menú de arriba) → `Cambiar tipo de entorno de ejecución` → selecciona **GPU** (la T4 gratuita) → **Guardar**.

### 1.4 Poner tu token de Kaggle en Colab

1. En la barra lateral **izquierda** de Colab, clic en el ícono de **llave 🔑** ("Secretos").
2. **+ Añadir secreto nuevo**.
3. **Nombre**: escribe exactamente `KAGGLE_API_TOKEN` (mayúsculas, con guiones bajos, sin espacios).
4. **Valor**: pega tu token (el `KGAT_...` del paso 1.1).
5. Activa el switch de **"Acceso desde el cuaderno"** de esa fila (debe quedar azul con una palomita ✓).
6. Cierra el panel con la ❌.

No necesitas tocar nada más ahí — el notebook ya está programado para leer ese secreto solo.

### 1.5 Correr todo el notebook

`Entorno de ejecución → Ejecutar todas` (o celda por celda con ▶️, de arriba hacia abajo).

Lo que vas a ver pasar, en orden:

1. **Clona el repo** — rápido, segundos.
2. **Instala `kagglehub` y verifica la GPU** — debe imprimir `GPU disponible: True`. Si dice `False`, te faltó el paso 1.3.
3. **Conecta tu Kaggle** — debe imprimir `Conectado como: <tu usuario de Kaggle>`.
4. **Fotos desde Drive** — déjala tal cual (`USAR_DRIVE = False`), no la necesitas.
5. **Descarga el dataset y entrena** — tiene DOS celdas:
   - La primera descarga el dataset de Kaggle (**pesa ~2 GB — la primera vez tarda varios minutos**, vas a ver una barra de progreso llenándose). Las siguientes veces es más rápido porque ya queda en caché mientras la sesión siga abierta.
   - La segunda es la que **entrena de verdad**: va a imprimir el progreso época por época (`Época 01/15 - train: ... val: ...`). Tarda varios minutos más. **Espera a que termine por completo** — al final debe decir `Modelo final en '.../salidas/modelo_frutas_verduras.pt'`. Si avanzas a la siguiente celda antes de que termine, te va a salir un error de "archivo no encontrado".
6. **Ver resultados** — te muestra la tabla de métricas (Accuracy, Precision, Recall, F1) y la imagen de la matriz de confusión.
7. **Descargar o subir el modelo** (opcional) — ver Parte 4.
8. **Probar con tu propia foto** (opcional) — ver Parte 5.

---

## Parte 2 — Cómo subir fotos con ruido (en cualquier momento, SIN necesidad de Colab)

Esto se hace desde el navegador, directo en GitHub — no toques Colab para este paso.

1. Entra a https://github.com/Gad-gg-23/Clasificador-Frutas-Verduras
2. Navega a la carpeta `data/fotos_propias/fruta/` (si tu foto es de una fruta) o `data/fotos_propias/verdura/` (si es de una verdura).
3. Botón **Add file → Upload files** (arriba a la derecha).
4. Arrastra tus fotos ahí (jpg o png, el nombre del archivo no importa).
5. Abajo, en "Commit changes": deja el mensaje o escribe algo como "fotos de \<tu nombre\>" → **Commit changes directly to the `main` branch** → **Commit changes**.

Consejos sobre qué fotos subir:
- Entre más variadas, mejor: cortadas, enteras, sobre distintas superficies, con distinta luz, distintos ángulos.
- No subas 10 fotos casi idénticas de lo mismo — mejor repartir entre frutas/verduras distintas y con fondos distintos.
- No necesitas avisarle a nadie ni tocar código: el modelo las detecta solas la próxima vez que alguien entrene.

---

## Parte 3 — Cómo reentrenar después de que se suban fotos nuevas

**La regla simple y segura: vuelve a correr TODO el notebook desde el principio** (`Entorno de ejecución → Ejecutar todas`), cada vez que quieras entrenar con las fotos más recientes.

¿Por qué todo y no solo un par de celdas? Porque Colab reinicia la sesión seguido (por inactividad, o simplemente al volver otro día), y cuando eso pasa se borra todo lo que se había descargado — el repo clonado, el dataset, todo. Si solo corres la celda de entrenar sin haber corrido antes la de clonar el repo *en esa sesión*, te va a salir un error de "archivo no encontrado". Correr todo de nuevo siempre funciona, sin importar si la sesión era nueva o no — es un poco redundante a veces, pero nunca falla.

Pasos:
1. Abre de nuevo el link del notebook (o sigue en la pestaña que ya tenías abierta).
2. `Entorno de ejecución → Ejecutar todas`.
3. El dataset de Kaggle se vuelve a descargar (si la sesión es nueva) — espera esos minutos otra vez.
4. El `git pull` de la celda 1 trae automáticamente las fotos nuevas que se hayan subido.
5. Al entrenar, el modelo ya las va a incluir.
6. Revisa la celda 6 para ver si las métricas cambiaron.

---

## Parte 4 — Compartir el modelo entrenado con el equipo (opcional)

Al final del notebook (sección 7) hay dos formas de quedarte con el modelo que acabas de entrenar:

**Opción A — descargarlo a tu computadora:** corre esa celda y Colab te lo baja directo (`modelo_frutas_verduras.pt`).

**Opción B — subirlo de vuelta al repo** (para que todo el equipo tenga el último modelo entrenado): esa celda te va a pedir:
- Tu **usuario de GitHub**.
- Un **Personal Access Token de GitHub** (NO tu contraseña, GitHub ya no la acepta para esto). Para sacarlo:
  1. Ve a https://github.com/settings/tokens
  2. **Generate new token → Generate new token (classic)**.
  3. Ponle un nombre (ej. "Colab Frutas Verduras") y una expiración (ej. 30 días).
  4. Marca la casilla **`repo`**.
  5. **Generate token** y cópialo — solo se muestra una vez.
  6. Pégalo en el campo de Colab (no se ve mientras escribes, es normal).

⚠️ Ese token es como una contraseña: no lo compartas ni lo pegues en chats. Si se expone, revócalo en esa misma página (botón "Delete") y genera uno nuevo.

---

## Parte 5 — Probar el modelo con una foto tuya

Sección 8 del notebook, al final. Al correrla te aparece un botón **"Elegir archivos"** — selecciona una foto desde tu computadora (lo más interesante es probar con fotos *tuyas*, con el ruido real: cortada, otro fondo). Te va a imprimir algo como:

```
mi_foto.jpg: FRUTA (confianza=0.954, prob_fruta=0.954, prob_verdura=0.046)
```

Puedes subir varias fotos a la vez y repetir la celda las veces que quieras.

---

## Problemas comunes

| Error | Causa | Solución |
|---|---|---|
| `GPU disponible: False` | No activaste GPU | `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU` |
| `FileNotFoundError: salidas/resultados_frutas_verduras.csv` | Corriste la celda 6 antes de que la celda de entrenar terminara | Espera a que la celda de entrenar imprima `Modelo final en '...'`, luego corre la 6 |
| `[Errno 2] No such file or directory: '/content/repo'` o `/content/Frutas_Verduras.py` | Colab reinició la sesión (VM nueva), se perdió todo lo descargado | `Entorno de ejecución → Ejecutar todas` desde el principio |
| El `git push` de la Parte 4 pide usuario/token y falla | Token mal copiado, expirado, o sin el permiso `repo` | Genera un token nuevo (Parte 4) y vuelve a intentar |

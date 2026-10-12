# Imágenes por agregar al dataset

El profe va a probar el modelo con **imágenes de internet**, así que de ahí también salen las nuestras. Ya hay unas 1,640 fotos propias (unas 140 especies) más las 36 clases del dataset de Kaggle. Lo que falta son **las especies que no están** y, sobre todo, **las "trampa" del criterio botánico**: se ven como verdura pero son fruta, o al revés. Esas son las que el profe va a usar para ver si el modelo entendió.

**Recordatorio del criterio (botánico):** *fruta* = viene del ovario de la flor y tiene semillas, aunque en la cocina se use como verdura. *Verdura* = todo lo demás: raíz, tubérculo, bulbo, tallo, hoja, flor o brote.

## Descarga automática

`herramientas/descargar_imagenes.py` baja unas 30 imágenes por especie de DuckDuckGo (enteras, cortadas, en montón). Quita las rotas, las chicas y las casi duplicadas, y las reduce a 800 px. Siempre hay que **revisarlas a mano** antes de moverlas al dataset:

```powershell
python -m pip install ddgs requests
python herramientas/descargar_imagenes.py descargar --solo Calabacita Ruibarbo   # sin --solo baja todas
# revisar descargas_revisar/ y borrar las malas
python herramientas/descargar_imagenes.py mover
```

**Estado:** ya se agregaron las especies de las tablas de abajo (unas 15–30 revisadas por especie), excepto las hierbas de verdura: cilantro, perejil, quelites, romeritos, hierbabuena, hoja santa y chaya.

**2a tanda:** 1,134 imágenes más de 58 especies (las que van al final de `ESPECIES` en el script): trampas como calabaza butternut/espagueti, melón amargo, luffa, edamame, chile manzano y rábano sandía, más las frutas y verduras raras que faltaban de `frutas_verduras.csv`. Hongos y hierbas siguen fuera hasta confirmarlo con el profe. Ojo: ese CSV usa criterio de cocina (jitomate, chiles, calabacita, elote, etc. como verdura); aquí van como **fruta** por el criterio botánico.

---

## Frutas por agregar

### Prioridad alta: parecen verdura pero son fruta

| Especie | Notas |
|---|---|
| Calabacita (italiana y redonda) | Enteras, rebanadas, en cubos |
| Calabaza de Castilla / de Halloween | Entera y abierta con semillas |
| Ejote | En manojo, cortado, en la mata |
| Tomate verde (tomatillo) | Con y sin cáscara |
| Chile poblano | Fresco y asado |
| Chile serrano | |
| Chile habanero | Naranja, rojo y amarillo |
| Chile güero / chilaca | |
| Okra (angú) | Entera y rebanada (se ven las semillas) |
| Aceituna | Verde y negra, en la rama |
| Chilacayote | |
| Pepino | Ya hay 9; faltan rebanado y pepino persa |
| Jitomate | Faltan saladet, cherry, bola y rebanado |
| Berenjena | Además de la morada: blanca, china (larga) y rebanada |
| Elote | Mexicano: con hojas, desgranado, asado |
| Haba verde y chícharo en vaina | La vaina es el fruto; solo con vaina, no granos secos |
| Pimiento morrón | Ya hay 8; faltan colores y abierto |

### Prioridad media: frutas comunes que no están

| Especie | Notas |
|---|---|
| Lima (dulce y la de agua) | Para distinguirla del limón |
| Xoconostle | Parecido a la tuna |
| Capulín | |
| Cacao (mazorca) | Abierta, con las semillas |
| Café (cereza de café) | En la rama |
| Nuez de Castilla en su cáscara verde | |
| Plátano macho | Distinto del plátano Tabasco |

---

## Verduras por agregar

### Prioridad alta: parecen fruta pero son verdura

| Especie | Notas |
|---|---|
| Ruibarbo | Tallo rojo, se cocina como fruta en postres. Es el caso trampa clásico |
| Jícama | Raíz dulce, se come como fruta con chile y limón |
| Camote (naranja y morado) | Ya está en Kaggle; agregar cortado y asado |
| Betabel | Rojo intenso, rebanado parece fruta; faltan cortes |
| Rábano rojo | Redondo y rojo, se confunde con cereza o jitomate cherry |
| Cebolla morada | Partida a la mitad |
| Col morada | Ya hay; agregar rallada o picada |
| Jengibre | Ya está en Kaggle; agregar rebanado y pelado |
| Cúrcuma (raíz) | Naranja por dentro |
| Yuca | Pelada y en trozos |

### Prioridad media: verduras comunes que no están

| Especie | Notas |
|---|---|
| Cilantro | Manojo |
| Perejil | Manojo |
| Quelites | |
| Romeritos | |
| Hierbabuena / menta | |
| Hoja santa | |
| Chaya | |
| Tupinambo | |
| Palmito | Fresco y en rodajas |
| Rábano picante | |
| Brócoli | Ya hay; agregar floretes cortados y cocidos al vapor |
| Coliflor | Ya está en Kaggle; agregar floretes y la morada/naranja |
| Papa | Ya hay; agregar cambray, roja, cortada en papas fritas crudas |
| Zanahoria | Ya hay 9; agregar baby, rallada y en rodajas |

> **Champiñones / hongos:** botánicamente no son fruta ni verdura (son hongos). Hay 11 en `verdura/` como `cham*`. Con nuestro criterio ("verdura = el resto") pueden quedarse ahí, pero vale la pena confirmarlo con el profe.

---

## Cómo elegir las imágenes

**Cantidad:** de 12 a 20 por especie. Entre más especies, mejor; no sirve meter 50 de lo mismo.

**Variedad en cada especie** (aquí está el "ruido" que pidió el profe):

- 3–4 **entera**, con fondos distintos: mesa, mercado, plato, mano, en la planta.
- 3–4 **cortada**: a la mitad, rebanada, en cubos, pelada.
- 2–3 **en montón**: puesto de mercado, canasta, caja.
- 2–3 **de cerca** (macro: textura, semillas) y 1–2 **de lejos** (la pieza ocupa poco de la imagen).
- 1–2 con **mala luz** o fondo saturado.

**Evitar:**

- Fotos que mezclen frutas y verduras (ensaladas, bodegones, cajas mixtas): el modelo no sabría qué etiqueta aprender.
- Productos procesados: jugo, mermelada, salsa, papas fritas, fruta en almíbar.
- Dibujos, íconos o renders 3D. Excepción: si el profe dijo que va a probar con ellos, agregar algunos.
- Marcas de agua muy grandes en el centro (Dreamstime, Shutterstock). Si es pequeña en una esquina, no pasa nada.
- Imágenes de menos de ~200×200 px.
- La misma imagen dos veces. El código ya quita duplicados exactos, pero no los recortes ni los reescalados de la misma foto.

**Nombres de archivo:** `Especie_N.jpg`, sin acentos, con la especie completa (`Calabacita_1.jpg`, `Tomate_verde_3.jpg`). Así se puede contar cuántas hay de cada una. Evitar abreviaturas tipo `cc1.jpg`.

**Carpeta:** `data/fotos_propias/fruta/` o `data/fotos_propias/verdura/` según el criterio botánico. Ante la duda: ¿tiene semillas y salió de la flor? → fruta.

**Después de entrenar:** revisar `salidas/errores_test.csv`. Las especies que más aparecen ahí son las que necesitan más imágenes.

# Servidor de inferencia — modelos intercambiables por petición

Servidor HTTP (FastAPI) que sirve modelos de prediseño de uniones. Los modelos son
**paquetes autocontenidos** dentro de `models/<NAME>/`, se instalan con el `Makefile` del
repositorio y **todos los instalados se cargan a la vez** (conviven en memoria). Cada
petición se **enruta al modelo** que se indique en `model`.

```
production/server/
├── main.py             Endpoints FastAPI (docs interactivas en /docs)
├── model_registry.py   Carga simultánea de modelos + recarga en caliente
├── predictor.py        Validación de contrato, routing e inferencia
├── batcher.py          Batching dinámico de inferencia (una copia del modelo)
├── escenarios.py       Escenarios E1/E2/E3 subiendo cargas
├── storage.py          Dataset acumulado (validaciones; sin reentreno por ahora)
├── config.py           Ajustes del servidor + MODEL_NAME por defecto
├── models/             Paquetes de modelo instalados (generado por `make`)
│   └── V11/
│       ├── core/adapter.py     contrato + preproceso + load + predict
│       ├── weights.pt
│       ├── preprocessor.joblib
│       └── data/               catálogos Tekla
└── requirements.txt    Solo si NO usas `uv` desde la raíz
```

## Arrancar (con `uv`, desde la raíz del proyecto)

```bash
make install-model MODEL=V11          # copia el modelo a production/server/models/V11
uv run uvicorn main:app --app-dir production/server --host 0.0.0.0 --port 8080
```

`uv` toma las dependencias del `pyproject.toml` de la raíz (fastapi, uvicorn, torch,
scikit-learn, joblib, pandas, pyarrow). Si prefieres el flujo antiguo con `pip`:
`pip install -r production/server/requirements.txt`.

> **1 worker** (por defecto). Con varios, cada proceso carga su propia copia de todos los
> modelos, su propio pool de refinado y su propio watcher de reentreno; una recarga
> (`POST /model/activate`) solo afectaría al worker que la atiende. Para escalar, usar Ray
> (ver `doc/ray_escalado.md`), no más workers de uvicorn.
>
> **No usar `--reload`** con MCP en uso: cada cambio reinicia el worker y corta las
> peticiones en vuelo. Reservado para desarrollo sin clientes conectados.

### Concurrencia y timeouts

Los tools MCP no bloquean el event loop: la predicción y la comprobación se ejecutan en
sendas hebras dedicadas, y el refinado en un pool de procesos aislado. Todo se corta con
timeout para que una petición no quede colgada:

El presupuesto maestro es `AI_CLIENT_TIMEOUT_S` (por defecto `100`, el límite del
conector MCP de Copilot Studio). El resto de tiempos se derivan de él salvo que se fijen
explícitamente:

| Variable | Defecto | Uso |
|---|---|---|
| `AI_CLIENT_TIMEOUT_S` | `100` | límite duro del cliente/gateway MCP |
| `AI_CLIENT_TIMEOUT_MS` | `100000` | alternativa en milisegundos si no se define `AI_CLIENT_TIMEOUT_S` |
| `AI_PREDICT_TIMEOUT_S` | 85 % del cliente | `predecir_conexion` |
| `AI_CHECK_TIMEOUT_S` | 85 % del cliente | `comprobar_aisc` |
| `AI_REFINE_TIMEOUT_S` | 65 % del cliente | refinado (pool) |
| `AI_REFINE_TIMEOUT_PER_ROW_S` | `5` | margen extra por ronda de filas en el pool |
| `AI_REFINE_BACKEND` | `process` | `process` (pool), `serial` (debug) o `ray` (futuro) |
| `AI_REFINE_WORKERS` | `min(8, cpus)` | nº de procesos del pool |
| `AI_BATCH_MAX_ROWS` | `64` | máximo de filas por forward del batcher |
| `AI_BATCH_MAX_WAIT_MS` | `10` | espera máxima para agrupar peticiones concurrentes |
| `AI_INFERENCE_TIMEOUT_S` | 50 % del cliente | timeout de una petición esperando al batcher |
| `AI_SCENARIO_WORKERS` | `4` | orquestadores de escenarios en paralelo |
| `AI_SCENARIO_BATCH_ROWS` | `50` | máximo de conexiones por llamada MCP `rows` |
| `AI_SCENARIO_MAX_QUEUE` | `= workers` | peticiones adicionales en cola antes de responder ocupado |
| `AI_SCENARIO_QUEUE_TIMEOUT_S` | 15 % del cliente | espera máxima en cola |
| `AI_SCENARIO_TIMEOUT_S` | 80 % del cliente | presupuesto total de ejecución del escenario |
| `AI_MCP_JSON_RESPONSE` | `1` | respuesta MCP en JSON; `0` para forzar SSE |

Con los valores por defecto: 15 s de cola + 80 s de ejecución = 95 s, por debajo de los
100 s del cliente. Si se cambia `AI_CLIENT_TIMEOUT_S`, todo se reajusta proporcionalmente.

El backend de refinado es una interfaz (`RefineBackend`) con implementaciones
`ProcessPoolBackend` y `SerialBackend`; añadir `RayBackend` no requiere tocar el servidor.
El lock del pool solo cubre el submit: varios escenarios/peticiones pueden tener lotes en
vuelo a la vez y los workers toman filas de cualquier lote.

La inferencia pasa por un **batcher dinámico** (`batcher.py`): una sola copia del modelo y
un worker que agrupa peticiones concurrentes en un único forward. El refinado sigue en el
pool de procesos y el codame/check por petición, fuera del worker de GPU. El ajuste
geométrico usa una vía rápida (`calcular_geometria`) que evita los 17 pasos de resistencia
y la construcción de objetos Pydantic.

`/scenarios` y el tool MCP `disenar_escenarios_carga` usan un **control de admisión**:
- Hasta `AI_SCENARIO_WORKERS` escenarios ejecutándose y `AI_SCENARIO_MAX_QUEUE` en espera.
- Si la cola está llena, respuesta inmediata **503 / SERVIDOR_OCUPADO** (sin esperar 100 s).
- Si un escenario espera más de `AI_SCENARIO_QUEUE_TIMEOUT_S` o ejecuta más de
  `AI_SCENARIO_TIMEOUT_S`, se corta con **504 / TIMEOUT_SERVIDOR**. No hay resultados
  parciales: o se devuelve el cálculo completo o un error claro.

El servidor MCP anuncia su capacidad en `instructions` (máximo de llamadas concurrentes) y
responde en JSON por defecto (`AI_MCP_JSON_RESPONSE=1`) para evitar los cuelgues de streams
SSE a través del túnel. `disenar_escenarios_carga` devuelve JSON por lotes con el detalle
completo en `lote`; se omite `historial` para no engordar innecesariamente la respuesta.

En MCP, los estados ocupado/timeout se devuelven como **resultado estructurado** (no como
error JSON-RPC) para que Copilot no los convierta en un 504: `status` =
`SERVIDOR_OCUPADO`/`TIMEOUT_SERVIDOR`, `http_status` = `503`/`504`, `retry_after_s` y
`mensaje`. En REST siguen siendo HTTP 503/504.

### MCP `disenar_escenarios_carga`: llamadas por lote

El contrato MCP ya no es una conexión por llamada: recibe `rows` (lista, objeto único o
string JSON) con hasta `AI_SCENARIO_BATCH_ROWS` conexiones (50 por defecto). Las filas del
lote se ejecutan **secuencialmente** dentro de una sola plaza del control de admisión.
Si llegan más de `AI_SCENARIO_BATCH_ROWS`, se procesan las primeras y el resto se descarta;
la respuesta lo indica en `n_descartadas` y `descartadas`.

La respuesta incluye `lote` con el detalle completo por conexión (`resultado.rows`,
`resultado.checks` compactos, `escenarios` y `tabla_markdown`), pero **sin `historial`**.
También incluye `status`, contadores, `resumen` compacto por conexión, `errores` y
`tabla_markdown` resumen. Una fila con error o timeout no aborta el resto del lote.
El cliente debe llamar en varios lotes si necesita más conexiones.

## Instalar / actualizar un modelo

El `Makefile` (raíz) copia el paquete del modelo (código `core/` + artefactos + datos):

```bash
make install-model MODEL=V11
```

Rutas de origen: cada modelo puede fijar las suyas en `<NAME>/model.mk` (relativas a su
carpeta). Si no existe o no define alguna, se usa el valor por defecto.

| destino en `models/<NAME>/` | variable en `<NAME>/model.mk` | por defecto |
|---|---|---|
| `core/` | `CORE` | `core` |
| `data/` | `DATA` | `data` |
| `weights.pt` | `WEIGHTS` | `artifacts/model_final.pt` |
| `preprocessor.joblib` | `PREPROC` | `artifacts/preprocessor.joblib` |

Ejemplo de `V11/model.mk` (V11 guarda sus pesos en `artifacts_v4/`):

```make
WEIGHTS = artifacts_v4/model_final.pt
PREPROC = artifacts_v4/preprocessor.joblib
```

Tras instalar/actualizar, recarga sin reiniciar:

```bash
curl -X POST localhost:8080/model/activate -H "Content-Type: application/json" -d '{"name":"V11"}'
```

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/health` | Estado, modelos instalados/cargados y nº de filas del dataset |
| GET | `/models` | Modelos instalados y cargados; modelo por defecto |
| GET | `/model` | Contrato del modelo por defecto (inputs/opcionales/outputs) |
| GET | `/schema?model=...` | Inputs del modelo (tipo/descripción/opciones) + columnas de resultado |
| GET | `/` y `/web/...` | Interfaz web (`web/conexiones_tr.html`) |
| POST | `/model/activate` | `{"name":"V11"}` → carga/recarga un modelo |
| POST | `/predict` | `{"model":"V11","rows":[...]}` → valida, predice y devuelve las 63 columnas CODAME |
| POST | `/scenarios` | `{"row":{...},"incremento":0.05,"max_pasos":40}` → E1/E2/E3; 503 si está ocupado, 504 si agota el tiempo |
| POST | `/check` | `{"rows":[...]}` (filas CODAME editadas) → comprobación AISC 360-10 |
| POST | `/dataset/append` | Filas marcadas → traduce a `data_final` y añade al parquet de reentreno (upsert por `Unicode`) |
| POST | `/validate` | Guarda filas validadas (sin reentreno) |
| GET | `/dataset/summary` | Resumen del dataset acumulado |

Sin autenticación (API abierta en la red interna). Documentación interactiva en `/docs`
(Swagger) y `/redoc`.

## `GET /schema?model=...`

Esquema del modelo: los **inputs** (cada uno con `key`, `description`, `type` y `options`
si es categórico) y las **columnas de resultado** (fijas). El input lo define cada modelo
(lo puede transformar o no) **+ `UNICODE`** siempre.

```json
{"model":"V11",
 "inputs":[{"key":"UNICODE","description":"Identificador unico de la union","type":"text"},
           {"key":"Normativas","description":"Normativa aplicable (columna Standards)","type":"categorical","options":["AISC-360-05/10"]},
           {"key":"BEAM","description":"Perfil de la viga","type":"categorical","options":["HEA100","..."]},
           {"key":"ratio_moment","description":"Relacion momento/demanda (0-1)","type":"number"}],
 "result_columns":["Unicode","Standards","...","Estado_IA"]}
```

La web construye el formulario dinámicamente a partir de este esquema.

## `POST /predict`

Cuerpo: `{"model":"V11","rows":[{...inputs del /schema...}]}`. Los campos vacíos se
omiten; `UNICODE` y los inputs de predicción son obligatorios.

```json
{"model":"V11",
 "rows":[{"UNICODE":"U1","Normativas":"AISC-360-05/10","BEAM":"IPE200","COLUMN":"HEA100",
          "ratio_moment":0.82,"ratio_shear":0.25,"ratio_axial":0.03,
          "grado_acero":"S355JR","clase_tornillo":"10.9"}]}
```

Respuesta `200`: por cada fila, **las 63 columnas de CODAME** (`Tabla_MOMENT_M1`),
idénticas para cualquier modelo. La traducción la hace el **servidor** (`codame.to_codame`,
usando el catálogo de `server/data`) e **imputa** lo que el modelo no predice (0/vacío/
constantes).

```json
{"model_name":"V11",
 "rows":[{"Unicode":"U1","Standards":"AISC-360-05/10","BoltName_ep":"M20",
          "MzEd_Porcen":82.0,"pfi":52.38,"ThicknessName_t_ep":"15mm",
          "ntensionrows":2,"BoltThreads":true}]}
```

### Error de contrato → HTTP 422

Si a alguna fila le falta un input requerido del modelo seleccionado:

```json
{"detail":"Faltan inputs requeridos por el modelo 'V11'","model":"V11",
 "missing":{"fila 1":["UNICODE","ratio_moment"]},
 "expected":["UNICODE","Normativas","BEAM","COLUMN","ratio_moment","ratio_shear",
             "ratio_axial","grado_acero","clase_tornillo"]}
```

También 422 si el perfil no está en el catálogo Tekla o si el material no es uno de los
vistos en entrenamiento (V11: `S355JR`; `10.9`/`A325M`). Otros códigos: `400` filas
vacías, `404` modelo no disponible, `503` sin modelos. **El servidor nunca devuelve 500
por culpa del modelo**: si la traducción a CODAME falla o no cubre alguna columna, esa
columna vuelve con su valor simbólico por defecto.

## Resultados CODAME (63 columnas)

`production/server/codame_schema.py` define `CODAME_COLUMNS` (orden canónico) y
`CODAME_DEFAULTS` (valores simbólicos de la red de seguridad). El mapeo
`input + diseño → CODAME` vive en `production/server/codame.py` y usa el catálogo de
`production/server/data` (`catalogo.py`). Referencia del mapeo:
`doc/codame/mapeo_columnas_codame.md`.

## `POST /check` — comprobación AISC 360-10

Recibe las filas del esquema **CODAME** (la tabla predicha, ya editada por el revisor) y
devuelve, por fila: `cumple`, `ratio_pesimo`, `descripcion_ratio_pesimo` y el detalle
(`ratios`, `complementarios`). No depende del modelo (usa el catálogo del servidor).

```json
{"rows":[{"Unicode":"U1","Standards":"AISC-360-05/10","b_ep":155.0,"p2_ep":80.0,
          "pfo":39.1,"pfi":52.4,"pb":90.0,"p3":105.0,"ThicknessName_t_ep":"15mm",
          "BoltName_ep":"M20","MzEd_Porcen":82.0,"fy":355.0,"fu":490.0}]}
```

Respuesta `200`: `{"status":"ok","rows":[{...,"cumple":true,"ratio_pesimo":0.62,
"descripcion_ratio_pesimo":"...","ratios":[...],"complementarios":[...]}]}`. Si una fila
no se puede comprobar, se devuelve un informe con `cumple:false` y el error (sin 500).

## Estructura servida y dataset de reentreno

`/predict` devuelve **CODAME + columnas extra** (`codame_schema.SERVED_COLUMNS`): las 62 de
CODAME (la columna `Normativas` sustituye a `Standards`; sin `Estado_IA`) más
`Capacity_Moment/Shear/Axial`, `SlipCritical`, `National_Annex`, `AceroEstructural`
(vacías/NULL si el modelo no las aporta).

`POST /dataset/append` recibe las filas **marcadas y validadas** por el revisor, las
**traduce** al esquema de reentreno (`train_dataset.to_train_row`), rellena con **NULL** lo
que falte (`DrawingCode`, demanda ausente, …) y las añade al parquet unificado
`production/server/data/global_dataset.parquet` **reemplazando `Unicode` duplicados**.
Devuelve `{added, updated, total_rows}`.

El parquet de reentreno tiene **41 columnas** (subconjunto de `data_final.csv`): `Capacity_*`,
diseño (`g,tws,Bp,Lp,tp,Lh,e,d,p1..p12,ts,tbs,tst,Hole`), ids/metadatos
(`Unicode,BEAM,COLUMN,DrawingCode,Bolt,SlipCritical,Normativas,AceroEstructural,EstandarTornillos,Source,National_Annex`).
**No** se guardan las columnas derivadas (`BEAM_*`, `COLUMN_*`, `fy/fu/fv`) ni `standard`,
`n_alma`, `n_cantonera` (se derivarán al adaptar el modelo).

En la web (tabla editable de resultados): casilla por fila, botones **Marcar todas** /
**Marcar "CUMPLE"** (solo las `CUMPLE`) / **Enviar conexiones marcadas** (con confirmación);
editar una celda marca la fila como `MODIFICADO` y la desmarca. El estado es solo de UI.

## Flujo de la web

1. Formulario dinámico (`/schema`) → **Pre-diseñar con IA** (`/predict`) → tabla CODAME
   **editable** (el revisor corrige lo que haga falta).
2. **Comprobar todas las entradas** (`/check`) sobre la tabla editada → informe por fila.

## Añadir un modelo nuevo (carpeta V)

1. Crea `<V>/core/adapter.py` con la interfaz:

   ```python
   MODEL_NAME = "V12"
   RAW_INPUT_COLUMNS = [...]        # inputs del modelo (+ UNICODE)
   OPTIONAL_INPUT_COLUMNS = [...]   # opcionales con default
   OUTPUT_FIELDS = [...]            # campos que predice
   class InputContractError(ValueError): ...

   def input_schema(bundle=None) -> list: ...       # [{key,description,type,options?}]
   def to_internal(table_df) -> pd.DataFrame: ...   # tabla -> features internas
   def from_internal(design_df) -> pd.DataFrame: ...# salidas -> claves canonicas del diseno
   def load() -> dict: ...                          # {"model","pre","device"}
   def predict(features_df, bundle) -> pd.DataFrame: ...
   def contract() -> dict: ...   # {"name","raw_inputs","optional_inputs","outputs"}
   ```

2. `make install-model MODEL=V12 [WEIGHTS=... PREPROC=... DATA=...]`
3. `POST /model/activate {"name":"V12"}` (o reiniciar).

No hace falta tocar `main.py` ni `predictor.py`: el servidor sólo conoce la interfaz del
`adapter.py`. La traducción a las 63 columnas CODAME (y la comprobación) las hace el
servidor con su catálogo (`production/server/data`), así el resultado es idéntico entre
modelos.

## Notas técnicas

- El `preprocessor.joblib` de los modelos está serializado como
  `core.dataset.Preprocessor`. El registro importa el `core/` de cada paquete con el
  nombre top-level `core` de forma **secuencial**, capturando las referencias en cada
  bundle; así varios modelos conviven. El código del modelo no debe hacer `import core.*`
  en tiempo de ejecución (solo en el import del módulo).
- Los checkpoints se cargan con `torch.load(..., weights_only=False)` porque los
  metadatos incluyen arrays numpy (`p_mean_mm`/`p_std_mm`). Son artefactos propios.
- Cada modelo puede tener su propia arquitectura y su propio preproceso: todo vive en su
  `core/`.

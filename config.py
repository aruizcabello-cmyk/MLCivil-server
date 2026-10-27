"""
Configuración global del servidor HTTP y del modelo de IA (v4).
Unifica la infraestructura web con la definición de características, catálogos y targets.
"""
import os
from pathlib import Path

# ============================================================================
# 1. CONFIGURACIÓN DEL SERVIDOR HTTP Y RUTAS DE PERSISTENCIA
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("AI_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset global de validación (filas guardadas para reentrenamiento)
DATASET_PATH = DATA_DIR / "global_dataset.parquet"
LOG_PATH = DATA_DIR / "server.log"

HOST = os.environ.get("AI_HOST", "0.0.0.0")
PORT = int(os.environ.get("AI_PORT", "8080"))

# Concurrencia y timeouts del servidor.
#   - AI_CLIENT_TIMEOUT_S define el limite duro del cliente/gateway (100s = 100000ms en
#     Copilot Studio). Si se cambia este valor, todos los demas tiempos se adaptan en
#     proporcion (salvo que se fijen explicitamente por variable de entorno):
#       cola escenarios (15%) + ejecucion escenarios (80%) = 95% < 100% del cliente.
#   - Refinado: backend `process` (pool de procesos, por defecto), `serial` (debug) o,
#     en el futuro, `ray` (ver doc/ray_escalado.md).
CLIENT_TIMEOUT_S = float(
    os.environ.get("AI_CLIENT_TIMEOUT_S", "").strip()
    or str(float(os.environ.get("AI_CLIENT_TIMEOUT_MS", "100000")) / 1000.0)
)

PREDICT_TIMEOUT_S = float(
    os.environ.get("AI_PREDICT_TIMEOUT_S", str(round(CLIENT_TIMEOUT_S * 0.85, 1)))
)
CHECK_TIMEOUT_S = float(
    os.environ.get("AI_CHECK_TIMEOUT_S", str(round(CLIENT_TIMEOUT_S * 0.85, 1)))
)
REFINE_TIMEOUT_S = float(
    os.environ.get("AI_REFINE_TIMEOUT_S", str(round(CLIENT_TIMEOUT_S * 0.65, 1)))
)
REFINE_TIMEOUT_PER_ROW_S = float(os.environ.get("AI_REFINE_TIMEOUT_PER_ROW_S", "5"))
REFINE_BACKEND = os.environ.get("AI_REFINE_BACKEND", "process").strip().lower()

# Batching dinamico de inferencia (una sola copia del modelo, un worker de GPU).
# Se agrupan peticiones concurrentes hasta AI_BATCH_MAX_ROWS o durante
# AI_BATCH_MAX_WAIT_MS, lo que ocurra primero.
BATCH_MAX_ROWS = max(1, int(os.environ.get("AI_BATCH_MAX_ROWS", "64")))
BATCH_MAX_WAIT_MS = max(0, int(os.environ.get("AI_BATCH_MAX_WAIT_MS", "10")))
INFERENCE_TIMEOUT_S = float(
    os.environ.get("AI_INFERENCE_TIMEOUT_S", str(round(CLIENT_TIMEOUT_S * 0.50, 1)))
)

# Escenarios: N orquestadores + cola acotada. Si se llena la cola o se supera
# SCENARIO_QUEUE_TIMEOUT_S esperando turno, el servidor responde "ocupado" de inmediato
# para no agotar el timeout de 100s del cliente MCP.
SCENARIO_WORKERS = max(1, int(os.environ.get("AI_SCENARIO_WORKERS", "6")))
SCENARIO_MAX_QUEUE = max(
    0, int(os.environ.get("AI_SCENARIO_MAX_QUEUE", "4"))
)
SCENARIO_BATCH_ROWS = max(1, int(os.environ.get("AI_SCENARIO_BATCH_ROWS", "25")))
SCENARIO_QUEUE_TIMEOUT_S = float(
    os.environ.get("AI_SCENARIO_QUEUE_TIMEOUT_S", str(round(CLIENT_TIMEOUT_S * 0.15, 1)))
)
SCENARIO_TIMEOUT_S = float(
    os.environ.get("AI_SCENARIO_TIMEOUT_S", str(round(CLIENT_TIMEOUT_S * 0.80, 1)))
)
SCENARIO_BATCH_ROW_TIMEOUT_S = float(
    os.environ.get("AI_SCENARIO_BATCH_ROW_TIMEOUT_S", "30")
)

# Transporte MCP: JSON suele evitar cuelgues de streams SSE en tuneles. Si tu
# conector solo soporta SSE, pon AI_MCP_JSON_RESPONSE=0.
MCP_JSON_RESPONSE = os.environ.get("AI_MCP_JSON_RESPONSE", "1").strip().lower() not in (
    "0", "false", "no"
)

# Orígenes web autorizados por CORS, separados por comas. Evitar "*" si la web
# se sirve fuera de la red interna.
# Para abrir el HTML con doble clic (file://) añadir "null" solo para pruebas.
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("AI_CORS_ORIGINS", "http://localhost:8000").split(",")
    if o.strip()
]

# Definición de rutas absolutas relativas a la carpeta 'server/'
MODELS_DIR = BASE_DIR / "models"

def _active():
    try:
        import json
        n = json.loads((MODELS_DIR / "active.json").read_text())["name"]
        return n if (MODELS_DIR / n / "core" / "adapter.py").exists() else None
    except Exception:
        return None

MODEL_NAME = _active() or "V11"

# ============================================================================
# 2. DEFINICIÓN DE ENTRADAS Y CARACTERÍSTICAS DE PRODUCCIÓN
# ============================================================================
# ---------------------------------------------------------------------------
# COLUMNAS DE ENTRADA DESDE EXCEL (23 columnas totales)
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "BEAM_type", "COLUMN_type",   # HE / W / IPE / HI
    "grado_acero",                # S275JR / S355JR...
    "clase_tornillo",             # 8.8 / 10.9 / A325M...
]

NUMERIC_FEATURES = [
    "BEAM_d_mm", "BEAM_bf_mm", "BEAM_tw_mm", "BEAM_tf_mm",
    "COLUMN_d_mm", "COLUMN_bf_mm", "COLUMN_tw_mm", "COLUMN_tf_mm",
    "fy_acero_mpa", "fu_acero_mpa", "fy_tornillo_mpa", "fu_tornillo_mpa", "fv_tornillo_mpa",
    "ratio_moment", "ratio_shear", "ratio_axial",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columnas brutas recibidas en el JSON desde Excel
RAW_INPUT_COLUMNS = ["UNICODE", "BEAM", "COLUMN"] + FEATURE_COLUMNS


# ============================================================================
# 3. TARGETS Y SALIDAS DEL MODELO
# ============================================================================
TARGET_CLS_STANDARD = "standard"
TARGET_CLS_BOLT = "Bolt"
TARGET_CLS_SLIP_CRITICAL = "SlipCritical"
TARGET_CLS_N_ALMA = "n_alma"
TARGET_CLS_N_CANTONERA = "n_cantonera"

N_Y = 12
P_LABEL_COLUMNS = [f"p{i}_label" for i in range(1, N_Y + 1)]
N_P_LABEL_CLASSES = 11

CLASSIFICATION_TARGETS = [
    TARGET_CLS_STANDARD, TARGET_CLS_BOLT, TARGET_CLS_SLIP_CRITICAL,
    TARGET_CLS_N_ALMA, TARGET_CLS_N_CANTONERA,
] + P_LABEL_COLUMNS

REGRESSION_ALWAYS = ["g", "e", "d", "Bp", "tp"]

Y_COLUMNS = [f"Y{i}" for i in range(1, N_Y + 1)]
P_COLUMNS = [f"p{i}" for i in range(1, N_Y + 1)]
STIFFENER_COLUMNS = ["tws", "ts", "tbs", "tst"]
MASKED_COLUMNS = P_COLUMNS + STIFFENER_COLUMNS

REGRESSION_TARGETS = REGRESSION_ALWAYS + MASKED_COLUMNS
ALL_TARGETS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS

MAX_NUM_FILAS = 1 + N_Y
N_ALMA_CLASSES = 7        # 0..6
N_CANTONERA_CLASSES = 6   # 0..5

DERIVED_COLUMNS = ["Hole", "num_filas_tornillos", "tiene_cantonera"]


# ============================================================================
# 4. CATÁLOGOS COMERCIALES Y TABLAS DE MATERIALES
# ============================================================================
CATALOGO_BOLT = [16, 20, 22, 24, 27, 30, 36]

CATALOGO_g = list(range(80, 171, 5))
CATALOGO_e = list(range(35, 76, 5))
CATALOGO_Bp = list(range(140, 400, 1))
CATALOGO_tp = [12, 15, 20, 25, 30, 35, 40, 45, 50, 60]
CATALOGO_d = None
CATALOGO_Lh = None
CATALOGO_tws = [0, 6, 8, 10, 12, 15, 20]
CATALOGO_ts = [0, 10, 12, 15, 20, 25, 30, 35, 40]
CATALOGO_tbs = [0, 10, 12, 15, 20, 25, 30, 35, 40]
CATALOGO_tst = [0, 10, 15, 20]

Y_GRID_STEP = 5

# grado_acero -> (fy, fu) MPa
ACERO_TABLE = {
    "S275JR": {"fy_acero_mpa": 275.0, "fu_acero_mpa": 430.0},
    "S355JR": {"fy_acero_mpa": 355.0, "fu_acero_mpa": 490.0},
}

# clase_tornillo -> (fy, fu, fv) MPa
TORNILLO_TABLE = {
    "8.8":   {"fy_tornillo_mpa": 640.0, "fu_tornillo_mpa": 800.0, "fv_tornillo_mpa": 480.0},
    "10.9":  {"fy_tornillo_mpa": 900.0, "fu_tornillo_mpa": 1000.0, "fv_tornillo_mpa": 500.0},
    "A325M": {"fy_tornillo_mpa": 660.0, "fu_tornillo_mpa": 830.0, "fv_tornillo_mpa": 372.0},
}
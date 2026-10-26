"""
Persistencia del dataset global (filas ya VALIDADAS por un ingeniero). Como las columnas de
entrada todavía son "de prueba" y van a cambiar, el esquema se trata como dinámico: cada
`append_rows` añade las columnas que falten (con NaN en las filas antiguas) en vez de
exigir un esquema fijo desde el principio. Cuando el esquema real esté cerrado esto se puede
endurecer (columnas fijas, tipos, validación) sin tocar el resto del servidor.

Formato en disco: parquet (rápido, tipado, columnar) con el mismo criterio que ya usamos en
la app de escritorio (`model_store/global_dataset.parquet`).

Además:
  - Normaliza los nombres de columna a los canónicos de config.py (P1 -> p1, bolt -> Bolt...),
    tanto al guardar como al cargar (migra datasets ya guardados con nombres antiguos).
  - Mantiene una caché de lo predicho en /predict (entradas + etiquetas de zona pX_label, que no
    se envían a Excel), indexada por PredID (y por UNICODE como respaldo). En /validate se usa para
    completar las entradas que Excel no haya reenviado y para adjuntar los pX_label.
  - Rechaza (InvalidRowsError) filas sin las entradas mínimas, para no ensuciar el entrenamiento.
  - Solo guarda las columnas útiles (entradas del modelo, targets y unas pocas informativas):
    lo demás que envía Excel (MzEd_Porcen, Bws, Electrode, PredID, UNICODE...) se descarta.
"""
import json
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

import config

_LOCK = threading.Lock()  # evita carreras si llegan varias validaciones a la vez

_LABEL_CACHE_PATH = config.DATA_DIR / "p_label_cache.json"

# nombre en minúsculas -> nombre canónico de config.py (P1->p1, bolt->Bolt, Unicode->UNICODE, ...).
# Se excluyen los nombres de una letra: en IM_MOMENT existe la columna 'E' (módulo de Young = 200000)
# y no debe confundirse con 'e' (distancia al borde de la placa).
_NO_ALIAS = {"g", "e", "d"}
_CANON = {
    n.lower(): n
    for n in set(config.RAW_INPUT_COLUMNS) | set(config.ALL_TARGETS) | set(config.DERIVED_COLUMNS)
    if n not in _NO_ALIAS
}
_INPUT_KEYS = set(config.RAW_INPUT_COLUMNS)   # únicas claves que se completan desde la caché

# Columnas que se almacenan en el dataset global. Todo lo demás se descarta al guardar y al cargar.
#   - entradas del modelo + targets (config.py) + derivadas
#   - informativas que ya existían en train_v3: BEAM, COLUMN, Lp, Source, *_r1_mm
#   - trazabilidad: _validated_at, _source
_KEEP_COLUMNS = (
    set(config.FEATURE_COLUMNS) | set(config.ALL_TARGETS) | set(config.DERIVED_COLUMNS)
    | {"BEAM", "COLUMN", "Lp", "Source", "BEAM_r1_mm", "COLUMN_r1_mm", "_validated_at", "_source"}
)


def keep_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Se queda solo con las columnas útiles (ver _KEEP_COLUMNS), conservando su orden."""
    return df[[c for c in df.columns if c in _KEEP_COLUMNS]]
# columnas que Excel manda pero que no aportan al entrenamiento
_DROP = {"n_y"}

# Entradas mínimas para que una fila sirva de entrenamiento (las de material se pueden derivar
# de grado_acero / clase_tornillo; Capacity_* puede ser legítimamente 0).
REQUIRED_CATEGORICAL = list(config.CATEGORICAL_FEATURES)
REQUIRED_GEOMETRY = [c for c in config.NUMERIC_FEATURES if c.startswith(("BEAM_", "COLUMN_"))]
_CACHE_MAX = 5000


class InvalidRowsError(ValueError):
    """Filas validadas sin las entradas necesarias para entrenar."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================================
# Normalización de nombres
# ============================================================================
def uid_key(v: Any) -> str:
    """UNICODE como clave estable (123, 123.0 y '123' -> '123')."""
    if v is None:
        return ""
    try:
        f = float(v)
        if f != f:  # NaN
            return ""
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """Renombra las claves de una fila a los nombres canónicos y descarta las sobrantes."""
    out: Dict[str, Any] = {}
    for k, v in r.items():
        name = str(k).strip()
        if name.lower() in _DROP:
            continue
        name = _CANON.get(name.lower(), name)
        if name in out and v is None:
            continue
        out[name] = v
    return out


def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas y fusiona duplicadas (migra datasets guardados con P1, bolt, N_Y...)."""
    if df.empty:
        return df
    df = df.rename(columns={c: _CANON.get(str(c).strip().lower(), c) for c in df.columns})
    df = df.drop(columns=[c for c in df.columns if str(c).lower() in _DROP])
    if df.columns.duplicated().any():
        cols = {}
        for name in dict.fromkeys(df.columns):
            sub = df.loc[:, df.columns == name]
            cols[name] = sub.bfill(axis=1).iloc[:, 0] if sub.shape[1] > 1 else sub.iloc[:, 0]
        df = pd.DataFrame(cols)
    return keep_known_columns(df)


# ============================================================================
# Entradas que Excel no trae y se derivan de otras
#   BEAM_type / COLUMN_type <- prefijo del nombre del perfil (HEA340->HE, IPE500->IPE, W610X195->W)
#   clase_tornillo          <- BoltGrade (A325M, 8.8, 10.9...)
#   grado_acero             <- fy del acero (275->S275JR, 355->S355JR)
# ============================================================================
def _type_from_section(name: Any):
    if name is None or (isinstance(name, float) and name != name) or str(name).strip() == "":
        return None
    m = re.match(r"[A-Za-z]+", str(name).strip())
    if not m:
        return None
    p = m.group(0).upper()
    if p.startswith("HE"):
        return "HE"
    if p.startswith("IPE"):
        return "IPE"
    if p.startswith("HI"):
        return "HI"
    if p == "W":
        return "W"
    return p


def _grade_from_fy(fy: Any):
    try:
        fy = float(fy)
    except (TypeError, ValueError):
        return None
    for grade, props in config.ACERO_TABLE.items():
        if abs(props["fy_acero_mpa"] - fy) < 1e-6:
            return grade
    return None


def derive_categoricals(row: Dict[str, Any], extra: Dict[str, Any] = None) -> Dict[str, Any]:
    """Rellena (in situ) las 4 categóricas si faltan. `extra` = entradas cacheadas de /predict."""
    src = {str(k).lower(): v for k, v in {**(extra or {}), **row}.items()}

    def pick(*names):
        for n in names:
            v = src.get(n)
            if not _is_missing(v):
                return v
        return None

    if _is_missing(row.get("BEAM_type")):
        t = _type_from_section(pick("beam"))
        if t:
            row["BEAM_type"] = t
    if _is_missing(row.get("COLUMN_type")):
        t = _type_from_section(pick("column"))
        if t:
            row["COLUMN_type"] = t
    if _is_missing(row.get("clase_tornillo")):
        v = pick("boltgrade")
        if v is not None:
            row["clase_tornillo"] = str(v).strip()
    if _is_missing(row.get("grado_acero")):
        g = _grade_from_fy(pick("fy_acero_mpa", "fy"))
        if g:
            row["grado_acero"] = g
    return row


def prepare_input_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """Fila cruda de Excel -> nombres canónicos + categóricas derivadas (usar en /predict)."""
    return derive_categoricals(normalize_row(r))


# ============================================================================
# Caché de predicciones (entradas + etiquetas de zona), escrita en /predict y leída en /validate
# ============================================================================
def _read_cache() -> Dict[str, Any]:
    try:
        return json.loads(_LABEL_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def cache_predictions(items: List[Dict[str, Any]]) -> None:
    """
    items: [{"pred_id", "uid", "inputs": {...}, "n_alma", "n_cantonera", "labels": {"p1_label": ...}}]
    Se indexa por pred_id y, como respaldo, por "uid:<UNICODE>".
    """
    if not items:
        return
    ts = _now_iso()
    entries: Dict[str, Any] = {}
    for it in items:
        entry = {
            "inputs": it.get("inputs", {}),
            "n_alma": it.get("n_alma"),
            "n_cantonera": it.get("n_cantonera"),
            "labels": it.get("labels", {}),
            "pred_id": it.get("pred_id"),
            "ts": ts,
        }
        if it.get("pred_id"):
            entries[it["pred_id"]] = entry
        uid = uid_key(it.get("uid"))
        if uid:
            entries["uid:" + uid] = entry
    with _LOCK:
        cache = _read_cache()
        cache.update(entries)
        if len(cache) > _CACHE_MAX:
            newest = sorted(cache.items(), key=lambda kv: kv[1].get("ts", ""), reverse=True)[:_CACHE_MAX]
            cache = dict(newest)
        _LABEL_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _same_int(a: Any, b: Any) -> bool:
    try:
        return int(float(a)) == int(float(b))
    except (TypeError, ValueError):
        return False


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and v.strip() == "")


def attach_cached_info(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Para cada fila (ya normalizada), busca su predicción en la caché (PredID, o UNICODE como respaldo) y:
      1) completa las ENTRADAS que falten (lo que Excel sí envió manda siempre sobre la caché);
      2) añade p1_label..p12_label:
           p_i == 0                                     -> 'UNUSED'
           p_i > 0 y (n_alma, n_cantonera) sin cambios  -> etiqueta predicha en /predict
           en cualquier otro caso                       -> None (el entrenamiento la ignora)
    """
    with _LOCK:
        cache = _read_cache()
    for r in rows:
        pid = r.pop("PredID", None)
        pid = str(pid).strip() if not _is_missing(pid) else ""
        entry = cache.get(pid) if pid else None
        if entry is None:
            uid = uid_key(r.get("UNICODE"))
            entry = cache.get("uid:" + uid) if uid else None
        if pid:
            r["_pred_id"] = pid

        if entry:
            cached = entry.get("inputs", {})
            for k, v in cached.items():
                if k in _INPUT_KEYS and _is_missing(r.get(k)) and not _is_missing(v):
                    r[k] = v
            derive_categoricals(r, extra=cached)
        else:
            derive_categoricals(r)

        labels = entry.get("labels", {}) if entry else {}
        same_struct = (
            bool(entry)
            and _same_int(r.get("n_alma"), entry.get("n_alma"))
            and _same_int(r.get("n_cantonera"), entry.get("n_cantonera"))
        )
        for i in range(1, config.N_Y + 1):
            try:
                active = float(r.get(f"p{i}")) > 0
            except (TypeError, ValueError):
                active = False
            lab = labels.get(f"p{i}_label")
            if not active:
                r[f"p{i}_label"] = "UNUSED"
            elif same_struct and lab and lab != "UNUSED":
                r[f"p{i}_label"] = lab
            else:
                r[f"p{i}_label"] = None
    return rows


# ============================================================================
# Validación de entradas mínimas
# ============================================================================
def missing_inputs(r: Dict[str, Any]) -> List[str]:
    miss = []
    for c in REQUIRED_CATEGORICAL:
        v = r.get(c)
        if _is_missing(v) or str(v).strip().lower() == "nan":
            miss.append(c)
    for c in REQUIRED_GEOMETRY:
        try:
            ok = float(r.get(c)) > 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            miss.append(c)
    return miss


def valid_input_mask(df: pd.DataFrame) -> pd.Series:
    """Máscara booleana: filas de un DataFrame que traen las entradas mínimas (para entrenar)."""
    if df.empty:
        return pd.Series(dtype=bool)
    ok = pd.Series(True, index=df.index)
    for c in REQUIRED_CATEGORICAL:
        if c not in df.columns:
            return pd.Series(False, index=df.index)
        col = df[c]
        ok &= col.notna() & (col.astype(str).str.strip() != "")
    for c in REQUIRED_GEOMETRY:
        if c not in df.columns:
            return pd.Series(False, index=df.index)
        ok &= pd.to_numeric(df[c], errors="coerce") > 0
    return ok


# ============================================================================
# Dataset global
# ============================================================================
def load_dataset() -> pd.DataFrame:
    if config.DATASET_PATH.exists():
        return normalize_df_columns(pd.read_parquet(config.DATASET_PATH))
    return pd.DataFrame()


def append_rows(rows: List[Dict[str, Any]], source: str = "excel") -> Dict[str, Any]:
    """Añade filas validadas al dataset global. Devuelve un resumen (nº guardadas, total, columnas)."""
    if not rows:
        current = load_dataset()
        return {"saved": 0, "total_rows": len(current), "columns": list(current.columns)}

    # 1) nombres canónicos  2) entradas y etiquetas recuperadas de la caché del servidor
    rows = attach_cached_info([normalize_row(r) for r in rows])

    # 3) no se guarda nada si alguna fila no trae las entradas mínimas
    problems = []
    for i, r in enumerate(rows, 1):
        miss = missing_inputs(r)
        if miss:
            problems.append(f"fila {i} (UNICODE={r.get('UNICODE')}): faltan {', '.join(miss)}")
    if problems:
        shown = "; ".join(problems[:5]) + (f"; ... y {len(problems) - 5} más" if len(problems) > 5 else "")
        raise InvalidRowsError(
            "No se ha guardado ninguna fila: faltan las entradas de la unión. " + shown +
            ". Vuelve a pulsar 'Enviar a IA' y luego 'Validar y enviar' (o revisa que las tablas de entrada "
            "sean accesibles desde el libro)."
        )

    stamped = []
    for r in rows:
        stamped.append({**r, "_validated_at": _now_iso(), "_source": source})
    new_df = keep_known_columns(pd.DataFrame(stamped))

    with _LOCK:
        current = load_dataset()
        combined = pd.concat([current, new_df], ignore_index=True) if not current.empty else new_df

        # Sanear columnas de tipo object para evitar conflictos en PyArrow
        # cuando conviven números y texto en la misma columna (ej. clase_tornillo)
        for col in combined.columns:
            if combined[col].dtype == "object":
                combined[col] = combined[col].apply(
                    lambda x: str(x) if pd.notna(x) and x is not None else None
                )

        combined.to_parquet(config.DATASET_PATH, index=False)

    return {"saved": len(rows), "total_rows": len(combined), "columns": list(combined.columns)}


def summary() -> Dict[str, Any]:
    df = load_dataset()
    if df.empty:
        return {"total_rows": 0, "columns": [], "last_validated_at": None}
    return {
        "total_rows": len(df),
        "columns": [c for c in df.columns if not c.startswith("_")],
        "last_validated_at": df["_validated_at"].max() if "_validated_at" in df.columns else None,
    }

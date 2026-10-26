"""Adaptador del modelo V11 (v4: p discretizada + topologia conjunta).

Interfaz unica que consume el servidor de produccion:
  - input_schema(): campos de entrada del modelo (tipo, descripcion y opciones).
  - preprocess()/to_internal(): input -> FEATURE_COLUMNS (lookup Tekla + materiales).
  - from_internal(): salidas internas -> claves canonicas del diseno.
  - load() / predict(): pesos + preprocesador e inferencia.

La traduccion a las 63 columnas de CODAME la hace el SERVIDOR (`codame.py`), que es la
fuente unica para todos los modelos.

Estructura del paquete instalado:

    models/V11/
        weights.pt
        preprocessor.joblib
        data/                   <- catalogos Tekla (para construir features)
        core/adapter.py         <- este archivo
"""

from pathlib import Path

import joblib
import logging
import numpy as np
import pandas as pd
import torch

from .config import (
    CATEGORICAL_FEATURES,
    N_Y,
    NUMERIC_FEATURES,
)
from .dataset import Preprocessor
from .model import ConnectionDesignNetV4
from .postprocess import decode_predictions, derive_dependent_columns

MODEL_NAME = "V11"

log = logging.getLogger("model-adapter")

# ---------------------------------------------------------------------------
# Contrato de entrada (lo define el modelo): sus inputs + UNICODE.
# ---------------------------------------------------------------------------
# `ratio_*` en escala 0-1 (igual que en entrenamiento). `Normativas` alimenta la
# columna `Normativas` de CODAME (la `standard` predicha NO se mapea a `Normativas`).
RAW_INPUT_COLUMNS = [
    "UNICODE",
    "Normativas",
    "BEAM",
    "COLUMN",
    "ratio_moment",
    "ratio_shear",
    "ratio_axial",
    "grado_acero",
    "clase_tornillo",
]

# Opcionales (con default): si no llegan, no bloquean la prediccion.
OPTIONAL_INPUT_COLUMNS = ["Normativas", "grado_acero", "clase_tornillo"]

DEFAULT_GRADO_ACERO = "S355JR"
DEFAULT_CLASE_TORNILLO = "A325M"

# Columnas completas que consume el modelo internamente (no forman parte de la tabla).
FEATURE_COLUMNS = list(CATEGORICAL_FEATURES) + list(NUMERIC_FEATURES)

# Directorios y artefactos relativos al paquete del modelo (models/V11/ o V11/).
MODEL_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = MODEL_DIR / "weights.pt"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
DATA_DIR = MODEL_DIR / "data"

_TEKLA_FILES = ("tekla_eu.csv", "tekla_cad.csv", "tekla_faltante.csv", "tekla_uk.csv")

# Catalogos de materiales (referencia: V7/variables.py).
ACERO_TABLE = {
    "S275JR": {"fy_acero_mpa": 275.0, "fu_acero_mpa": 430.0},
    "S355JR": {"fy_acero_mpa": 355.0, "fu_acero_mpa": 490.0},
}
TORNILLO_TABLE = {
    "8.8": {"fy_tornillo_mpa": 640.0, "fu_tornillo_mpa": 800.0, "fv_tornillo_mpa": 480.0},
    "10.9": {"fy_tornillo_mpa": 900.0, "fu_tornillo_mpa": 1000.0, "fv_tornillo_mpa": 500.0},
    "A325M": {"fy_tornillo_mpa": 660.0, "fu_tornillo_mpa": 830.0, "fv_tornillo_mpa": 372.0},
}

OUTPUT_FIELDS = (
    ["standard", "Bolt", "SlipCritical", "n_alma", "n_cantonera"]
    + ["g", "e", "d", "Bp", "tp", "tws", "ts", "tbs", "tst"]
    + [f"p{i}" for i in range(1, N_Y + 1)]
    + [f"p{i}_transition" for i in range(1, N_Y + 1)]
    + ["Hole", "num_filas_tornillos", "tiene_cantonera"]
)


class InputContractError(ValueError):
    """La fila recibida no encaja con el contrato de entrada del modelo."""


# ---------------------------------------------------------------------------
# Catalogo de perfiles
# ---------------------------------------------------------------------------
_PROFILE_LOOKUP = None


def _profile_lookup() -> pd.DataFrame:
    global _PROFILE_LOOKUP
    if _PROFILE_LOOKUP is None:
        frames = [pd.read_csv(DATA_DIR / name) for name in _TEKLA_FILES]
        union = pd.concat(frames, ignore_index=True)
        union = union.dropna(subset=["tekla_nomenclature_1"])   # <- nuevo
        union["_key"] = union["tekla_nomenclature_1"].astype(str).str.strip().str.upper()
        _PROFILE_LOOKUP = union.drop_duplicates(subset="_key").set_index("_key")
    return _PROFILE_LOOKUP


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra alias (Unicode, beam, ...) a los nombres canonicos del contrato."""
    alias = {c.lower(): c for c in RAW_INPUT_COLUMNS + OPTIONAL_INPUT_COLUMNS}
    return df.rename(
        columns={c: alias.get(str(c).strip().lower(), c) for c in df.columns}
    )


def _add_profiles(df: pd.DataFrame) -> pd.DataFrame:
    lookup = _profile_lookup()
    for col in ("BEAM", "COLUMN"):
        keys = df[col].astype(str).str.strip().str.upper()
        unknown = sorted(set(df.loc[~keys.isin(lookup.index), col].astype(str)))
        if unknown:
            raise InputContractError(
                f"Perfiles no reconocidos en {col} (no estan en los catalogos "
                f"Tekla): {unknown}"
            )
        geom = lookup.loc[keys].reset_index(drop=True)
        df[f"{col}_type"] = geom["type"].to_numpy()
        df[f"{col}_d_mm"] = pd.to_numeric(geom["d_mm"].to_numpy(), errors="coerce")
    return df


def _add_materials(df: pd.DataFrame) -> pd.DataFrame:
    for column, table, default in (
        ("grado_acero", ACERO_TABLE, DEFAULT_GRADO_ACERO),
        ("clase_tornillo", TORNILLO_TABLE, DEFAULT_CLASE_TORNILLO),
    ):
        if column not in df.columns:
            df[column] = default
        values = (
            df[column]
            .replace(r"^\s*$", np.nan, regex=True)
            .fillna(default)
            .astype(str)
            .str.strip()
        )
        unknown = sorted({v for v in values if v not in table})
        if unknown:
            raise InputContractError(
                f"Valores no reconocidos en {column}: {unknown}. "
                f"Admitidos: {sorted(table)}"
            )
        df[column] = values.to_numpy()
        props = pd.DataFrame([table[v] for v in values], index=df.index)
        for name in props.columns:
            df[name] = props[name].to_numpy()
    return df


def _to_number(series: pd.Series, column: str) -> pd.Series:
    """Convierte a numero aceptando coma decimal; error de contrato si no es posible."""
    cleaned = series.map(lambda v: v.replace(",", ".") if isinstance(v, str) else v)
    try:
        return pd.to_numeric(cleaned, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise InputContractError(f"{column} debe ser numerico: {exc}") from exc


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Input basico -> dataframe con las FEATURE_COLUMNS completas del modelo."""
    df = _canonicalize(raw_df.copy())
    df = _add_profiles(df)
    df = _add_materials(df)
    for col in ("ratio_moment", "ratio_shear", "ratio_axial"):
        df[col] = _to_number(df[col], col)

    keep = list(
        dict.fromkeys(
            [c for c in RAW_INPUT_COLUMNS if c in df.columns] + list(FEATURE_COLUMNS)
        )
    )
    return df[keep]


def to_internal(table_df: pd.DataFrame) -> pd.DataFrame:
    """Esquema de tabla (inputs) -> features internas que consume el modelo."""
    return preprocess(table_df)


def from_internal(design_df: pd.DataFrame) -> pd.DataFrame:
    """Salidas internas -> claves canonicas del diseno; rellena con 0 lo que falte."""
    out = design_df.copy()
    for column in OUTPUT_FIELDS:
        if column not in out.columns:
            out[column] = 0
    return out[OUTPUT_FIELDS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Carga e inferencia
# ---------------------------------------------------------------------------
def load() -> dict:
    """Carga pesos + preprocesador y devuelve el bundle de inferencia."""
    if not WEIGHTS_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Faltan artefactos del modelo en {MODEL_DIR}. "
            f"Ejecuta 'make install-model MODEL=V11' desde la raiz del proyecto."
        )

    pre: Preprocessor = joblib.load(PREPROCESSOR_PATH)

    # weights_only=False: el checkpoint lleva arrays numpy en los metadatos
    # (p_mean_mm/p_std_mm). Es un artefacto propio y de confianza.
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    cfg = checkpoint.get("cfg", {})

    model = ConnectionDesignNetV4(
        cat_cardinalities=checkpoint["cat_cardinalities"],
        num_continuous=len(checkpoint["numeric_features"]),
        n_standard_classes=checkpoint["n_standard_classes"],
        p_mean_mm=checkpoint["p_mean_mm"],
        p_std_mm=checkpoint["p_std_mm"],
        hidden_dim=checkpoint.get("hidden_dim", 192),
        num_blocks=checkpoint.get("num_blocks", 3),
        dropout=checkpoint.get("dropout", 0.1),
        gru_hidden_dim=checkpoint.get("gru_hidden_dim", 128),
        n_p_bins=checkpoint["n_p_bins"],
        p_bin_step_mm=checkpoint["p_bin_step_mm"],
        joint_topology=bool(cfg.get("joint_topology", False)),
    )
    model.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return {"model": model, "pre": pre, "device": device}


@torch.no_grad()
def predict(features_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Features completas -> dataframe con las predicciones decodificadas."""
    model = bundle["model"]
    pre: Preprocessor = bundle["pre"]
    device = bundle["device"]

    try:
        x_cat, x_num = pre.transform_features(features_df)
    except ValueError as exc:
        raise InputContractError(f"Valores no admitidos por el modelo: {exc}") from exc

    outputs = model(
        torch.as_tensor(x_cat, dtype=torch.long, device=device),
        torch.as_tensor(x_num, dtype=torch.float32, device=device),
    )
    outputs = {k: v.detach().cpu().numpy() for k, v in outputs.items()}

    presence = 1.0 / (1.0 + np.exp(-np.clip(outputs["masked_presence_logit"], -30, 30)))
    design = decode_predictions(
        reg_always_real=pre.inverse_reg_always(outputs["reg_always"]),
        masked_value_real=pre.inverse_masked(outputs["masked_value"]),
        masked_presence_prob=presence,
        bolt_logits=outputs["bolt_logits"],
        n_alma_logits=outputs["n_alma_logits"],
        n_cantonera_logits=outputs["n_cantonera_logits"],
        standard_logits=outputs["standard_logits"],
        standard_classes=pre.standard_encoder.classes_,
        slip_critical_logit=outputs["slip_critical_logit"],
    )
    design = derive_dependent_columns(design)
    return design.drop(columns=["_bolt_nominal_mm"], errors="ignore")


# ---------------------------------------------------------------------------
# Contrato, schema de entrada y opciones
# ---------------------------------------------------------------------------
def contract() -> dict:
    """Contrato de tabla del modelo (inputs + opcionales + outputs)."""
    return {
        "name": MODEL_NAME,
        "raw_inputs": list(RAW_INPUT_COLUMNS),
        "optional_inputs": list(OPTIONAL_INPUT_COLUMNS),
        "outputs": list(OUTPUT_FIELDS),
    }


def _pre(bundle: dict = None) -> Preprocessor:
    pre = (bundle or {}).get("pre")
    return pre if pre is not None else joblib.load(PREPROCESSOR_PATH)


def _profile_names() -> list:
    names = _profile_lookup()["tekla_nomenclature_1"].astype(str).str.strip()
    return sorted(n for n in names.unique() if n)


def input_schema(bundle: dict = None) -> list:
    """Campos de entrada del modelo: key, descripcion, tipo y opciones si categorico."""
    pre = _pre(bundle)
    perfiles = _profile_names()
    return [
        {"key": "UNICODE", "description": "Identificador unico",
         "type": "text"},
        {"key": "Normativas", "description": "Normativa aplicable",
         "type": "categorical",
         "options": [str(c) for c in pre.standard_encoder.classes_]},
        {"key": "BEAM", "description": "Perfil de la viga", "type": "categorical",
         "options": perfiles},
        {"key": "COLUMN", "description": "Perfil de la columna", "type": "categorical",
         "options": perfiles},
        {"key": "ratio_moment", "description": "(0-1)",
         "type": "number"},
        {"key": "ratio_shear", "description": "(0-1)",
         "type": "number"},
        {"key": "ratio_axial", "description": "(0-1)",
         "type": "number"},
        {"key": "grado_acero", "description": "Grado del acero", "type": "categorical",
         "options": [str(c) for c in pre.cat_encoders["grado_acero"].classes_]},
        {"key": "clase_tornillo", "description": "Clase del tornillo",
         "type": "categorical",
         "options": [str(c) for c in pre.cat_encoders["clase_tornillo"].classes_]},
    ]


def options(bundle: dict = None) -> dict:
    """Valores validos (compatibilidad). El servidor usa input_schema()."""
    pre = _pre(bundle)
    perfiles = _profile_names()
    return {
        "BEAM": perfiles,
        "COLUMN": perfiles,
        "grado_acero": [str(c) for c in pre.cat_encoders["grado_acero"].classes_],
        "clase_tornillo": [str(c) for c in pre.cat_encoders["clase_tornillo"].classes_],
        "defaults": {
            "grado_acero": DEFAULT_GRADO_ACERO,
            "clase_tornillo": DEFAULT_CLASE_TORNILLO,
        },
    }

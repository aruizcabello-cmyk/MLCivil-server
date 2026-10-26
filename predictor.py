"""Inferencia y comprobacion estructural en produccion sobre modelos intercambiables.

Cada modelo expone su `input_schema()` y su diseno en claves canonicas. El servidor:
  - valida el contrato de entrada y enruta al modelo,
  - mapea input + diseno a las 63 columnas de CODAME (`codame.to_codame`, catalogo del
    servidor) e imputa lo que falte (nunca responde 500 por culpa del modelo),
  - ejecuta la comprobacion AISC 360-10 (`checker`) sobre el esquema CODAME.
"""

import logging
import math
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import batcher
import codame
import config
import model_registry
import refinar
from codame_schema import CODAME_COLUMNS, fill_missing
from checker import check_rows as perform_check

log = logging.getLogger("ai-server")

# La carga del modelo y el preproceso comparten estado global (model_registry purga
# `sys.modules["core"]`). La inferencia en si la serializa el batcher dinamico, que
# agrupa peticiones concurrentes en un unico forward.
_PREDICT_LOCK = threading.Lock()
_BATCHER_LOCK = threading.Lock()
_INFERENCE_BATCHER: Optional[batcher.InferenceBatcher] = None


def start_inference(max_rows: Optional[int] = None, max_wait_ms: Optional[int] = None) -> None:
    """Arranca el worker de inferencia (idempotente)."""
    global _INFERENCE_BATCHER
    with _BATCHER_LOCK:
        if _INFERENCE_BATCHER is not None:
            return
        _INFERENCE_BATCHER = batcher.InferenceBatcher(
            max_rows=config.BATCH_MAX_ROWS if max_rows is None else max_rows,
            max_wait_ms=config.BATCH_MAX_WAIT_MS if max_wait_ms is None else max_wait_ms,
        )
        _INFERENCE_BATCHER.start()
        log.info(
            "batcher: iniciado (max_rows=%d, max_wait_ms=%d)",
            _INFERENCE_BATCHER.max_rows, _INFERENCE_BATCHER.max_wait_ms,
        )


def shutdown_inference() -> None:
    global _INFERENCE_BATCHER
    with _BATCHER_LOCK:
        if _INFERENCE_BATCHER is not None:
            log.info("batcher: cerrando")
            _INFERENCE_BATCHER.close()
            _INFERENCE_BATCHER = None


def _get_batcher() -> batcher.InferenceBatcher:
    if _INFERENCE_BATCHER is None:
        start_inference()
    assert _INFERENCE_BATCHER is not None
    return _INFERENCE_BATCHER


class InputContractError(ValueError):
    """Los inputs recibidos no cumplen el contrato del modelo seleccionado."""

    def __init__(self, message: str, model: str = None, missing=None, expected=None):
        super().__init__(message)
        self.model = model
        self.missing = missing
        self.expected = expected


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _clean_value(value: Any) -> Any:
    """Garantiza compatibilidad JSON (sin NaN/inf)."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return None if (math.isnan(value) or math.isinf(value)) else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _clean_value(v) for k, v in data.items()}


def _normalize_rows(
    rows: List[Dict[str, Any]], raw_inputs: List[str], optional_inputs: List[str]
) -> List[Dict[str, Any]]:
    """Renombra alias (Unicode -> UNICODE...) sin distinguir mayusculas/espacios."""
    alias = {c.lower(): c for c in list(raw_inputs) + list(optional_inputs)}
    normalized = []
    for row in rows:
        new_row: Dict[str, Any] = {}
        for key, value in row.items():
            new_row[alias.get(str(key).strip().lower(), key)] = value
        normalized.append(new_row)
    return normalized


def _validate_contract(
    rows: List[Dict[str, Any]], raw_inputs: List[str], optional_inputs: List[str], model: str
) -> None:
    if not rows:
        raise InputContractError("La peticion no trae filas (rows vacio)", model=model, expected=raw_inputs)
    optional = set(optional_inputs)
    required = [c for c in raw_inputs if c not in optional]  # UNICODE siempre obligatorio
    missing = {}
    for i, row in enumerate(rows):
        absent = [c for c in required if _is_missing(row.get(c))]
        if absent:
            missing[f"fila {i + 1}"] = absent
    if missing:
        raise InputContractError(
            f"Faltan inputs requeridos por el modelo '{model}'",
            model=model,
            missing=missing,
            expected=raw_inputs,
        )


def predict_rows(model_name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _predict(model_name, rows, with_checks=False)[0]


def predict_and_check(model_name: str, rows: List[Dict[str, Any]]):
    """Prediccion + ajuste + comprobacion: (filas, comprobaciones)."""
    output, checks, _ = _predict(model_name, rows, with_checks=True)
    return output, checks


def predict_full(model_name: str, rows: List[Dict[str, Any]]):
    """Prediccion + ajuste + comprobacion + filas RAW (sin ajustes): (filas, checks, raw)."""
    return _predict(model_name, rows, with_checks=True)


def predict_detailed(model_name: str, rows: List[Dict[str, Any]], maximize_load: bool = False):
    """Para los escenarios de carga. Por fila: {"row", "check", "raw", "n_cantonera"}.
    `maximize_load=False`: NO se reescriben las cargas de entrada (ver escenarios.py)."""
    detail: List[Dict[str, Any]] = []
    output, checks, raw = _predict(model_name, rows, True, maximize_load=maximize_load, detail=detail)
    return [{"row": o, "check": c, "raw": r, "n_cantonera": d["n_cantonera"]}
            for o, c, r, d in zip(output, checks, raw, detail)]


def _predict(model_name: str, rows: List[Dict[str, Any]], with_checks: bool,
             maximize_load: Optional[bool] = None, detail: Optional[list] = None):
    # Solo se serializa la carga/preproceso: la inferencia la agrupa el batcher y el
    # refinado (la parte cara) queda fuera del lock.
    with _PREDICT_LOCK:
        bundle = model_registry.get(model_name)
        adapter = bundle["adapter"]
        contract = bundle["contract"]
        raw_inputs = contract["raw_inputs"]
        optional_inputs = contract.get("optional_inputs", [])
        name = bundle["name"]

        rows = _normalize_rows(rows, raw_inputs, optional_inputs)
        _validate_contract(rows, raw_inputs, optional_inputs, name)

        df = pd.DataFrame(rows)
        try:
            features = adapter.to_internal(df)
        except adapter.InputContractError as exc:
            raise InputContractError(str(exc), model=name, expected=raw_inputs) from exc

    timeout = config.INFERENCE_TIMEOUT_S if config.INFERENCE_TIMEOUT_S > 0 else None
    future = _get_batcher().submit(bundle, features)
    try:
        design = future.result(timeout=timeout)
    except adapter.InputContractError as exc:
        raise InputContractError(str(exc), model=name, expected=raw_inputs) from exc
    except FutureTimeoutError:
        future.cancel()
        raise TimeoutError(
            f"inferencia: tiempo de espera agotado ({config.INFERENCE_TIMEOUT_S:.0f}s)"
        ) from None

    coded: List[Dict[str, Any]] = []
    raw: List[Dict[str, Any]] = []
    # Una sola conversion a dicts (antes: iloc[i].to_dict() por fila, muy lento con pandas).
    input_rows = features.to_dict("records")
    design_rows = design.to_dict("records")
    for i in range(len(df)):
        input_row = input_rows[i]
        design_row = design_rows[i]
        if detail is not None:
            try:
                detail.append({"n_cantonera": int(round(float(design_row.get("n_cantonera") or 0)))})
            except (TypeError, ValueError):
                detail.append({"n_cantonera": 0})
        # Prediccion RAW: sin ajustes de chapa/cantonera ni pfi (para comparar en el visor).
        try:
            raw_partial = codame.to_codame(input_row, design_row, apply_adjustments=False)
            if not isinstance(raw_partial, dict):
                raise TypeError("codame.to_codame no devolvio un dict")
        except Exception:  # noqa: BLE001
            log.exception("codame.to_codame(raw) fallo (modelo=%s, fila=%d)", name, i)
            raw_partial = {}
        raw.append(fill_missing(raw_partial))
        # Prediccion AJUSTADA (comportamiento normal).
        try:
            partial = codame.to_codame(input_row, design_row)
            if not isinstance(partial, dict):
                raise TypeError("codame.to_codame no devolvio un dict")
        except Exception:  # noqa: BLE001  (nunca fallar por culpa del modelo)
            log.exception(
                "codame.to_codame fallo (modelo=%s, fila=%d); se devuelven valores por defecto",
                name,
                i,
            )
            partial = {}
        coded.append(fill_missing(partial))

    # Ajuste de placas/cargas (+ comprobacion final): una union por proceso (ver refinar.py).
    refined = refinar.refine_many(coded, with_check=with_checks, maximize_load=maximize_load)

    output: List[Dict[str, Any]] = []
    checks: List[Dict[str, Any]] = []
    for i, (row, check) in enumerate(refined):
        output.append(_clean_dict(row))
        if with_checks:
            checks.append(_clean_dict(check) if check is not None else _check_one(i, output[-1]))
    rows_raw = [_clean_dict(r) for r in raw]
    return output, checks, rows_raw


def _check_one(i: int, row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _clean_dict(perform_check([row])[0])
    except Exception as exc:  # noqa: BLE001  (informe de error por entrada)
        log.exception("check_rows fallo (fila=%d)", i)
        return {
            "Unicode": row.get("Unicode", row.get("UNICODE", "")),
            "cumple": False,
            "ratio_pesimo": None,
            "descripcion_ratio_pesimo": f"Error en la comprobacion: {exc}",
            "ratios": [],
            "complementarios": [],
        }


def check_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Comprobacion AISC 360-10 sobre filas del esquema CODAME (editadas)."""
    return [_check_one(i, row) for i, row in enumerate(rows)]


def schema(model_name: str) -> Dict[str, Any]:
    """Inputs del modelo + columnas de resultado (comunes)."""
    bundle = model_registry.get(model_name)
    adapter = bundle["adapter"]
    fn = getattr(adapter, "input_schema", None)
    inputs = fn(bundle) if fn is not None else []
    return {
        "model": bundle["name"],
        "inputs": inputs,
        "result_columns": list(CODAME_COLUMNS),
    }


def model_status() -> Dict[str, Any]:
    return {
        "available_models": model_registry.list_models(),
        "loaded_models": model_registry.loaded_models(),
    }
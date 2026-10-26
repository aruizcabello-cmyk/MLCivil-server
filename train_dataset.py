"""Traduccion de filas servidas (CODAME + extras) al esquema de reentreno y persistencia.

- `to_train_row(row)`: fila servida -> columnas `train_schema.TRAIN_COLUMNS` (con tipo y
  NULL cuando falta el dato). NO se guardan columnas derivadas (BEAM_*/COLUMN_*/fy/fu/fv);
  solo se usa el catalogo para derivar `BEAM_tf` y calcular `p1`.
- `append_to_parquet(rows)`: anade al parquet unificado reemplazando `Unicode` duplicados.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import config
from catalogo import ACERO_TABLE, get_profile
from train_schema import BOOL_COLUMNS, NUMERIC_COLUMNS, TEXT_COLUMNS, TRAIN_COLUMNS

_LOCK = threading.Lock()


def _num(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _text(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none") else None


def _bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().upper()
    if s in ("TRUE", "YES", "1", "Y"):
        return True
    if s in ("FALSE", "NO", "0", "N"):
        return False
    return None


def _parse_mm(v):
    """'20mm' / '20' -> 20.0 ; vacio -> None."""
    if v is None:
        return None
    s = str(v).strip().lower().replace("mm", "").replace("t", "").strip()
    if s == "":
        return None
    return _num(s)


def _bolt_int(v):
    """'M20' / 20 -> 20."""
    if v is None:
        return None
    return _num(str(v).strip().upper().replace("M", ""))


def to_train_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fila servida (CODAME + extras) -> dict con las columnas de TRAIN_COLUMNS."""
    r = row or {}

    beam = _text(r.get("BEAM"))
    geom = get_profile(beam) if beam else None
    tf_b = _num(geom.get("tf_mm")) if geom is not None else None

    pfo = _num(r.get("pfo"))
    pfi = _num(r.get("pfi"))
    p1 = (pfo + pfi + (tf_b or 0.0)) if (pfo is not None and pfi is not None) else None

    bolt = _bolt_int(r.get("BoltName_ep") or r.get("Bolt"))
    hole = (bolt + 2) if bolt is not None else None

    grado = _text(r.get("AceroEstructural") or r.get("grado_acero"))
    fy = _num(r.get("fy"))
    if grado is None and fy is not None:
        for k, v in ACERO_TABLE.items():
            if abs(v["fy_acero_mpa"] - fy) < 1e-6:
                grado = k
                break
    clase = _text(r.get("BoltGrade") or r.get("EstandarTornillos"))

    out = {
        # demanda
        "Capacity_Moment_Percentage": _num(r.get("MzEd_Porcen")),
        "Capacity_Moment": _num(r.get("Capacity_Moment")),
        "Capacity_Shear_Percentage": _num(r.get("FyEd_Porcen")),
        "Capacity_Shear": _num(r.get("Capacity_Shear")),
        "Capacity_Axial_Percentage": _num(r.get("FxEd_Porcen")),
        "Capacity_Axial": _num(r.get("Capacity_Axial")),
        # diseno
        "g": _num(r.get("p2_ep")),
        "tws": _parse_mm(r.get("ThicknessName_t_webstiff")),
        "Bp": _num(r.get("b_ep")),
        "Lp": _num(r.get("Lp")),
        "tp": _parse_mm(r.get("ThicknessName_t_ep")),
        "Lh": _num(r.get("Lh")),
        "e": _num(r.get("dev")),
        "d": pfo,
        "p1": p1,
        "p2": _num(r.get("pb")),
        "ts": _parse_mm(r.get("ThicknessName_t_compstiff")),
        "tbs": _parse_mm(r.get("ThicknessName_t_BeamHcompstiff")),
        "tst": _parse_mm(r.get("ThicknessName_t_tenstiff")),
        "Hole": hole,
        # identificacion / metadatos
        "Unicode": _text(r.get("Unicode")),
        "BEAM": beam,
        "COLUMN": _text(r.get("COLUMN")),
        "DrawingCode": _text(r.get("DrawingCode")),
        "Bolt": bolt,
        "SlipCritical": _bool(r.get("SlipCritical")),
        "Normativas": _text(r.get("Normativas")),
        "AceroEstructural": grado,
        "EstandarTornillos": clase,
        "Source": _text(r.get("Source")),
        "National_Annex": _text(r.get("National_Annex")),
    }
    for n in range(3, 13):
        out[f"p{n}"] = _num(r.get(f"p{n}"))

    # Coercion final por tipo.
    for c in NUMERIC_COLUMNS:
        out[c] = _num(out.get(c))
    for c in BOOL_COLUMNS:
        out[c] = _bool(out.get(c))
    for c in TEXT_COLUMNS:
        out[c] = _text(out.get(c))

    return {c: out.get(c) for c in TRAIN_COLUMNS}


def append_to_parquet(rows: List[Dict[str, Any]], path: Path = None) -> Dict[str, Any]:
    """Traduce `rows`, reemplaza `Unicode` duplicados y guarda el parquet unificado."""
    path = Path(path or config.DATASET_PATH)
    new_df = pd.DataFrame([to_train_row(r) for r in rows], columns=TRAIN_COLUMNS)

    with _LOCK:
        if path.exists():
            current = pd.read_parquet(path).reindex(columns=TRAIN_COLUMNS)
        else:
            current = pd.DataFrame(columns=TRAIN_COLUMNS)

        existing = set(current["Unicode"].dropna().astype(str))
        incoming = set(new_df["Unicode"].dropna().astype(str))
        added = len(incoming - existing)
        updated = len(incoming & existing)

        combined = pd.concat([current, new_df], ignore_index=True)
        if not combined.empty and combined["Unicode"].notna().any():
            combined = combined.drop_duplicates(subset="Unicode", keep="last")

        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False)

    return {"added": added, "updated": updated, "total_rows": int(len(combined))}

"""Catalogo de perfiles y materiales a nivel de servidor.

Fuente de datos: `production/server/data` (catalogos Tekla). Se usa tanto para el mapeo
a CODAME como para la comprobacion estructural, con independencia del modelo activo.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

import config

DATA_DIR = config.DATA_DIR
_TEKLA_FILES = ("tekla_eu.csv", "tekla_cad.csv", "tekla_faltante.csv", "tekla_uk.csv")

# grado_acero -> (fy, fu) MPa
ACERO_TABLE = {
    "S275JR": {"fy_acero_mpa": 275.0, "fu_acero_mpa": 430.0},
    "S355JR": {"fy_acero_mpa": 355.0, "fu_acero_mpa": 490.0},
}

# clase_tornillo -> (fy, fu, fv) MPa
TORNILLO_TABLE = {
    "8.8": {"fy_tornillo_mpa": 640.0, "fu_tornillo_mpa": 800.0, "fv_tornillo_mpa": 480.0},
    "10.9": {"fy_tornillo_mpa": 900.0, "fu_tornillo_mpa": 1000.0, "fv_tornillo_mpa": 500.0},
    "A325M": {"fy_tornillo_mpa": 660.0, "fu_tornillo_mpa": 830.0, "fv_tornillo_mpa": 372.0},
}

DEFAULT_GRADO_ACERO = "S355JR"
DEFAULT_CLASE_TORNILLO = "A325M"

_LOOKUP: pd.DataFrame | None = None


def profile_lookup() -> pd.DataFrame:
    global _LOOKUP
    if _LOOKUP is None:
        frames = [pd.read_csv(DATA_DIR / name) for name in _TEKLA_FILES]
        union = pd.concat(frames, ignore_index=True)
        union = union.dropna(subset=["tekla_nomenclature_1"])   # <- nuevo
        union["_key"] = union["tekla_nomenclature_1"].astype(str).str.strip().str.upper()
        _LOOKUP = union.drop_duplicates(subset="_key").set_index("_key")
    return _LOOKUP


def profile_names() -> list:
    names = profile_lookup()["tekla_nomenclature_1"].astype(str).str.strip()
    return sorted(n for n in names.unique() if n)


@lru_cache(maxsize=4096)
def _profile_by_key(key: str):
    lut = profile_lookup()
    return lut.loc[key].to_dict() if key in lut.index else None


def get_profile(name):
    """Dict de geometria de un perfil (o None si no existe).

    Cacheado por nombre normalizado: antes devolvia una Series de pandas (`.loc` en cada
    llamada, varias por fila y por iteracion del refinado). Los llamadores solo leen (`.get`,
    `in`), asi que el dict se comparte: NO modificarlo.
    """
    return _profile_by_key(str(name).strip().upper())
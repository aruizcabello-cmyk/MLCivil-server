import numpy as np
import pandas as pd
from .config import (
    CATALOGO_BOLT,
    CATALOGO_g,
    CATALOGO_e,
    CATALOGO_d,
    CATALOGO_Bp,
    CATALOGO_tp,
    CATALOGO_tws,
    CATALOGO_ts,
    CATALOGO_tbs,
    CATALOGO_tst,
    REGRESSION_ALWAYS,
    P_COLUMNS,
    STIFFENER_COLUMNS,
    Y_GRID_STEP,
    N_Y,
    secuencia_transiciones,
)

CATALOGS = {
    "g": CATALOGO_g,
    "e": CATALOGO_e,
    "d": CATALOGO_d,
    "Bp": CATALOGO_Bp,
    "tp": CATALOGO_tp,
    "tws": CATALOGO_tws,
    "ts": CATALOGO_ts,
    "tbs": CATALOGO_tbs,
    "tst": CATALOGO_tst,
}


def snap_to_catalog(values, catalog):
    values = np.asarray(values, float)
    if catalog is None:
        return values
    arr = np.asarray(catalog, float)
    return arr[np.abs(values[:, None] - arr[None, :]).argmin(1)]


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def decode_predictions(
    reg_always_real,
    masked_value_real,
    masked_presence_prob,
    bolt_logits,
    n_alma_logits,
    n_cantonera_logits,
    standard_logits=None,
    standard_classes=None,
    slip_critical_logit=None,
    thresholds=None,
):
    n = len(reg_always_real)
    if thresholds is None:
        thresholds = np.full(masked_value_real.shape[1], 0.5)
    out = {}
    for j, name in enumerate(REGRESSION_ALWAYS):
        out[name] = snap_to_catalog(reg_always_real[:, j], CATALOGS.get(name))

    na = n_alma_logits.argmax(1)
    nc = n_cantonera_logits.argmax(1)
    out["n_alma"], out["n_cantonera"] = na, nc
    k = na + nc + 1
    p = np.maximum(
        np.round(masked_value_real[:, :N_Y] / Y_GRID_STEP) * Y_GRID_STEP, Y_GRID_STEP
    )
    active = np.arange(N_Y)[None, :] < k[:, None]
    p = np.where(active, p, 0.0)
    for j, name in enumerate(P_COLUMNS):
        out[name] = p[:, j]

    probs = masked_presence_prob[:, N_Y:]
    vals = masked_value_real[:, N_Y:]
    for j, name in enumerate(STIFFENER_COLUMNS):
        snapped = snap_to_catalog(vals[:, j], CATALOGS[name])
        out[name] = np.where(probs[:, j] >= thresholds[N_Y + j], snapped, 0.0)

    bolt_idx = bolt_logits.argmax(1)
    out["Bolt"] = np.asarray(CATALOGO_BOLT)[bolt_idx]
    out["_bolt_nominal_mm"] = out["Bolt"].astype(int)
    if standard_logits is not None and standard_classes is not None:
        out["standard"] = np.asarray(standard_classes)[standard_logits.argmax(1)]
    if slip_critical_logit is not None:
        out["SlipCritical"] = np.where(
            _sigmoid(slip_critical_logit) >= 0.5, "YES", "NO"
        )

    for i in range(1, N_Y + 1):
        out[f"p{i}_transition"] = []
    for row in range(n):
        trans = secuencia_transiciones(int(na[row]), int(nc[row]))
        for j in range(N_Y):
            out[f"p{j + 1}_transition"].append(trans[j] if j < len(trans) else "UNUSED")
    return pd.DataFrame(out)


def derive_dependent_columns(df):
    out = df.copy().reset_index(drop=True)
    out["Hole"] = out["_bolt_nominal_mm"].astype(int) + 2
    out["num_filas_tornillos"] = (
        out["n_alma"].astype(int) + out["n_cantonera"].astype(int) + 2
    )
    out["tiene_cantonera"] = out["n_cantonera"].astype(int) > 0
    return out

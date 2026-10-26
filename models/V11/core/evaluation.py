import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import (
    REGRESSION_ALWAYS,
    MASKED_COLUMNS,
    P_COLUMNS,
    N_Y,
    secuencia_transiciones,
)


def _matriz_transiciones(df):
    """(n, N_Y) con el nombre de la transicion de cada p.

    Usa la columna pX_transition del dataset cuando existe (columna a columna, como
    hacia el notebook original) y, si falta, la deriva de n_alma / n_cantonera."""
    derivada = np.full((len(df), N_Y), "UNUSED", dtype=object)
    for r, (a, c) in enumerate(zip(df["n_alma"].astype(int), df["n_cantonera"].astype(int))):
        for j, t in enumerate(secuencia_transiciones(a, c)):
            derivada[r, j] = t
    out = derivada.copy()
    for j in range(N_Y):
        col = f"p{j + 1}_transition"
        if col in df.columns:
            out[:, j] = df[col].fillna("UNUSED").astype(str).to_numpy()
    return out


def preparar_para_evaluar(test_df, full_design):
    cols = REGRESSION_ALWAYS + MASKED_COLUMNS
    real = (
        test_df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .reset_index(drop=True)
    )
    pred = (
        full_design[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .reset_index(drop=True)
    )
    return real, pred


def evaluar_predicciones(df_real, df_pred, tol_taller=(5.0, 10.0)):
    rows = []
    for col in [c for c in df_real.columns if c in df_pred.columns]:
        y = df_real[col].to_numpy()
        p = df_pred[col].to_numpy()
        if col in MASKED_COLUMNS:
            active = y > 0
            if active.sum() == 0:
                continue
            y, p = y[active], p[active]
        err = np.abs(y - p)
        r2 = r2_score(y, p) if len(y) > 1 and np.std(y) > 0 else np.nan
        rows.append(
            {
                "Variable": col,
                "N": len(y),
                "MAE_mm": mean_absolute_error(y, p),
                "RMSE_mm": np.sqrt(mean_squared_error(y, p)),
                "R2": r2,
                "pct_5mm": 100 * np.mean(err <= tol_taller[0]),
                "pct_10mm": 100 * np.mean(err <= tol_taller[1]),
            }
        )
    result = pd.DataFrame(rows)
    print(result.round(3).to_string(index=False))
    return result


def evaluar_estructura(test_df, design):
    real_na = test_df["n_alma"].astype(int).to_numpy()
    pred_na = design["n_alma"].astype(int).to_numpy()
    real_nc = test_df["n_cantonera"].astype(int).to_numpy()
    pred_nc = design["n_cantonera"].astype(int).to_numpy()
    result = {
        "n_alma_exacto": np.mean(real_na == pred_na),
        "n_cantonera_exacto": np.mean(real_nc == pred_nc),
        "ambos_exactos": np.mean((real_na == pred_na) & (real_nc == pred_nc)),
    }
    print(pd.Series(result).to_string())
    return result


def evaluar_sesgo_p(test_df, design):
    rows = []
    for col in P_COLUMNS:
        y = test_df[col].fillna(0).to_numpy(float)
        p = design[col].fillna(0).to_numpy(float)
        active = y > 0
        if active.sum() == 0:
            continue
        e = p[active] - y[active]
        rows.append(
            {
                "Variable": col,
                "N": active.sum(),
                "MAE_mm": np.abs(e).mean(),
                "sesgo_mm": e.mean(),
                "subestimadas_pct": 100 * np.mean(e < 0),
                "subestimadas_mas_10_pct": 100 * np.mean(e < -10),
            }
        )
    return pd.DataFrame(rows)


def evaluar_acumulado_y(test_df, design, incluir_e=False):
    """Error de posicion acumulada. incluir_e=True suma la distancia e (posicion
    absoluta desde el borde superior, como en el notebook)."""
    real = test_df[P_COLUMNS].fillna(0).to_numpy(float)
    pred = design[P_COLUMNS].fillna(0).to_numpy(float)
    active = real > 0
    real_y = np.cumsum(real, axis=1)
    pred_y = np.cumsum(pred, axis=1)
    if incluir_e:
        real_y = real_y + test_df["e"].to_numpy(float)[:, None]
        pred_y = pred_y + design["e"].to_numpy(float)[:, None]
    err = np.abs(pred_y - real_y)
    rows = []
    for j, col in enumerate(P_COLUMNS):
        m = active[:, j]
        if m.sum():
            rows.append(
                {
                    "Y_hasta": col,
                    "N": m.sum(),
                    "MAE_acumulado_mm": err[m, j].mean(),
                    "pct_10mm": 100 * np.mean(err[m, j] <= 10),
                    "pct_20mm": 100 * np.mean(err[m, j] <= 20),
                }
            )
    return pd.DataFrame(rows)


def evaluar_por_transicion(test_df, design):
    transiciones_reales = _matriz_transiciones(test_df)
    records = []
    for i, col in enumerate(P_COLUMNS, 1):
        y = test_df[col].fillna(0).to_numpy(float)
        p = design[col].fillna(0).to_numpy(float)
        trans = transiciones_reales[:, i - 1]
        for row in np.where(y > 0)[0]:
            e = p[row] - y[row]
            records.append(
                {"transicion": trans[row], "error_mm": e, "abs_error_mm": abs(e)}
            )
    detail = pd.DataFrame(records)
    summary = (
        detail.groupby("transicion")
        .agg(
            N=("error_mm", "size"),
            MAE_mm=("abs_error_mm", "mean"),
            sesgo_mm=("error_mm", "mean"),
            dentro_10_pct=("abs_error_mm", lambda x: 100 * np.mean(x <= 10)),
            subestimadas_mas_10_pct=("error_mm", lambda x: 100 * np.mean(x < -10)),
        )
        .reset_index()
    )
    return summary.sort_values("dentro_10_pct"), detail


# =============================================================================
# Metricas nuevas
# =============================================================================


def _errores_p(test_df, design):
    real = test_df[P_COLUMNS].fillna(0).to_numpy(float)
    pred = design[P_COLUMNS].fillna(0).to_numpy(float)
    return real, pred, real > 0


def resumen_p(test_df, design, nombre=""):
    """Metricas globales de p sobre todas las separaciones activas, incluyendo las
    COLAS (percentiles 95/99 y % por encima de 20 mm), que es donde esta el problema."""
    real, pred, act = _errores_p(test_df, design)
    err = np.abs(pred - real)[act]
    return pd.Series(
        {
            "N_separaciones": int(act.sum()),
            "MAE_mm": err.mean(),
            "RMSE_mm": np.sqrt(np.mean(err**2)),
            "mediana_abs_mm": np.median(err),
            "pct_<=5mm": 100 * np.mean(err <= 5),
            "pct_<=10mm": 100 * np.mean(err <= 10),
            "pct_>20mm": 100 * np.mean(err > 20),
            "p95_abs_mm": np.percentile(err, 95),
            "p99_abs_mm": np.percentile(err, 99),
        },
        name=nombre or "p",
    )


def contribucion_error_topologia(test_df, design):
    """Separa el error de p segun la topologia sea correcta o no y calcula que
    fraccion del error absoluto total aporta cada grupo."""
    real, pred, act = _errores_p(test_df, design)
    ok = (
        (test_df["n_alma"].astype(int).to_numpy() == design["n_alma"].astype(int).to_numpy())
        & (test_df["n_cantonera"].astype(int).to_numpy() == design["n_cantonera"].astype(int).to_numpy())
    )
    err = np.abs(pred - real)
    total = err[act].sum()
    filas = []
    for nombre, m in [("topologia correcta", ok), ("topologia incorrecta", ~ok)]:
        a = act & m[:, None]
        e = err[a]
        filas.append(
            {
                "grupo": nombre,
                "n_conexiones": int(m.sum()),
                "n_separaciones": int(a.sum()),
                "MAE_mm": e.mean() if len(e) else np.nan,
                "pct_<=10mm": 100 * np.mean(e <= 10) if len(e) else np.nan,
                "pct_<=20mm": 100 * np.mean(e <= 20) if len(e) else np.nan,
                "pct_del_error_total": 100 * e.sum() / total if total > 0 else np.nan,
            }
        )
    return pd.DataFrame(filas)


def calibracion_confianza_p(test_df, design, prob_dentro_10, bins=(0, 0.5, 0.7, 0.85, 0.95, 1.0001)):
    """Calibracion de la confianza de la cabeza discreta: para cada tramo de
    probabilidad predicha de acertar dentro de +-10 mm, el % que realmente acierta.
    Solo tiene sentido con topologia correcta; se calcula sobre separaciones
    activas reales y se reporta tambien la tasa de acierto global del tramo."""
    real, pred, act = _errores_p(test_df, design)
    hit = (np.abs(pred - real) <= 10)[act]
    conf = np.asarray(prob_dentro_10)[act]
    filas = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            filas.append(
                {
                    "confianza_predicha": f"[{lo:.2f}, {min(hi, 1.0):.2f}]",
                    "N": int(m.sum()),
                    "prob_media_predicha_%": 100 * conf[m].mean(),
                    "acierto_real_<=10mm_%": 100 * hit[m].mean(),
                }
            )
    return pd.DataFrame(filas)

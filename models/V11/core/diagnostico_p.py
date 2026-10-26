"""Diagnosticos baratos (solo pandas/numpy) para decidir QUE mejora aporta mas.

Copiar a core/diagnostico_p.py y ejecutar sobre df_train (no sobre test):

    from core.diagnostico_p import cierre_geometrico, multimodalidad, duplicados_entrada
    display(cierre_geometrico(df_train))
    display(multimodalidad(df_train))
    display(duplicados_entrada(df_train))
"""

import numpy as np
import pandas as pd

from .config import (
    P_COLUMNS,
    N_Y,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    secuencia_transiciones,
)


def _transiciones(df):
    """Matriz (n, N_Y) de nombres de transicion, de las columnas o de los conteos."""
    cols = [f"p{i}_transition" for i in range(1, N_Y + 1)]
    if all(c in df.columns for c in cols):
        return df[cols].fillna("UNUSED").astype(str).to_numpy()
    out = np.full((len(df), N_Y), "UNUSED", dtype=object)
    for r, (a, c) in enumerate(zip(df.n_alma.astype(int), df.n_cantonera.astype(int))):
        for j, t in enumerate(secuencia_transiciones(a, c)):
            out[r, j] = t
    return out


def cierre_geometrico(df, beam_col="BEAM_d_mm"):
    """Existe una restriccion de cierre? (suma de p ~ f(canto de viga))

    Prueba tres totales candidatos contra BEAM_d_mm con una recta y reporta el
    error residual. Si algun total tiene residuo << 10 mm (idealmente < 3-5 mm),
    hay una restriccion fisica que el modelo no conoce: parametrizar las p como
    fracciones de ese total (o forzar que la ultima p sea el resto) elimina la
    acumulacion de error en las ultimas p.
    """
    p = df[P_COLUMNS].fillna(0).to_numpy(float)
    e = df["e"].to_numpy(float)
    d = df["d"].to_numpy(float)
    H = df[beam_col].to_numpy(float)
    totales = {
        "sum(p)": p.sum(1),
        "e + sum(p)": e + p.sum(1),
        "e + sum(p) + d": e + p.sum(1) + d,
    }
    filas = []
    for nombre, T in totales.items():
        for etiqueta, mask in [("todas", np.ones(len(df), bool))] + [
            (f"na={a},nc={c}", (df.n_alma.astype(int) == a).to_numpy() & (df.n_cantonera.astype(int) == c).to_numpy())
            for a, c in sorted(set(zip(df.n_alma.astype(int), df.n_cantonera.astype(int))))
        ]:
            if mask.sum() < 20:
                continue
            A = np.c_[H[mask], np.ones(mask.sum())]
            coef, *_ = np.linalg.lstsq(A, T[mask], rcond=None)
            res = T[mask] - A @ coef
            filas.append(
                {
                    "total": nombre,
                    "grupo": etiqueta,
                    "N": int(mask.sum()),
                    "pendiente": coef[0],
                    "residuo_std_mm": res.std(),
                    "residuo_MAE_mm": np.abs(res).mean(),
                    "pct_residuo_<=5mm": 100 * np.mean(np.abs(res) <= 5),
                }
            )
    return pd.DataFrame(filas).round(2)


def multimodalidad(df):
    """Por transicion: cuantos valores distintos toman las p y cuanta masa
    concentran los 1 y 3 mas frecuentes. Si top3 es alto (>70-80%) las p son
    casi categoricas y la regresion a la media es mala idea."""
    T = _transiciones(df)
    P = df[P_COLUMNS].fillna(0).to_numpy(float)
    filas = []
    for t in sorted(set(T[P > 0])):
        v = P[(T == t) & (P > 0)]
        vc = pd.Series(v).value_counts(normalize=True)
        filas.append(
            {
                "transicion": t,
                "N": len(v),
                "valores_distintos": len(vc),
                "media": v.mean(),
                "std": v.std(),
                "pct_en_rejilla_5": 100 * np.mean(v % 5 == 0),
                "top1_pct": 100 * vc.iloc[:1].sum(),
                "top3_pct": 100 * vc.iloc[:3].sum(),
            }
        )
    return pd.DataFrame(filas).sort_values("N", ascending=False).round(2)


def duplicados_entrada(df, keys=None):
    """Mide el suelo de ruido: filas con ENTRADAS identicas pero p distintas.

    Si hay muchos grupos con entradas iguales y p distintas, parte del error es
    irreducible (el generador no es determinista) y ningun modelo lo bajara:
    en ese caso hay que mirar mas entradas, no mas arquitectura.
    """
    keys = keys or (CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    g = df.groupby(keys, dropna=False)
    sizes = g.size()
    dup = sizes[sizes > 1]
    if dup.empty:
        return pd.DataFrame({"resultado": ["No hay filas con entradas identicas: no se puede medir el suelo de ruido."]})
    filas = []
    for col in P_COLUMNS:
        sub = df[col].fillna(0)
        dev = sub - g[col].transform(lambda s: s.fillna(0).median())
        act = (sub > 0) & (g[col].transform("size") > 1)
        if act.sum():
            filas.append(
                {
                    "p": col,
                    "N_en_duplicados": int(act.sum()),
                    "MAE_vs_mediana_grupo_mm": dev[act].abs().mean(),
                    "pct_dentro_10mm": 100 * np.mean(dev[act].abs() <= 10),
                }
            )
    return pd.DataFrame(filas).round(2)

"""Escenarios de carga: de una fila de entrada a 3 diseños de referencia.

Flujo:
  1. Se predice la fila tal cual llega (paso 0): modelo -> CODAME -> ajuste -> comprobacion.
  2. Se suben las cargas (ratio_moment, ratio_shear, ratio_axial) un `incremento` (5 %) y se
     vuelve a predecir, paso a paso (k = 1, 2, 3 ...), hasta que el diseno deja de cumplir.
     Ninguna carga supera el 100 % (1.0): se topan y, cuando todas llegan al tope, se para.
  3. Del historial se extraen 3 disenos. E2 y E3 priorizan SIEMPRE la combinacion que CUMPLE
     la comprobacion AISC (misma regla que la web, ver `_cumple`):
       E1  diseno de entrada (paso 0, sin tocar cargas)
       E2  mayor ratio_moment SIN cantonera que cumple. Si ninguno cumple, el de mayor carga
           (con aviso). Si todas las predicciones llevan cantonera, el escenario queda vacio.
       E3  mayor ratio_moment CON cantonera que cumple. Si ninguno cumple, el de mayor carga
           (con aviso). Si ninguna prediccion lleva cantonera, el escenario queda vacio.
  El barrido no se corta por fallos salvo que `max_fallos_seguidos` > 0: por defecto se
  exploran todos los pasos hasta el 100 %, para encontrar siempre el maximo que cumple.

Sobre el ajuste: en este flujo NO se maximiza la carga (`maximize_load=False`), porque
`refinar._maximize_load` reescribe MzEd_Porcen/FyEd_Porcen/FxEd_Porcen y falsearia el
escalon de carga que se esta evaluando. Si se ajustan geometria y espesores.

Este modulo no depende del modelo ni del servidor: recibe `predict_fn` (inyeccion), por lo
que se puede probar sin pesos. `run()` lo conecta con `predictor.predict_detailed`.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("ai-server")

LOAD_KEYS = ("ratio_moment", "ratio_shear", "ratio_axial")

# Máximos de carga del dataset de entrenamiento (datos_modelo.parquet)
RANGO_ENTRENAMIENTO = {"ratio_moment": 1.21, "ratio_shear": 1.0, "ratio_axial": 0.49}

# Cargas base fijas para el inicio del barrido (pasos k >= 1)
BASE_SWEEP_MOMENT = 0.20
BASE_SWEEP_SHEAR = 0.50
BASE_SWEEP_AXIAL = 0.05

CARGA_MAX = 1.0            # Carga máxima (100 %)
INCREMENTO = 0.05          # +5 % por paso en ratio_moment
MAX_PASOS = 40             # Pasos máximos de exploración
MAX_FALLOS_SEGUIDOS = 0    # 0 = no cortar el barrido por fallos (explorar hasta el 100 %)
LOTE = MAX_PASOS + 1       # Todos los pasos en un solo lote: el barrido real (~18 pasos) se reparte
                           # entre los workers de una vez y el batcher recibe una sola peticion

NOMBRES = {
    "E1": "Diseño de entrada",
    "E2": "Máximo sin cantonera",
    "E3": "Máximo con cantonera",
}


def _num(v, default=0.0) -> float:
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _find_key(row: Dict[str, Any], name: str) -> Optional[str]:
    for k in row:
        if str(k).strip().lower() == name.lower():
            return k
    return None

def _cumple(chk: Optional[Dict[str, Any]]) -> bool:
    """Misma regla que `evaluateCumple` de la web: `cumple` del checker Y ratio pesimo <= 1
    Y todos los ratios <= 1 Y ninguna comprobacion geometrica fallida."""
    if not chk:
        return False
    ok = bool(chk["cumple"]) if "cumple" in chk else True
    rp = chk.get("ratio_pesimo")
    if rp is not None and _num(rp, 0.0) > 1.0:
        ok = False
    for r in chk.get("ratios") or []:
        if isinstance(r, dict) and r.get("ratio") is not None and _num(r["ratio"], 0.0) > 1.0:
            ok = False
            break
    for c in chk.get("complementarios") or []:
        if not isinstance(c, dict):
            continue
        txt = str(c.get("resultado") or "").lower()
        if "no cumple" in txt or "fail" in txt or "error" in txt or c.get("cumple") is False:
            ok = False
            break
    return ok


def _cargas_base(row: Dict[str, Any]) -> Dict[str, float]:
    """{clave_real_en_la_fila: valor} de las 3 cargas. ratio_moment es obligatorio."""
    out: Dict[str, float] = {}
    for name in LOAD_KEYS:
        key = _find_key(row, name)
        if key is None or row[key] is None or str(row[key]).strip() == "":
            if name == "ratio_moment":
                raise ValueError("Falta ratio_moment: es necesario para generar los escenarios de carga")
            continue
        out[key] = _num(str(row[key]).replace(",", "."))
    if not any(v > 0 for v in out.values()):
        raise ValueError("Todas las cargas son 0: no hay nada que aumentar")
    return out


def _fila_paso(row: Dict[str, Any], k: int, incremento: float = INCREMENTO) -> tuple[Dict[str, Any], float]:
    """Genera la fila de entrada del paso k.
    k = 0: Mantiene la fila de entrada tal cual llega.
    k >= 1: ratio_shear=0.50, ratio_axial=0.05, ratio_moment = min(1.0, 0.20 + (k-1)*incremento).
    """
    new = dict(row)
    ukey = _find_key(row, "unicode") or "UNICODE"
    mkey = _find_key(row, "ratio_moment") or "ratio_moment"
    vkey = _find_key(row, "ratio_shear") or "ratio_shear"
    akey = _find_key(row, "ratio_axial") or "ratio_axial"

    if k == 0:
        new[ukey] = f"{row.get(ukey, 'UNI')}_E00"
        return new, 1.0

    m_val = round(min(CARGA_MAX, BASE_SWEEP_MOMENT + (k - 1) * incremento), 2)
    new[mkey] = m_val
    new[vkey] = round(min(CARGA_MAX, BASE_SWEEP_SHEAR), 2)
    new[akey] = round(min(CARGA_MAX, BASE_SWEEP_AXIAL), 2)
    new[ukey] = f"{row.get(ukey, 'UNI')}_E{k:02d}"

    return new, m_val


def _filas_tornillos(row: Dict[str, Any]) -> int:
    seps = [row.get("pfi"), row.get("pb")] + [row.get(f"p{i}") for i in range(3, 13)]
    return sum(1 for s in seps if _num(s) > 0) + 1


def _registro(k: int, factor: float, entrada: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
    row = res.get("row") or {}
    chk = res.get("check") or {}
    cumple = _cumple(chk)
    if chk:
        chk = {**chk, "cumple": cumple}
        res["check"] = chk
    lh = _num(row.get("Lh"))
    ncan = int(_num(res.get("n_cantonera")))
    cargas = {n: _num(entrada.get(_find_key(entrada, n))) for n in LOAD_KEYS}
    fuera = [n for n, lim in RANGO_ENTRENAMIENTO.items() if cargas.get(n, 0.0) > lim + 1e-9]
    return {
        "paso": k,
        "factor_carga": round(factor, 4),
        "ratio_moment": cargas["ratio_moment"],
        "ratio_shear": cargas["ratio_shear"],
        "ratio_axial": cargas["ratio_axial"],
        "cumple": cumple,
        "ratio_pesimo": chk.get("ratio_pesimo"),
        "descripcion_ratio_pesimo": chk.get("descripcion_ratio_pesimo"),
        "Lh": lh,
        "n_cantonera": ncan,
        "con_cantonera": lh > 0 or ncan > 0,
        "fuera_rango_entrenamiento": fuera,
        "_res": res,
    }


def seleccionar(historial: List[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Extrae E1 (entrada), E2 (máximo sin cantonera) y E3 (máximo con cantonera)."""
    if not historial:
        return {"E1": None, "E2": None, "E3": None}

    # E1: Fila de entrada exacta (paso 0)
    paso0 = next((r for r in historial if r["paso"] == 0), historial[0])

    # Pasos del barrido de cargas (k >= 1)
    barrido = [r for r in historial if r["paso"] >= 1] or historial

    sin_ok = [r for r in barrido if not r["con_cantonera"] and r["cumple"]]
    sin_all = [r for r in barrido if not r["con_cantonera"]]

    con_ok = [r for r in barrido if r["con_cantonera"] and r["cumple"]]
    con_all = [r for r in barrido if r["con_cantonera"]]

    # E2: Mayor ratio_moment sin cantonera que cumpla; si ninguna cumple, la mayor sin cantonera.
    # Si todas las predicciones del barrido llevan cantonera, E2 queda vacio (None).
    if sin_ok:
        e2 = max(sin_ok, key=lambda r: (r["ratio_moment"], r["paso"]))
    elif sin_all:
        e2 = max(sin_all, key=lambda r: (r["ratio_moment"], r["paso"]))
    else:
        e2 = None

    # E3: Mayor ratio_moment con cantonera que cumpla; si ninguna cumple, la mayor con cantonera.
    # Si ninguna predicción del barrido lleva cantonera, E3 queda vacío (None): NO se rellena
    # con un diseño sin cantonera.
    if con_ok:
        e3 = max(con_ok, key=lambda r: (r["ratio_moment"], r["paso"]))
    elif con_all:
        e3 = max(con_all, key=lambda r: (r["ratio_moment"], r["paso"]))
    else:
        e3 = None

    return {"E1": paso0, "E2": e2, "E3": e3}


def _hay_cantonera(historial: List[Dict[str, Any]]) -> bool:
    return any(r["con_cantonera"] for r in historial if r["paso"] >= 1)


def _hay_sin_cantonera(historial: List[Dict[str, Any]]) -> bool:
    return any(not r["con_cantonera"] for r in historial if r["paso"] >= 1)


def _motivo(clave: str, historial: List[Dict[str, Any]]) -> str:
    if clave == "E3" and not _hay_cantonera(historial):
        return "Ninguna predicción del barrido de cargas genera cantonera: no hay opción de colocar cantonera."
    if clave == "E2" and not _hay_sin_cantonera(historial):
        return "Todas las predicciones del barrido de cargas llevan cantonera: no hay diseño sin cantonera."
    ok = [r for r in historial if r["cumple"]]
    if not ok:
        return "Ningún paso cumple la comprobación AISC."
    if clave == "E1":
        return "No se ha procesado el diseño de entrada."
    if clave == "E2":
        return "Ningún diseño sin cantonera cumple la comprobación AISC en el barrido de cargas."
    return "Ningún diseño con cantonera cumple la comprobación AISC en el barrido de cargas."


def generar_escenarios(
    row: Dict[str, Any],
    predict_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    incremento: float = INCREMENTO,
    max_pasos: int = MAX_PASOS,
    max_fallos_seguidos: int = MAX_FALLOS_SEGUIDOS,
    lote: int = LOTE,
    compuesto: bool = True,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """`predict_fn(filas_entrada) -> [{"row", "check", "n_cantonera"}]` (mismo orden).

    Si se supera `timeout_s`, lanza `TimeoutError` (no se devuelven resultados parciales).
    """
    if incremento <= 0:
        raise ValueError("incremento debe ser > 0")
    t0 = time.monotonic()
    deadline = t0 + timeout_s if timeout_s is not None and timeout_s > 0 else None
    base = _cargas_base(row)
    avisos: List[str] = []
    historial: List[Dict[str, Any]] = []
    fallos, hay_cantonera_valida, parada = 0, False, None
    previas, techo, k = None, False, 0

    def _check_timeout() -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"escenarios: tiempo de espera agotado ({timeout_s:g}s)")

    while parada is None and not techo and k <= max_pasos:
        _check_timeout()
        pasos, entradas, factores = [], [], []
        for kk in range(k, min(k + max(1, lote), max_pasos + 1)):
            ent, f = _fila_paso(row, kk, incremento)
            cargas = (ent.get("ratio_moment"), ent.get("ratio_shear"), ent.get("ratio_axial"))
            if kk >= 1 and cargas == previas:
                techo = True
                break
            if kk >= 1:
                previas = cargas
            pasos.append(kk)
            entradas.append(ent)
            factores.append(f)

        if entradas:
            resultados = predict_fn(entradas)
            for kk, ent, f, res in zip(pasos, entradas, factores, resultados):
                reg = _registro(kk, f, ent, res)
                historial.append(reg)
                if kk >= 1:
                    if reg["cumple"]:
                        fallos = 0
                        hay_cantonera_valida = hay_cantonera_valida or reg["con_cantonera"]
                    else:
                        fallos += 1
                    tolerancia = max_fallos_seguidos * (1 if hay_cantonera_valida else 2)
                    if max_fallos_seguidos > 0 and fallos >= tolerancia:
                        parada = f"{fallos} pasos seguidos sin cumplir (paso {kk})"
                        break
        k += max(1, lote)
        if parada is None and not techo:
            _check_timeout()
    if parada is None:
        parada = "ratio_moment al 100 %" if techo else f"alcanzado max_pasos={max_pasos}"

    elegidos = seleccionar(historial)
    if historial and not historial[0]["cumple"]:
        avisos.append("El diseño de entrada (paso 0) NO cumple la comprobación AISC.")

    escenarios, rows, checks, rows_raw = [], [], [], []
    for clave in ("E1", "E2", "E3"):
        r = elegidos[clave]
        if r is None:
            sin_cant = clave == "E3" and not _hay_cantonera(historial)
            escenarios.append({"id": clave, "escenario": NOMBRES[clave], "encontrado": False,
                               "sin_opcion_cantonera": sin_cant,
                               "motivo": _motivo(clave, historial)})
            if sin_cant:
                avisos.append("E3: ninguna predicción genera cantonera; no hay opción de colocar cantonera.")
            continue
        res = r["_res"]
        etiqueta = f"{clave} - {NOMBRES[clave]}"
        if clave in ("E2", "E3") and not r["cumple"]:
            avisos.append(f"{clave}: ningún diseño del barrido cumple la comprobación AISC; "
                          f"se muestra el de mayor carga, que NO cumple.")
        if r["fuera_rango_entrenamiento"]:
            avisos.append(f"{clave}: cargas fuera del rango de entrenamiento "
                          f"({', '.join(r['fuera_rango_entrenamiento'])}); el modelo extrapola.")
        escenarios.append({
            "id": clave, "escenario": NOMBRES[clave], "encontrado": True,
            **{x: r[x] for x in ("paso", "factor_carga", "ratio_moment", "ratio_shear", "ratio_axial",
                                 "cumple", "ratio_pesimo", "descripcion_ratio_pesimo", "Lh",
                                 "n_cantonera", "fuera_rango_entrenamiento")},
            "row": res.get("row"),
        })
        rows.append({**(res.get("row") or {}), "Escenario": etiqueta})
        checks.append({**(res.get("check") or {}), "Escenario": etiqueta})
        rows_raw.append({**(res.get("raw") or {}), "Escenario": etiqueta})

    tabla = [_fila_tabla(e) for e in escenarios]
    return {
        "rows": rows,
        "checks": checks,
        "rows_raw": rows_raw,
        "escenarios": [{k_: v for k_, v in e.items() if k_ != "row"} for e in escenarios],
        "tabla": tabla,
        "tabla_markdown": tabla_markdown(tabla),
        "historial": [{k_: v for k_, v in r.items() if k_ != "_res"} for r in historial],
        "parada": parada,
        "tiempo_s": round(time.monotonic() - t0, 2),
        "avisos": avisos,
        "parametros": {"incremento": incremento, "compuesto": compuesto, "max_pasos": max_pasos,
                       "max_fallos_seguidos": max_fallos_seguidos, "carga_max": CARGA_MAX,
                       "timeout_s": timeout_s},
    }


# ---------------------------------------------------------------------------
# Tabla
# ---------------------------------------------------------------------------
TABLA_COLUMNAS = [
    "Escenario", "Paso", "Factor carga", "Mz %", "Fy %", "Fx %", "Cumple", "Ratio pésimo",
    "Cantonera (Lh mm)", "n_cantonera", "Viga", "Pilar", "Tornillo", "Clase", "tp (placa)",
    "g", "b_ep", "Lp", "Filas tornillos", "Rig. col.", "Placa alma", "Rig. cartela", "Fuera de rango",
]


def _fila_tabla(e: Dict[str, Any]) -> Dict[str, Any]:
    out = {c: None for c in TABLA_COLUMNAS}
    out["Escenario"] = f"{e['id']} - {e['escenario']}"
    if not e["encontrado"]:
        out["Cumple"] = "Sin opción de cantonera" if e.get("sin_opcion_cantonera") else "No encontrado"
        out["Cantonera (Lh mm)"] = 0 if e.get("sin_opcion_cantonera") else None
        out["Ratio pésimo"] = e.get("motivo")
        return out
    r = e["row"] or {}
    pct = lambda v: int(round(_num(v) * 100))
    out.update({
        "Paso": e["paso"], "Factor carga": e["factor_carga"],
        "Mz %": pct(e["ratio_moment"]), "Fy %": pct(e["ratio_shear"]), "Fx %": pct(e["ratio_axial"]),
        "Cumple": "Sí" if e["cumple"] else "No",
        "Ratio pésimo": f"{_num(e['ratio_pesimo']):.2f} ({str(e['descripcion_ratio_pesimo']).strip()})",
        "Cantonera (Lh mm)": round(e["Lh"], 1), "n_cantonera": e["n_cantonera"],
        "Viga": r.get("BEAM"), "Pilar": r.get("COLUMN"), "Tornillo": r.get("BoltName_ep"),
        "Clase": r.get("BoltGrade"), "tp (placa)": r.get("ThicknessName_t_ep"),
        "g": r.get("p2_ep"), "b_ep": r.get("b_ep"), "Lp": r.get("Lp"),
        "Filas tornillos": _filas_tornillos(r),
        "Rig. col.": r.get("ThicknessName_t_tenstiff"), "Placa alma": r.get("ThicknessName_t_webstiff"),
        "Rig. cartela": r.get("ThicknessName_t_BeamHcompstiff"),
        "Fuera de rango": ", ".join(e["fuera_rango_entrenamiento"]) or "No",
    })
    return out


def tabla_markdown(tabla: List[Dict[str, Any]]) -> str:
    def cel(v):
        return "" if v is None else str(v).replace("|", "/")
    lineas = ["| " + " | ".join(TABLA_COLUMNAS) + " |", "|" + "---|" * len(TABLA_COLUMNAS)]
    lineas += ["| " + " | ".join(cel(f[c]) for c in TABLA_COLUMNAS) + " |" for f in tabla]
    return "\n".join(lineas)


def run(model_name: str, row: Dict[str, Any], **opciones) -> Dict[str, Any]:
    import predictor

    def predict_fn(filas):
        return predictor.predict_detailed(model_name, filas, maximize_load=False)

    return generar_escenarios(row, predict_fn, **opciones)
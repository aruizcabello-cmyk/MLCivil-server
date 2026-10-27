"""Ajuste automatico de geometría y placas tras la prediccion del modelo.

El modelo propone geometría y espesores que a veces no pasan la comprobacion AISC 360-10.
Aqui se usa el propio comprobador (`checker`) como oraculo:

1. REAJUSTE GEOMÉTRICO ITERATIVO (`_ajustar_geometria`):
   Corrige de forma iterativa las cotas que generan avisos geométricos (REVISAR):
   gramil p2_ep, ancho b_ep, borde dev, separaciones, colisiones con alas, cantonera
   sobre tornillos (Lh NO se toca: se mueven las filas) y, como último recurso, el diámetro del
   tornillo. Se prioriza mantener los valores predichos: solo se cambia lo imprescindible.
   Reglas fijas (ver `codame.ajustar_cantonera_y_chapa`): último tornillo por debajo de
   viga/cantonera, chapa = última fila + e, b_ep entre el ala de la viga y la del pilar.

2. REAJUSTE DE ESPESORES DE PLACAS:
   Mientras algun ratio (sin contar soldaduras ni tornillos) supere 1, se sube UN escalon
   del catalogo el espesor de la placa responsable (tp, tst, tws, tbs) y se vuelve a comprobar.

3. OPTIMIZACIÓN Y MAXIMIZACIÓN DE CARGA:
   Quita elementos sobrantes con demanda residual <= 0 y busca el mayor porcentaje de carga.

Desactivar con la variable de entorno AI_AUTO_REFINE=0.
"""

import logging
import math
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import codame
import config
from calculador import CalculadorAISC36010
from catalogo import get_profile
from checker import calcular_ratios_row, check_single_row, row_to_conexion_input
from externos import tablapernos

log = logging.getLogger("ai-server")

ENABLED = os.environ.get("AI_AUTO_REFINE", "1").strip().lower() not in ("0", "false", "no")
TARGET_RATIO = 1.0
MAX_STEPS = 50

# Catalogos de espesores (mm). Mantener alineados con los del modelo (core/config.py).
CAT_TP = [12, 15, 20, 25, 30, 35, 40, 45, 50, 60]
CAT_TST = [0, 10, 12, 15, 20, 25, 30, 35, 40]          # ts = tst
CAT_TWS = [0, 6, 8, 10, 12, 15, 20]
CAT_TBS = [0, 10, 12, 15, 20, 25, 30, 35, 40]

# prefijo del ratio -> variable a subir
_REMEDY = (
    (("End plate", "Shear yielding", "Shear rupture"), "tp"),
    (("Column flange", "Column web crippling", "Transverse stiffeners"), "tst"),
    (("Column panel zone", "Web doubler"), "tws"),
    (("Beam web yielding", "Haunch web"), "tbs"),
)


def _mm(v) -> int:
    try:
        return int(round(float(str(v).lower().replace("mm", "").strip() or 0)))
    except ValueError:
        return 0


def _next(catalog: List[int], value: int) -> Optional[int]:
    return next((c for c in catalog if c > value), None)


def _eval(row: Dict[str, Any]) -> Tuple[Dict[str, float], Any]:
    """(ratios {descripcion: valor}, calculador). Via rapida: sin el resultado completo."""
    return calcular_ratios_row(dict(row))


def _ratios(row: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(v) for k, v in _eval(row)[0].items() if v is not None and v == v}


def _variable_for(ratio_name: str) -> Optional[str]:
    if "Weld" in ratio_name:
        return None
    for prefixes, var in _REMEDY:
        if ratio_name.startswith(prefixes):
            return var
    return None


def _apply(row: Dict[str, Any], var: str) -> Optional[str]:
    """Sube un escalon `var`. Devuelve la descripcion del cambio o None si no hay mas catalogo."""
    if var == "tp":
        old = _mm(row["ThicknessName_t_ep"]); new = _next(CAT_TP, old)
        if new is None:
            return None
        row["ThicknessName_t_ep"] = f"{new}mm"
    elif var == "tst":
        old = _mm(row["ThicknessName_t_tenstiff"]); new = _next(CAT_TST, old)
        if new is None:
            return None
        row["ThicknessName_t_tenstiff"] = row["ThicknessName_t_compstiff"] = f"{new}mm"
    elif var == "tws":
        old = _mm(row["ThicknessName_t_webstiff"]); new = _next(CAT_TWS, old)
        if new is None:
            return None
        row["ThicknessName_t_webstiff"] = f"{new}mm"
    elif var == "tbs":
        old = _mm(row["ThicknessName_t_BeamHcompstiff"]); new = _next(CAT_TBS, old)
        if new is None:
            return None
        row["ThicknessName_t_BeamHcompstiff"] = f"{new}mm"
    else:
        return None

    if var in ("tws", "tst"):   # Bws/Lws dependen de tws (y de los rigidizadores)
        _refresh_web_plate(row)
    return f"{var} {old}->{new}"


def _refresh_web_plate(row: Dict[str, Any]) -> None:
    if _mm(row["ThicknessName_t_webstiff"]) <= 0:
        row["Bws"], row["Lws"] = 0.0, 0.0
        return
    beam, col = get_profile(row["BEAM"]), get_profile(row["COLUMN"])
    row["Bws"], row["Lws"] = codame._web_plate_dims(
        col, codame._geom(beam, "d_mm"), codame._geom(beam, "tf_mm"), _mm(row["ThicknessName_t_tenstiff"])
    )


# ---------------------------------------------------------------------------
# Reescalado y ajuste geométrico iterativo
# ---------------------------------------------------------------------------
SEP_KEYS = ("pb", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12")
CLAVES_GEO = ("p2_ep", "b_ep", "dev", "pfo", "pfi") + SEP_KEYS + ("FactorCantoCantonera", "BoltName_ep")
_FALLOS_PLACA = {"PRMIN", "PRMAX", "DEH_MIN", "DEH_MAX", "bep&bb", "bep&bepmin", "bep&bc"}
BOLT_MIN = "M20"   # no se baja de este tornillo (igual que geometria.py)


def _n(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _c5(x: float) -> float:
    return math.ceil(round(x, 6) / 5.0) * 5.0


def _f5(x: float) -> float:
    return math.floor(round(x, 6) / 5.0) * 5.0


def _r5(x: float) -> float:
    return round(x / 5.0) * 5.0


def _cambiado(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) > 1e-9
    except (TypeError, ValueError):
        return str(a) != str(b)


def _fmt(v) -> str:
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _evaluar_geometria(work: Dict[str, Any]):
    """(calculador ya ejecutado, {descripcion: texto REVISAR}). El calculador trae los limites
    (MINedge, MAXedge, MINspacing, MAXspacing, c_, De_...) que NO estan en ConexionInput."""
    calc = CalculadorAISC36010(inp=row_to_conexion_input(work))
    return calc, calc.calcular_geometria()


# --- placa: gramil p2_ep y ancho b_ep -------------------------------------
def _ajustar_placa(work: Dict[str, Any], calc) -> bool:
    """Mete p2_ep en [Prmin, Prmax] y b_ep en el rango que cumple a la vez:
         b_ep >= ancho ala viga (y viga+20 si cabe), b_ep <= ala pilar,
         MINedge <= (b_ep - p2_ep)/2 <= MAXedge.
    Ademas b_ep debe quedar ESTRICTAMENTE entre el ala de la viga y la del pilar: si iguala
    o supera alguna, se lleva al punto medio. Cambia lo minimo posible.
    Devuelve False si no existe solucion con este tornillo (hay que bajar de diametro)."""
    pilar, viga = calc.pilar, calc.viga
    mn, mx = calc.MINedge, calc.MAXedge
    g0 = _n(work.get("p2_ep"), 100.0)
    b0 = _n(work.get("b_ep"), 200.0)
    g, b = g0, b0
    bep_min = _c5(viga.b + 20.0)

    if pilar.Prmin > 0:
        g = max(g, _c5(pilar.Prmin))
    if pilar.Prmax > 0:
        g = min(g, _f5(pilar.Prmax))

    def rango(gv):
        hi = min(pilar.b, gv + 2.0 * mx)
        lo = max(viga.b, gv + 2.0 * mn)
        if bep_min <= hi:
            lo = max(lo, bep_min)
        return lo, hi

    lo, hi = rango(g)
    ok = True
    if lo > hi:
        # mover el gramil: bajarlo si la placa no cabe en el ala del pilar, subirlo si no llega a la viga
        g_new = min(max(g, _c5(viga.b - 2.0 * mx)), _f5(pilar.b - 2.0 * mn))
        dentro = ((pilar.Prmin <= 0 or g_new >= pilar.Prmin) and
                  (pilar.Prmax <= 0 or g_new <= pilar.Prmax))
        if dentro:
            g = g_new
            lo, hi = rango(g)
        if not dentro or lo > hi:
            ok = False

    if ok:
        lo_r = _c5(lo) if _c5(lo) <= hi else math.ceil(lo)
        hi_r = _f5(hi) if _f5(hi) >= lo_r else math.floor(hi)
        b = min(max(b, lo_r), hi_r)
        if b >= pilar.b or b <= viga.b:        # preferencia: estrictamente entre ambas alas
            b = min(max(_r5((viga.b + pilar.b) / 2.0), lo_r), hi_r)

    if _cambiado(g, g0):
        work["p2_ep"] = g
    if _cambiado(b, b0):
        work["b_ep"] = b
    return ok


# --- distancia al borde superior dev ---------------------------------------
def _ajustar_dev(work: Dict[str, Any], calc) -> None:
    dev = _n(work.get("dev"), 50.0)
    new = dev
    if dev < calc.MINedge:
        new = _c5(calc.MINedge)
    elif dev > calc.MAXedge:
        new = _f5(calc.MAXedge)
    if _cambiado(new, dev):
        work["dev"] = new


# --- separaciones entre filas ----------------------------------------------
def _ajustar_separaciones(work: Dict[str, Any], calc) -> None:
    """La 1a separacion entre filas es c = redondeo5(pfo + pfi + tf): se corrige con pfi.
    El resto (pb, p3..p12) se corrigen directamente."""
    mn, mx = calc.MINspacing, calc.MAXspacing
    for k in SEP_KEYS:
        v = _n(work.get(k))
        if 0.0 < v < mn:
            work[k] = _c5(mn)
        elif v > mx:
            work[k] = _f5(mx)

    c = calc.c_
    pfo, pfi, tf = _n(work.get("pfo"), 50.0), _n(work.get("pfi"), 30.0), calc.viga.tf
    if c < mn:
        nuevo = _c5(mn) - pfo - tf
        if nuevo > pfi:
            work["pfi"] = nuevo
    elif c > mx:
        nuevo = _f5(mx) - pfo - tf
        if nuevo > 0:
            work["pfi"] = nuevo


# --- colisiones de filas de tornillos con las caras de las alas ------------
def _margen(calc) -> float:
    digitos = "".join(ch for ch in str(calc.BoltName_ep) if ch.isdigit())
    return tablapernos.DATOS_PERNOS.get(int(digitos), {}).get("clearance", 0.0) if digitos else 0.0


def _obstaculos(work: Dict[str, Any], calc) -> List[float]:
    """Caras de las alas de la VIGA. Las de la cantonera las trata `_ajustar_cantonera`
    (aumentando su canto), no moviendo filas."""
    dev, pfo = _n(work.get("dev"), 50.0), _n(work.get("pfo"), 50.0)
    h, tf = calc.viga.h, calc.viga.tf
    return [dev + pfo, dev + pfo + tf, dev + pfo + h - tf, dev + pfo + h]


def _filas(work: Dict[str, Any], calc) -> List[Tuple[float, str]]:
    dev = _n(work.get("dev"), 50.0)
    y = dev + calc.c_
    rows = [(dev, "dev"), (y, "pfi")]
    for k in SEP_KEYS:
        v = _n(work.get(k))
        if v > 0:
            y += v
            rows.append((y, k))
    return rows


def _despejar_alas(work: Dict[str, Any], calc) -> bool:
    """Corrige la PRIMERA fila que choca con una cara de ala de la viga. Un solo cambio por
    llamada (mover una fila desplaza las siguientes); el bucle principal vuelve a comprobar."""
    m = _margen(calc)
    if m <= 0:
        return False
    obs = _obstaculos(work, calc)
    for y, key in _filas(work, calc):
        if all(abs(y - o) >= m for o in obs):
            continue
        if key == "dev":                       # fila 0 vs cara superior del ala: distancia = pfo
            pfo = _n(work.get("pfo"), 50.0)
            work["pfo"] = _c5(m) if pfo < m else pfo + 5.0
            return True
        if key == "pfi":                       # fila 1 vs cara inferior: distancia = c - pfo - tf
            falta = m - (calc.c_ - _n(work.get("pfo"), 50.0) - calc.viga.tf)
            work["pfi"] = _n(work.get("pfi"), 30.0) + max(5.0, _c5(falta))
            return True
        v = _n(work.get(key))                  # fila intermedia: desplazar su separacion previa
        for k in range(1, 13):
            for d in (5.0 * k, -5.0 * k):
                nv = v + d
                if nv < calc.MINspacing or nv > calc.MAXspacing:
                    continue
                if all(abs((y + d) - o) >= m for o in obs):
                    work[key] = nv
                    return True
        return False
    return False


# --- cantonera sobre tornillos / ultimo tornillo / largo de chapa -----------
def _ajustar_cantonera(work: Dict[str, Any], calc) -> bool:
    """Reglas de `codame.ajustar_cantonera_y_chapa` con los limites reales del calculador:
    el ala de la cantonera no puede caer sobre una fila (Lh es fijo: se mueven las filas
    por encima/debajo del ala), el ultimo tornillo va por debajo de viga/cantonera y
    Lp = ultima fila + e."""
    lim = {"clearance": _margen(calc), "minsp": calc.MINspacing,
           "maxsp": calc.MAXspacing, "c": calc.c_}
    return codame.ajustar_cantonera_y_chapa(work, lim)


# --- ultimo recurso: tornillo de menor diametro ----------------------------
def _bajar_tornillo(work: Dict[str, Any]) -> bool:
    nombre = str(work.get("BoltName_ep") or "").strip().upper()
    if not nombre or nombre == BOLT_MIN:
        return False
    nuevo = str(tablapernos().NombreTornilloAnterior(nombre) or "").strip().upper()
    if not nuevo or nuevo == nombre:
        return False
    work["BoltName_ep"] = nuevo
    return True


def _actualizar_lp(work: Dict[str, Any], calc) -> None:
    """Lp (informativo en CODAME) = ultima fila de tornillos + e (dev), con las cotas finales."""
    if "Lp" not in work:
        return
    dev = _n(work.get("dev"))
    c = codame._redond5(_n(work.get("pfo")) + _n(work.get("pfi")) + calc.viga.tf)
    db = codame.diametro_tornillo(work.get("BoltName_ep"))
    work["Lp"] = dev + c + sum(_n(work.get(k)) for k in SEP_KEYS) + db + dev


def _ajustar_geometria(work: Dict[str, Any], max_iter: int = 40) -> List[str]:
    """Reajusta la geometria hasta que no quede ningun REVISAR ni regla incumplida.
    Cada iteracion hace UNA tanda de cambios y vuelve a comprobar con el calculador.
    Orden: placa/dev/separaciones -> colisiones con alas -> cantonera y ultimo tornillo
    -> bajar tornillo."""
    changes: List[str] = []
    vistos = set()
    calc, fallos = None, {}

    for _ in range(max_iter):
        calc, fallos = _evaluar_geometria(work)
        estado = tuple(str(work.get(k)) for k in CLAVES_GEO)
        if estado in vistos:                   # oscilacion A->B->A
            break
        vistos.add(estado)
        antes = dict(work)

        def sin_cambio() -> bool:
            return not any(_cambiado(work.get(k), antes.get(k)) for k in CLAVES_GEO)

        falla_placa = bool(_FALLOS_PLACA & set(fallos))
        ok_placa = _ajustar_placa(work, calc)
        _ajustar_dev(work, calc)
        if "SPACING_MIN" in fallos or "SPACING_MAX" in fallos:
            _ajustar_separaciones(work, calc)

        if sin_cambio() and any(k.startswith("ENTER_") and "HAUNCH" not in k for k in fallos):
            _despejar_alas(work, calc)
        if sin_cambio():                       # cantonera sobre tornillo / ultimo tornillo / Lp
            _ajustar_cantonera(work, calc)
        if sin_cambio() and falla_placa and not ok_placa:
            _bajar_tornillo(work)
        if sin_cambio():
            break

        for k in CLAVES_GEO:
            if _cambiado(work.get(k), antes.get(k)):
                changes.append(f"{k} {_fmt(antes.get(k))}->{_fmt(work.get(k))}")
    else:
        calc, fallos = _evaluar_geometria(work)

    if fallos and calc is not None:
        # (p. ej. bc&bb: ala del pilar mas estrecha que la viga, no se arregla con la geometria)
        log.warning("refinar %s: geometria sin resolver: %s", work.get("Unicode"), sorted(fallos))
    if changes and calc is not None:
        _actualizar_lp(work, calc)
    return changes


# Elementos que SOBRAN cuando la demanda residual es <= 0 (el perfil solo ya aguanta):
_SOBRANTES = (
    ("Ffts_", "rigidizadores transversales", ("ThicknessName_t_tenstiff", "ThicknessName_t_compstiff")),
    ("Vfdp_", "placa de refuerzo del alma", ("ThicknessName_t_webstiff",)),
    ("C1wsAME_", "rigidizadores de cartela", ("ThicknessName_t_BeamHcompstiff",)),
)


def _residuos(calc) -> Dict[str, float]:
    """Residuos del calculador (no dependen de los espesores, solo de perfiles y cargas)."""
    return {attr: float(getattr(calc, attr)) for attr, _, _ in _SOBRANTES}


def _fallan(rs: Dict[str, float]) -> set:
    return {k for k, v in rs.items() if _variable_for(k) and v > TARGET_RATIO}


def _quitar_sobrantes(work: Dict[str, Any], rs: Dict[str, float], res: Dict[str, float]):
    """Pone a 0 los elementos con residuo <= 0, siempre que sin ellos no falle nada nuevo."""
    quitados: List[str] = []
    presentes = [(a, n, c) for a, n, c in _SOBRANTES if any(_mm(work.get(x)) > 0 for x in c)]
    if not presentes:
        return quitados, rs
    candidatos = [(n, c) for a, n, c in presentes if res[a] <= 0]
    for nombre, cols in candidatos:
        prueba = dict(work)
        for c in cols:
            prueba[c] = "0mm"
        _refresh_web_plate(prueba)
        rs_p = _ratios(prueba)
        if _fallan(rs_p) <= _fallan(rs):
            work.clear(); work.update(prueba); rs = rs_p
            quitados.append(f"{nombre} quitados")
    return quitados, rs


# ---------------------------------------------------------------------------
# Maximizar la carga
# ---------------------------------------------------------------------------
MAXIMIZE_LOAD = os.environ.get("AI_MAXIMIZE_LOAD", "1").strip().lower() not in ("0", "false", "no")
RATIO_TOPE = 0.85
LOAD_MAX_PCT = 100

_COMBINADO = "Bolts in combined"
_FAM_CORTANTE = ("Bolts in Shear", _COMBINADO)
_FAM_AXIL_MOMENTO = ("Bolts in Tension", "End plate", "Shear yielding", "Shear rupture",
                     "Column flange", _COMBINADO)


def _pct(v) -> int:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return 0


def _limites(base: Dict[str, float], familias) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Limites a partir de los ratios iniciales: (por familia, por ratio suelto)."""
    fam = {f: max(RATIO_TOPE, _fam_max(base, f)) for f in familias}
    resto = {k: max(1.0, v) for k, v in base.items()
             if "Weld" not in k and not k.startswith(tuple(familias))}
    return fam, resto


def _fam_max(rs: Dict[str, float], prefijo: str) -> float:
    return max([v for k, v in rs.items() if k.startswith(prefijo)] or [0.0])


def _cumple(rs: Dict[str, float], fam: Dict[str, float], resto: Dict[str, float]) -> bool:
    eps = 1e-9
    if any(_fam_max(rs, f) > lim + eps for f, lim in fam.items()):
        return False
    return all(v <= resto.get(k, 1.0) + eps for k, v in rs.items()
               if k in resto or ("Weld" not in k and not k.startswith(tuple(fam))))


def _subir(work: Dict[str, Any], keys, familias, rs: Dict[str, float]):
    """Sube `keys` (porcentajes de carga, escalados juntos) hasta el mayor valor que cumple."""
    ini = {k: _pct(work.get(k)) for k in keys}
    base = max(ini.values())
    if base <= 0 or base >= LOAD_MAX_PCT:
        return None, rs
    fam, resto = _limites(rs, familias)
    cache: Dict[int, Optional[Dict[str, float]]] = {}

    def probar(s: int) -> bool:
        if s not in cache:
            prueba = dict(work)
            prueba.update({k: min(LOAD_MAX_PCT, int(round(v * s / base))) for k, v in ini.items()})
            r = _ratios(prueba)
            cache[s] = r if _cumple(r, fam, resto) else None
        return cache[s] is not None

    ok, paso = base, 1
    ko = None
    while ok < LOAD_MAX_PCT:                      # fase de saltos
        s = min(LOAD_MAX_PCT, ok + paso)
        if probar(s):
            ok, paso = s, paso * 2
        else:
            ko = s
            break
    while ko is not None and ko - ok > 1:         # biseccion entre ok (cumple) y ko (falla)
        mid = (ok + ko) // 2
        if probar(mid):
            ok = mid
        else:
            ko = mid
    if ok == base:
        return None, rs
    work.update({k: min(LOAD_MAX_PCT, int(round(v * ok / base))) for k, v in ini.items()})
    return ("/".join(str(ini[k]) for k in keys) + "->" + "/".join(str(_pct(work[k])) for k in keys)), cache[ok]


def _maximize_load(work: Dict[str, Any], rs: Dict[str, float]) -> List[str]:
    """Fase 1: cortante. Fase 2: axil + momento. Devuelve los cambios hechos."""
    out: List[str] = []
    c, rs = _subir(work, ("FyEd_Porcen",), _FAM_CORTANTE, rs)
    if c:
        out.append(f"cortante {c}")
    c, rs = _subir(work, ("MzEd_Porcen", "FxEd_Porcen"), _FAM_AXIL_MOMENTO, rs)
    if c:
        out.append(f"momento/axil {c}")
    return out


def _resumen(changes: List[str]) -> str:
    """Agrupa los cambios por variable: 'tp 12->15' + 'tp 15->20' -> 'tp 12->20'."""
    out: List[str] = []
    seen: Dict[str, int] = {}
    for c in changes:
        var, _, rng = c.partition(" ")
        if "->" in rng and var in seen and out[seen[var]].startswith(var + " "):
            ini = out[seen[var]].split(" ", 1)[1].split("->", 1)[0]
            out[seen[var]] = f"{var} {ini}->{rng.split('->', 1)[1]}"
        else:
            seen[var] = len(out)
            out.append(c)
    return " | ".join(out)


def refine(row: Dict[str, Any], maximize_load: Optional[bool] = None) -> Tuple[Dict[str, Any], List[str]]:
    """Fila CODAME -> (fila ajustada, lista de cambios). Nunca lanza: ante un error devuelve la original."""
    uid = row.get("Unicode")
    changes: List[str] = []
    try:
        work = dict(row)

        # 1. Corrección geométrica iterativa (un fallo aqui no debe tirar el resto del refinado)
        try:
            changes.extend(_ajustar_geometria(work))
        except Exception:  # noqa: BLE001
            log.exception("refinar %s: ajuste geometrico fallo; se sigue sin el", uid)

        # 2. Ajuste de espesores de placas
        blocked = set()
        rs, calc = _eval(work)
        res = _residuos(calc)
        for _ in range(MAX_STEPS):
            failing = {k: v for k, v in rs.items()
                       if v > TARGET_RATIO and _variable_for(k) and k not in blocked}
            if not failing:
                break
            worst = max(failing, key=failing.get)
            change = _apply(work, _variable_for(worst))
            if change is None:
                blocked.add(worst)
                log.warning("refinar %s: %s sigue en %.2f (sin mas catalogo)",
                            uid, worst, failing[worst])
                continue
            changes.append(change)
            rs, _calc = _eval(work)

        # 3. Quitar sobrantes y maximizar cargas
        quitados, rs = _quitar_sobrantes(work, rs, res)
        changes += quitados
        if (MAXIMIZE_LOAD if maximize_load is None else maximize_load):
            changes += _maximize_load(work, rs)

        if changes:
            log.info("refinar %s: %s", uid, _resumen(changes))
        return work, changes
    except Exception:  # noqa: BLE001
        log.exception("refinar fallo (UNICODE=%s); se deja la fila sin ajustar", uid)
        return row, []


# ---------------------------------------------------------------------------
# Paralelismo (interfaz facil de sustituir: local -> Ray)
# ---------------------------------------------------------------------------
WORKERS = int(os.environ.get("AI_REFINE_WORKERS", "0") or 0) or min(8, os.cpu_count() or 1)
PARALLEL_MIN_ROWS = 4


def _init_worker() -> None:
    """Los procesos hijos no heredan el logging: escriben en el mismo archivo de sesion."""
    try:
        import config
        destino = os.environ.get("AI_LOG_FILE") or str(config.LOG_PATH)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(destino, encoding="utf-8")],
        )
    except Exception:  # noqa: BLE001
        pass
    # Warm-up: con `spawn` cada worker leeria los 4 CSV de Tekla en su primera tarea.
    try:
        import catalogo
        catalogo.profile_lookup()
    except Exception:  # noqa: BLE001
        log.exception("refinar: no se pudo precargar el catalogo en el worker")


def refine_and_check(row: Dict[str, Any], with_check: bool = True,
                     maximize_load: Optional[bool] = None) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Ajusta una fila y, si se pide, devuelve ademas su comprobacion final (None si fallo)."""
    new, _ = refine(row, maximize_load)
    if not with_check:
        return new, None
    try:
        return new, check_single_row(dict(new))
    except Exception:  # noqa: BLE001  (el llamador la repite y deja constancia del error)
        return new, None


class RefineTimeout(TimeoutError):
    """El refinado paralelo supero AI_REFINE_TIMEOUT_S."""


class RefineBackend:
    """Interfaz minima: mapear filas -> [(fila, check | None)]."""

    name = "base"

    def start(self) -> None:
        pass

    def map_refine(self, rows: List[Dict[str, Any]], with_check: bool,
                   maximize_load: Optional[bool] = None):
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class SerialBackend(RefineBackend):
    """Ejecucion en el proceso actual (debug o entornos sin pool)."""

    name = "serial"

    def map_refine(self, rows: List[Dict[str, Any]], with_check: bool,
                   maximize_load: Optional[bool] = None):
        return [refine_and_check(r, with_check, maximize_load) for r in rows]


class ProcessPoolBackend(RefineBackend):
    """Pool de procesos con timeout global y recreacion segura.

    Para escalar con Ray basta crear un `RayBackend(RefineBackend)` y devolverlo desde
    `_make_backend()` (ver doc/ray_escalado.md); el resto del servidor no cambia.
    """

    name = "process"

    def __init__(self) -> None:
        self._executor: Optional[ProcessPoolExecutor] = None
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(
                    max_workers=WORKERS,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                )
                log.info("refinar: pool de procesos iniciado (%d workers)", WORKERS)

    def _get(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self.start()
        return self._executor

    def _drop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def reset(self) -> None:
        with self._lock:
            self._drop()

    def _drop_if_current(self, executor: ProcessPoolExecutor) -> None:
        """Reinicia el pool solo si sigue siendo el que uso esta peticion."""
        with self._lock:
            if self._executor is executor:
                self._drop()

    def map_refine(self, rows: List[Dict[str, Any]], with_check: bool,
                   maximize_load: Optional[bool] = None):
        if not rows:
            return []
        # El lock solo protege la creacion del pool y el submit. La espera queda fuera
        # para que varios escenarios/peticiones alimenten los workers a la vez.
        with self._lock:
            executor = self._get()
            futures = [
                executor.submit(refine_and_check, row, with_check, maximize_load)
                for row in rows
            ]
        # Plazo proporcional al lote: base + por_fila * rondas necesarias con WORKERS procesos.
        rondas = math.ceil(len(rows) / max(1, WORKERS))
        timeout_s = max(1.0, config.REFINE_TIMEOUT_S) + max(0.0, config.REFINE_TIMEOUT_PER_ROW_S) * rondas
        deadline = time.monotonic() + timeout_s
        out = []
        try:
            for fut in futures:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                out.append(fut.result(timeout=remaining))
            return out
        except TimeoutError:
            # No se reinicia el pool: es compartido y hacerlo tiraria el trabajo de las demas
            # peticiones en vuelo (y dejaria procesos huerfanos compitiendo por CPU).
            # Solo se cancelan las tareas de ESTA peticion que aun no han empezado.
            pendientes = sum(1 for fut in futures if fut.cancel())
            log.error(
                "refinar: timeout de %.0fs para %d filas (%d sin empezar canceladas); pool conservado",
                timeout_s, len(rows), pendientes,
            )
            raise RefineTimeout(
                f"El refinado supero {timeout_s:.0f}s para {len(rows)} filas"
            )
        except Exception:
            for fut in futures:
                fut.cancel()
            self._drop_if_current(executor)
            raise

    def close(self) -> None:
        with self._lock:
            if self._executor is not None:
                log.info("refinar: cerrando pool de procesos")
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None


_BACKEND: Optional[RefineBackend] = None
_BACKEND_LOCK = threading.Lock()


def _make_backend() -> RefineBackend:
    kind = (config.REFINE_BACKEND or "process").lower()
    if kind == "serial":
        return SerialBackend()
    if kind == "process":
        return ProcessPoolBackend()
    log.warning("AI_REFINE_BACKEND=%s no soportado; se usa 'process'", kind)
    return ProcessPoolBackend()


def get_backend() -> RefineBackend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = _make_backend()
        return _BACKEND


def start_backend() -> None:
    get_backend().start()


def shutdown_backend() -> None:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is not None:
            try:
                _BACKEND.close()
            finally:
                _BACKEND = None


def refine_many(
    rows: List[Dict[str, Any]],
    with_check: bool = True,
    maximize_load: Optional[bool] = None,
):
    """[(fila ajustada, comprobacion | None)] en el mismo orden; en paralelo si compensa."""
    if not ENABLED:
        return [(r, None) for r in rows]
    if not rows:
        return []

    backend = get_backend()
    if backend.name == "process" and (WORKERS <= 1 or len(rows) < PARALLEL_MIN_ROWS):
        return [refine_and_check(r, with_check, maximize_load) for r in rows]

    try:
        return backend.map_refine(rows, with_check, maximize_load)
    except RefineTimeout:
        raise
    except Exception:  # noqa: BLE001 (pool roto, sin recursos...): se sigue en serie
        log.exception("refinar en %s fallo; se calcula en serie", backend.name)
        backend.reset()
        return [refine_and_check(r, with_check, maximize_load) for r in rows]
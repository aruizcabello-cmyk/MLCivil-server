"""Mapeo input + diseno -> 63 columnas de CODAME (`Tabla_MOMENT_M1`), con imputacion.

Vive en el servidor (fuente unica para todos los modelos) y usa `catalogo` (datos en
`production/server/data`). Devuelve todas las columnas del esquema CODAME con su tipo y
formato correctos; `codame_schema.fill_missing` actua como red de seguridad.

Referencia: `doc/codame/mapeo_columnas_codame.md`.

Reglas con criterio de ingenieria PROVISIONAL (marcadas con "PROVISIONAL"): ancho de
chapa, Bws/Lws, cantonera y largo de chapa. Estan aisladas en funciones y constantes para
poder ajustarlas sin tocar el resto.

Reglas de geometria de chapa/cantonera (`ajustar_cantonera_y_chapa`, tambien la usa refinar.py):
  - Lh = FactorCantoCantonera * d_viga / sin(AnguloCantonera)   (0.7 y 60 grados por defecto).
  - Lh es un DATO: se queda con el valor de la formula y NUNCA se modifica al corregir.
    Si el ala de la cantonera cae sobre una fila de tornillos (a menos del `clearance`),
    se adaptan las FILAS a ella: las intermedias se suben hasta quedar por encima del ala
    y la ultima baja hasta quedar por debajo (anadiendo filas si hace falta).
  - El ultimo tornillo va siempre por debajo de la viga/cantonera (+ holgura) y la chapa
    termina `e` mm por debajo del ultimo tornillo: Lp = ultima fila + e.
  - b_ep queda entre el ala de la viga y el ala del pilar (punto medio si el modelo se sale).
"""
import logging
import math

from catalogo import ACERO_TABLE, DEFAULT_GRADO_ACERO, DEFAULT_CLASE_TORNILLO, get_profile
from externos import Externos, tablapernos, tablaplaca

log = logging.getLogger("ai-server")

# ---------------------------------------------------------------------------
# Valores por defecto de la tabla CODAME (Cambios_a_aplicar.txt)
# ---------------------------------------------------------------------------
DEFAULT_WELD_MM = 6.0             # wf, ww, w3, w4, wh
DEFAULT_PR_PORCEN = 50
DEFAULT_TRACCION = True
DEFAULT_ELECTRODE = "E49XX"
DEFAULT_SISUNI = True             # SisUniBoltSI, SisUniPlateSI
DEFAULT_DESIGNATION = "Tekla19"
DEFAULT_E_MPA = 200000.0
DEFAULT_HAUNCH_ANGLE_DEG = 60.0   # solo si hay cantonera (Lh > 0)
DEFAULT_HAUNCH_FACTOR = 0.7       # solo si hay cantonera (Lh > 0)

# Red de seguridad: pfi minimo (mm) que se garantiza aqui si el modelo entrega un
# p1 incoherente. El minimo "de diseno" lo aplica el adaptador del modelo (PFI_MIN_MM).
PFI_MIN_SEGURIDAD = 5.0

# PROVISIONAL: holgura (mm) entre la placa del alma de la columna y los acuerdos / rigidizadores.
WEB_PLATE_GAP_MM = 5.0

# Claves de separacion entre filas de tornillos (la 1a, pfo+pfi+tf, va aparte)
_SEP_KEYS = ("pb", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12")


def _num(v, default=0.0):
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _mm_text(v) -> str:
    """CODAME guarda espesores como texto: '20mm'."""
    return f"{int(round(_num(v)))}mm"


def _ceil5(x: float) -> float:
    return math.ceil(round(x, 6) / 5.0) * 5.0


def _floor5(x: float) -> float:
    return math.floor(round(x, 6) / 5.0) * 5.0


def _round5(x: float) -> float:
    return round(x / 5.0) * 5.0


def _pct_int(ratio) -> int:
    """Ratio 0-1 -> porcentaje entero (sin decimales ni ruido de coma flotante)."""
    return int(round(_num(ratio) * 100.0))


def _infer_source(beam) -> str:
    return "Canadian" if str(beam).strip().upper().startswith("W") else "European"


def _n_years() -> int:
    return 12


def _geom(profile, col, default=0.0):
    return _num(profile.get(col), default) if profile is not None else default


def _tension_rows(design_row: dict, beam_d_mm: float) -> int:
    """ntensionrows: 3 si hay >=2 filas en [d, d+BEAM_d], si no 2 (logica del cliente)."""
    d = _num(design_row.get("d"))
    finish = d + _num(beam_d_mm)
    position = 0.0
    rows_inside = 0
    if d <= position <= finish:
        rows_inside += 1
    for n in range(1, _n_years() + 1):
        p = _num(design_row.get(f"p{n}"))
        if p == 0.0:
            break
        position += p
        if d <= position <= finish:
            rows_inside += 1
    return 3 if rows_inside >= 2 else 2


def _plate_width(bp, g, bolt, bf_beam, bf_col, uid) -> float:
    """b_ep: ancho de la chapa. PROVISIONAL.

    Debe ser MAYOR que el ala de la viga (con +20 mm de margen si cabe) y MENOR que el ala
    del pilar. Se respeta lo que predice el modelo si esta dentro de ese rango; si se sale
    (o iguala el ala del pilar) se lleva al punto medio entre ambas alas (multiplo de 5).
      - minimo por distancia al borde: g + 2*Le_min del tornillo,
      - maximo: ancho del ala del pilar.
    Si el minimo no cabe en el ala del pilar, se deja el valor del modelo.
    """
    try:
        le_min = tablapernos.DATOS_PERNOS[int(bolt)]["Le_min"]
    except (KeyError, TypeError, ValueError):
        le_min = 0.0
    lo_bolts = g + 2.0 * le_min
    if bf_col <= 0:
        return float(math.ceil(max(bp, lo_bolts, bf_beam)))

    hi = bf_col
    lo = max(lo_bolts, _ceil5(bf_beam + 20.0) if bf_beam > 0 else 0.0)
    if lo > hi:                                  # no cabe el +20: basta con superar el ala
        lo = max(lo_bolts, bf_beam)
    if lo > hi:
        log.warning("b_ep: no cabe g + 2*Le_min = %.0f en el ala del pilar (%.0f); "
                    "se mantiene el valor del modelo (UNICODE=%s)", lo_bolts, hi, uid)
        return bp

    medio = min(max(_round5((bf_beam + bf_col) / 2.0), lo), hi)
    new = bp if lo <= bp < hi else medio
    new = float(math.ceil(new))
    if new != bp:
        log.info("b_ep %.0f -> %.0f (min %.0f, max %.0f, medio %.0f) (UNICODE=%s)",
                 bp, new, lo, hi, medio, uid)
    return new


def _web_plate_dims(col, beam_d, beam_tf, t_stiff):
    """(Bws, Lws) de la placa del alma de la columna. PROVISIONAL.

    Hipotesis: placa paralela al alma, entre las alas de la columna.
      Bws = alma libre entre acuerdos: d_col - 2*tf_col - 2*r_col, menos holgura a cada lado.
      Lws = entre rigidizadores transversales (a la altura de las alas de la viga):
            (d_viga - tf_viga) - t_rigidizador, menos holgura a cada lado.
    No tiene en cuenta los tornillos (por confirmar).
    """
    if col is None:
        return 0.0, 0.0
    hw = _geom(col, "d_mm") - 2.0 * _geom(col, "tf_mm") - 2.0 * _geom(col, "r1_mm")
    bws = _floor5(hw - 2.0 * WEB_PLATE_GAP_MM)
    lws = _floor5(beam_d - beam_tf - t_stiff - 2.0 * WEB_PLATE_GAP_MM)
    return max(bws, 0.0), max(lws, 0.0)


def _haunch_length(beam_d, n_cantonera, factor=DEFAULT_HAUNCH_FACTOR,
                   angle_deg=DEFAULT_HAUNCH_ANGLE_DEG) -> float:
    """Lh = FactorCantoCantonera * h_viga / sin(angulo)  (formula del C#). 0 si no hay cantonera."""
    if n_cantonera <= 0:
        return 0.0
    return factor * beam_d / math.sin(math.radians(angle_deg))


# ---------------------------------------------------------------------------
# Cantonera y largo de chapa (comun a codame.py y refinar.py)
# ---------------------------------------------------------------------------
_EXT = None


def _redond5(x: float) -> float:
    """Mismo redondeo a multiplo de 5 que usa el calculador para c = pfo + pfi + tf."""
    global _EXT
    try:
        if _EXT is None:
            _EXT = Externos()
        return float(_EXT.moduloauxiliar.redond_mult(x, 5.0))
    except Exception:  # noqa: BLE001
        return _round5(x)


def _limites_fisicos(bolt_name, tp_text) -> dict:
    """{'clearance', 'minsp', 'maxsp'} (mm) del tornillo y la chapa. Si no hay `clearance`
    en la tabla de tornillos se usa el diametro del tornillo como holgura minima."""
    out = {"clearance": 0.0, "minsp": 0.0, "maxsp": math.inf}
    db = 0.0
    try:
        tb = tablapernos()
        tb.nombretornillo = str(bolt_name)
        out["minsp"] = _num(tb.pitchminmet)
        db = _num(tb.dmetrico)
    except Exception:  # noqa: BLE001
        pass
    try:
        tpl = tablaplaca()
        tpl.nombreplaca = str(tp_text)
        mx = _num(tpl.pitchmaxmet)
        out["maxsp"] = mx if mx > 0 else math.inf
    except Exception:  # noqa: BLE001
        pass
    digitos = "".join(ch for ch in str(bolt_name) if ch.isdigit())
    cl = _num(tablapernos.DATOS_PERNOS.get(int(digitos), {}).get("clearance")) if digitos else 0.0
    out["clearance"] = cl if cl > 0 else (db or (float(digitos) if digitos else 0.0))
    return out


def diametro_tornillo(bolt_name) -> float:
    """'M24' -> 24.0 (0.0 si no hay digitos)."""
    digitos = "".join(ch for ch in str(bolt_name) if ch.isdigit())
    return float(digitos) if digitos else 0.0


def ajustar_cantonera_y_chapa(row: dict, limites: dict = None) -> bool:
    """Aplica sobre una fila CODAME (la modifica) las reglas de cantonera y largo de chapa:

      1. Con cantonera: si su ala toca una fila de tornillos, mueve las filas (nunca Lh):
         las intermedias suben hasta quedar sobre el ala y la ultima queda bajo ella.
         Si no hay hueco, anade filas por debajo.
      2. El ultimo tornillo queda siempre por debajo de la viga/cantonera (+ holgura).
      3. Lp = ultima fila + diametro del tornillo + e.
    La ultima fila queda en e + pfo + d_viga + Lh + holgura (clearance): se SUBE si queda corta
    y se ACERCA si el modelo la dejo mas abajo (antes solo se subia, y quedaba lejisimos).

    `limites` (opcional): {'clearance', 'minsp', 'maxsp', 'c'}; si falta se obtiene de las tablas.
    No toca dev, pfo ni pfi. Devuelve True si cambio la geometria (separaciones o cantonera).
    """
    beam = get_profile(str(row.get("BEAM", "")).strip())
    h, tf = _geom(beam, "d_mm"), _geom(beam, "tf_mm")
    if beam is None or h <= 0 or tf <= 0:
        return False

    lim = limites or _limites_fisicos(row.get("BoltName_ep", ""), row.get("ThicknessName_t_ep", "0mm"))
    m = _num(lim.get("clearance"))
    minsp = _num(lim.get("minsp"))
    maxsp = _num(lim.get("maxsp"), math.inf)
    if maxsp <= 0:
        maxsp = math.inf
    if m <= 0:
        return False

    dev, pfo, pfi = _num(row.get("dev")), _num(row.get("pfo")), _num(row.get("pfi"))
    c = _num(lim.get("c")) or _redond5(pfo + pfi + tf)
    seps = {k: _num(row.get(k)) for k in _SEP_KEYS}
    seps0 = dict(seps)

    fac0 = _num(row.get("FactorCantoCantonera"))
    ang = _num(row.get("AnguloCantonera"))
    tiene = ang != 0.0 and fac0 > 0.0
    nueva = False                           # cantonera que no predijo el modelo y hay que crear
    yb = dev + pfo + h                      # cara inferior del ala inferior de la viga
    if not tiene:                           # sin cantonera: bajo el ala solo puede haber 1 fila
        ys0 = [dev, dev + c]
        for k in _SEP_KEYS:
            if seps[k] > 0:
                ys0.append(ys0[-1] + seps[k])
        if sum(1 for y in ys0 if y > yb) > 1:
            tiene, ang, fac0, nueva = True, DEFAULT_HAUNCH_ANGLE_DEG, 0.0, True
            row["AnguloCantonera"] = ang
    sin_a = math.sin(math.radians(ang)) if tiene else 1.0
    fac = DEFAULT_HAUNCH_FACTOR if nueva else fac0     # Lh = fac*h/sen(ang): dato, no se ajusta

    def filas():
        ys = [dev, dev + c]
        y = dev + c
        for k in _SEP_KEYS:
            if seps[k] > 0:
                y += seps[k]
                ys.append(y)
        return ys

    def ultima():
        return max((i for i, k in enumerate(_SEP_KEYS) if seps[k] > 0), default=-1)

    def anadir(delta) -> bool:
        i = ultima() + 1
        if i >= len(_SEP_KEYS):
            return False
        seps[_SEP_KEYS[i]] = delta
        return True

    def paso_valido(delta) -> float:
        tope = delta if math.isinf(maxsp) else min(delta, _floor5(maxsp))
        return max(tope, _ceil5(minsp))

    def llevar_ultima(objetivo) -> bool:
        """Lleva la ultima fila hasta `objetivo` (alargando su separacion o anadiendo filas)."""
        for _ in range(6):
            delta = _ceil5(objetivo - filas()[-1])
            if delta <= 0:
                return True
            i = ultima()
            if i >= 0 and seps[_SEP_KEYS[i]] + delta <= maxsp:
                seps[_SEP_KEYS[i]] += delta
                return True
            if not anadir(paso_valido(delta)):
                if i >= 0:
                    seps[_SEP_KEYS[i]] += delta
                    return True
                return False
        return False

    def acercar_ultima(objetivo) -> None:
        """Si la ultima fila queda MAS ABAJO de `objetivo` (el modelo la dejo lejos, o una
        correccion de separacion maxima la dejo clavada en el tope), reduce su separacion
        hasta `objetivo`, sin bajar del minimo entre tornillos."""
        i = ultima()
        if i < 0:                                      # solo filas 0 y 1 (dev y pfi): no se toca
            return
        ys = filas()
        if ys[-1] <= objetivo:
            return
        delta = max(_ceil5(objetivo - ys[-2]), _ceil5(minsp), 5.0)
        if delta < seps[_SEP_KEYS[i]] and delta <= maxsp:
            seps[_SEP_KEYS[i]] = delta

    if tiene:
        ye = yb + fac * h / sin_a                      # cara inferior del ala de la cantonera (Lh fijo)

        def claro() -> bool:
            ys = filas()
            return (all(abs(y - (ye - tf)) >= m and abs(y - ye) >= m for y in ys)
                    and ys[-1] >= ye + m
                    and sum(1 for y in ys if y > ye) == 1)

        if not claro():
            # Todas las filas salvo la ultima por encima del ala; la ultima por debajo.
            ys = filas()
            n = len(ys)
            claves = [k for k in _SEP_KEYS if seps[k] > 0]
            obj = list(ys)
            tope = _floor5(ye - tf - m)
            for j in range(n - 2, 1, -1):              # intermedias: suben si tocan el ala
                obj[j] = min(obj[j], tope)
                tope = obj[j] - _ceil5(minsp)
            if n >= 3 and obj[-1] - obj[-2] > maxsp:
                obj[-1] = obj[-2] + _floor5(maxsp)
            valido = all(obj[j] - obj[j - 1] >= minsp - 1e-9 for j in range(2, n)) and \
                all(obj[j] - obj[j - 1] <= maxsp + 1e-9 for j in range(2, n))
            if valido:
                for j in range(2, n):
                    seps[claves[j - 2]] = obj[j] - obj[j - 1]
                if filas()[-1] < ye + m:               # la ultima, por debajo del ala
                    llevar_ultima(ye + m)
            if not claro():
                log.warning("cantonera: no se pudo despejar el ala de los tornillos con Lh fijo "
                            "(UNICODE=%s)", row.get("Unicode"))
        if claro():                                    # ultima fila = ye + holgura, ni mas abajo
            acercar_ultima(ye + m)
    else:
        objetivo = _ceil5(yb + max(pfo, m + 0.5))
        if filas()[-1] < yb + m:
            llevar_ultima(objetivo)
        else:
            acercar_ultima(objetivo)

    cambio = any(abs(seps[k] - seps0[k]) > 1e-9 for k in _SEP_KEYS) or abs(fac - fac0) > 1e-9
    for k in _SEP_KEYS:
        if abs(seps[k] - seps0[k]) > 1e-9:
            row[k] = seps[k]
    if abs(fac - fac0) > 1e-9:
        row["FactorCantoCantonera"] = fac
        row["Lh"] = fac * h / sin_a
        log.info("cantonera creada: Factor %.3f, Lh = %.1f (UNICODE=%s)",
                 fac, row["Lh"], row.get("Unicode"))
    db = _num(lim.get("db")) or diametro_tornillo(row.get("BoltName_ep"))
    row["Lp"] = filas()[-1] + db + dev                 # ultima fila + diametro del tornillo + e
    return cambio


def to_codame(input_row: dict, design_row: dict, apply_adjustments: bool = True) -> dict:
    """Input + diseno -> fila de CODAME (todas las columnas, imputadas).

    `apply_adjustments=False` devuelve la prediccion del modelo SIN los ajustes
    (no recorta b_ep, no corrige pfi, no recoloca filas/cantonera/Lp). Se mantienen
    las derivaciones puras (pfo, pfi, pb, Lh, Bws/Lws, textos mm, Lp=2e+sum(p)).
    """
    inp = input_row or {}
    d = design_row or {}
    uid = inp.get("UNICODE")

    beam = str(inp.get("BEAM", "")).strip()
    column = str(inp.get("COLUMN", "")).strip()
    geom = get_profile(beam)
    col_geom = get_profile(column)
    tf = _geom(geom, "tf_mm")
    beam_d = _geom(geom, "d_mm")

    e = _num(d.get("e"))
    g = _num(d.get("g"))
    p = {n: _num(d.get(f"p{n}")) for n in range(1, _n_years() + 1)}

    grado = str(inp.get("grado_acero") or DEFAULT_GRADO_ACERO).strip()
    acero = ACERO_TABLE.get(grado, ACERO_TABLE[DEFAULT_GRADO_ACERO])
    fy = _num(inp.get("fy_acero_mpa", acero["fy_acero_mpa"]))
    fu = _num(inp.get("fu_acero_mpa", acero["fu_acero_mpa"]))

    bolt = d.get("Bolt")
    try:
        bolt_name = f"M{int(bolt)}" if bolt is not None else ""
    except (TypeError, ValueError):
        bolt_name = str(bolt or "")

    pfo = _num(d.get("d"))

    # pfi = p1 - pfo - tf debe ser siempre positivo. Si el modelo entrega un p1
    # incoherente se corrige aqui (red de seguridad) y se deja constancia en el log.
    if p[1] <= 0:
        log.error("p1 ausente o 0 en el diseno (UNICODE=%s): pfi no es fiable", uid)
    elif apply_adjustments and p[1] - pfo - tf < PFI_MIN_SEGURIDAD:
        nuevo_p1 = math.ceil((pfo + tf + PFI_MIN_SEGURIDAD) / 5.0) * 5.0
        log.warning("pfi=%.1f < %.1f; p1 %.1f -> %.1f (UNICODE=%s)",
                    p[1] - pfo - tf, PFI_MIN_SEGURIDAD, p[1], nuevo_p1, uid)
        p[1] = nuevo_p1
    p1 = p[1]
    pfi = p1 - pfo - tf

    # Lp = suma de todas las p + 2e (e = distancia del primer y ultimo tornillo al borde).
    # (ajustar_cantonera_y_chapa lo recalcula al final con las filas definitivas)
    lp = 2.0 * e + sum(p.values())

    # ts y tst son siempre iguales: se toma tst (su catalogo {0,10,15,20} esta contenido
    # en el de ts) y se copia. Si un modelo antiguo solo trae ts, se usa ese.
    t_stiff = _num(d["tst"]) if "tst" in d else _num(d.get("ts"))

    b_ep = (
        _plate_width(
            _num(d.get("Bp")), g, bolt,
            _geom(geom, "b_mm") or _geom(geom, "bf_mm"),
            _geom(col_geom, "b_mm") or _geom(col_geom, "bf_mm"), uid,
        )
        if apply_adjustments else _num(d.get("Bp"))
    )

    tws = _num(d.get("tws"))
    bws, lws = _web_plate_dims(col_geom, beam_d, tf, t_stiff) if tws > 0 else (0.0, 0.0)

    # Asegurar que si t_stiff o tbs son <= 0 se representen como "0mm"
    t_stiff_text = _mm_text(t_stiff) if t_stiff > 0 else "0mm"
    tbs_text = _mm_text(d.get("tbs")) if _num(d.get("tbs")) > 0 else "0mm"
    tws_text = _mm_text(tws) if tws > 0 else "0mm"

    n_cantonera = int(_num(d.get("n_cantonera")))
    lh = _haunch_length(beam_d, n_cantonera)
    has_haunch = lh > 0

    row = {
        "Unicode": uid if uid is not None else "",
        "Normativas": inp.get("Normativas", ""),
        "BEAM": beam,
        "COLUMN": column,
        "MzEd_Porcen": _pct_int(inp.get("ratio_moment")),
        "FyEd_Porcen": _pct_int(inp.get("ratio_shear")),
        "FxEd_Porcen": _pct_int(inp.get("ratio_axial")),
        "Pr_Porcen": DEFAULT_PR_PORCEN,
        "Traccion": DEFAULT_TRACCION,
        "MzEd_PanelZone": 0.0,
        "Vgra": 0.0,
        "Havg": 0.0,
        "BoltName_ep": bolt_name,
        "p2_ep": g,
        "ThicknessName_t_webstiff": tws_text,
        "Bws": bws,
        "Lws": lws,
        "b_ep": b_ep,
        "Lp": lp,
        "ThicknessName_t_ep": _mm_text(d.get("tp")),
        "Lh": lh,
        "AnguloCantonera": DEFAULT_HAUNCH_ANGLE_DEG if has_haunch else 0.0,
        "dev": e,
        "p12_Outer": 0.0,
        "pfo": pfo,
        "pfi": pfi,
        "pb": p[2],
        "ThicknessName_t_compstiff": t_stiff_text,
        "ThicknessName_t_BeamHcompstiff": tbs_text,
        "ThicknessName_t_tenstiff": t_stiff_text,
        "wf": DEFAULT_WELD_MM,
        "ww": DEFAULT_WELD_MM,
        "w3": DEFAULT_WELD_MM,
        "w4": DEFAULT_WELD_MM,
        "wh": DEFAULT_WELD_MM,
        "FactorCantoCantonera": DEFAULT_HAUNCH_FACTOR if has_haunch else 0.0,
        "ntensionrows": _tension_rows({**d, "p1": p1}, beam_d),
        "fy": fy,
        "fu": fu,
        "fyp": fy,
        "fup": fu,
        "E": DEFAULT_E_MPA,
        "BoltGrade": str(inp.get("clase_tornillo") or DEFAULT_CLASE_TORNILLO).strip(),
        "BoltThreads": True,
        "Electrode": DEFAULT_ELECTRODE,
        "yieldmethod": True,
        "WeldFactor": 1,
        "BoltsInTensionFactor": 2,
        "SisUniBoltSI": DEFAULT_SISUNI,
        "SisUniPlateSI": DEFAULT_SISUNI,
        "Source": _infer_source(beam),
        "Designation": DEFAULT_DESIGNATION,
    }
    for n in range(3, _n_years() + 1):
        row[f"p{n}"] = p[n]

    # Cantonera y largo de chapa (ultimo tornillo por debajo de viga/cantonera, Lp = ultima fila + e)
    if apply_adjustments:
        try:
            ajustar_cantonera_y_chapa(row)
        except Exception:  # noqa: BLE001
            log.exception("ajustar_cantonera_y_chapa fallo (UNICODE=%s); se deja la fila del modelo", uid)
    return row
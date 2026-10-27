"""
Modulo de comprobacion estructural (AISC 360-10, calculador Python).

Convierte una fila del esquema CODAME (Tabla_MOMENT_M1, editada por el revisor) a
`ConexionInput` y ejecuta `CalculadorAISC36010`. Usa el catalogo del servidor
(`catalogo.py`, datos en `production/server/data`); no depende de ningun modelo.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from modelos import ConexionInput, Perfil
from calculador import CalculadorAISC36010
from catalogo import (
    ACERO_TABLE,
    DEFAULT_GRADO_ACERO,
    DEFAULT_CLASE_TORNILLO,
    get_profile,
)

P_COLUMNS = [f"p{i}" for i in range(1, 13)]


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool(v, default=True):
    """Booleano robusto: la tabla se edita en la web y llega como texto ("true"/"false")."""
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    t = str(v).strip().lower()
    if t in ("true", "1", "yes", "si", "sí", "verdadero"):
        return True
    if t in ("false", "0", "no", "falso"):
        return False
    return default


def _mm_or_none(v):
    """'15mm' -> '15mm' (nombre de tablaplaca); '0'/'0mm'/'' -> None."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s in ("0", "0mm", "0.0", "None"):
        return None
    return s if s.endswith("mm") else f"{int(round(_num(s)))}mm"


@lru_cache(maxsize=1024)
def _get_perfil(nombre: str, is_column: bool = False) -> Perfil:
    # En cache: consultar el catalogo (pandas) y construir el Perfil en cada comprobacion
    # era lo mas caro de preparar la entrada. Los perfiles no se modifican.
    geom = get_profile(nombre)
    if geom is None:
        raise ValueError(f"Perfil '{nombre}' no encontrado en los catalogos Tekla.")

    def g(col, default=0.0):
        v = geom.get(col, default)
        try:
            f = float(v)
            return 0.0 if f != f else f  # NaN -> 0
        except (TypeError, ValueError):
            return default

    h = g("d_mm")
    b = g("b_mm") or g("bf_mm")   # el catalogo Tekla usa b_mm (bf_mm solo como respaldo)
    tf = g("tf_mm")
    tw = g("tw_mm")
    r = g("r1_mm")
    # OJO: los catalogos Tekla traen A en cm2 y wply en cm3 (pese al sufijo _mm).
    A = g("A_mm") * 100.0       # cm2 -> mm2
    Z = g("wply_mm") * 1000.0   # cm3 -> mm3

    prmin = 0.0
    prmax = 0.0

    if is_column:
        prmin = g("prmin")
        prmax = g("prmax")

        # Si el catalogo no define prmin/prmax (o valen 0), se derivan de la geometria:
        # Prmin: espacio libre minimo entre tornillos dada la garganta del alma (tw + 2r + holguras)
        # Prmax: ancho del ala menos las distancias minimas al borde exterior
        if prmin <= 0:
            prmin = max(50.0, tw + 2.0 * r + 60.0)
        if prmax <= 0 or prmax <= prmin:
            prmax = max(prmin + 30.0, b - 2.0 * 30.0)

    # Geometria basica obligatoria: con alguna a 0 el calculador acaba dividiendo por cero.
    faltan = [n for n, v in (("d_mm", h), ("b_mm", b), ("tf_mm", tf), ("tw_mm", tw)) if v <= 0]
    if faltan:
        raise ValueError(f"Perfil '{nombre}': faltan datos geometricos en el catalogo Tekla ({', '.join(faltan)} = 0 o vacio).")

    return Perfil(
        nombre=str(nombre),
        h=h,
        b=b,
        tf=tf,
        tw=tw,
        r=r,
        A=A,
        Z=Z,
        Prmin=prmin,
        Prmax=prmax,
    )


def row_to_conexion_input(row: Dict[str, Any]) -> ConexionInput:
    """Fila CODAME -> ConexionInput (para el calculador AISC)."""
    beam_name = row.get("BEAM", "IPE300")
    col_name = row.get("COLUMN", "HEA200")
    viga = _get_perfil(beam_name, is_column=False)
    pilar = _get_perfil(col_name, is_column=True)

    fy = _num(row.get("fy"), ACERO_TABLE[DEFAULT_GRADO_ACERO]["fy_acero_mpa"])
    fu = _num(row.get("fu"), ACERO_TABLE[DEFAULT_GRADO_ACERO]["fu_acero_mpa"])

    bolt_raw = str(row.get("BoltName_ep") or row.get("Bolt") or "M24").strip().upper()
    bolt_name = bolt_raw if bolt_raw.startswith("M") else f"M{bolt_raw}"

    b_ep = _num(row.get("b_ep"), 200.0)
    p2_ep = _num(row.get("p2_ep"), 100.0)
    pfo = _num(row.get("pfo"), 50.0)
    dev = _num(row.get("dev"), 50.0)
    pfi = _num(row.get("pfi"), 30.0)

    if p2_ep <= 0:
        raise ValueError(f"p2_ep (gramil g) = {p2_ep:g}: debe ser mayor que 0 para comprobar la union.")
    if b_ep <= 0:
        raise ValueError(f"b_ep (ancho de chapa) = {b_ep:g}: debe ser mayor que 0 para comprobar la union.")
    pb = _num(row.get("pb"), 80.0)

    p3 = _num(row.get("p3")); p4 = _num(row.get("p4")); p5 = _num(row.get("p5"))
    p6 = _num(row.get("p6")); p7 = _num(row.get("p7")); p8 = _num(row.get("p8"))
    p9 = _num(row.get("p9")); p10 = _num(row.get("p10")); p11 = _num(row.get("p11"))
    p12 = _num(row.get("p12"))

    nbrtension = int(_num(row.get("ntensionrows"), 2))
    # Total de filas derivado de los pasos activos (pfi/pb/p3..p12) + 1.
    separations = [pfi, pb, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
    n_total_filas = sum(1 for x in separations if x > 0) + 1

    lh = _num(row.get("Lh"))

    # Axial del pilar: Pr_Porcen = % de su carga de plastificacion Py = A*fy (kN).
    pr = _num(row.get("Pr_Porcen")) / 100.0 * pilar.A * fy / 1000.0
    e_mod = _num(row.get("E"))
    if e_mod <= 0:
        e_mod = 200000.0
    weld_factor = _num(row.get("WeldFactor"), 1.0) or 1.0
    bolts_per_row = _num(row.get("BoltsInTensionFactor"), 2.0) or 2.0
    electrode = str(row.get("Electrode") or "").strip() or None

    # `row_to_conexion_input` ya sanea y convierte todos los campos; evitar la
    # validacion/coercion de Pydantic en cada una de las ~25 evaluaciones por fila.
    return ConexionInput.model_construct(
        viga=viga,
        pilar=pilar,
        E=e_mod,
        fy=fy,
        fu=fu,
        fyp=fy,
        fup=fu,
        BoltName_ep=bolt_name,
        BoltGrade=str(row.get("BoltGrade") or DEFAULT_CLASE_TORNILLO).strip(),
        BoltsInTensionFactor=bolts_per_row,
        nbrtension_ep_AME=nbrtension,
        NumeTotalFilasTornillos=n_total_filas,
        b_ep=b_ep,
        p2_ep=p2_ep,
        pfo=pfo,
        pfi=pfi,
        dev=dev,
        pb=pb,
        p1=None,
        p3=p3, p4=p4, p5=p5, p6=p6, p7=p7, p8=p8, p9=p9, p10=p10, p11=p11, p12=p12,
        p12_Outer=_num(row.get("p12_Outer")),
        ThicknessName_t_ep=_mm_or_none(row.get("ThicknessName_t_ep")) or "0mm",
        ThicknessName_t_tenstiff=_mm_or_none(row.get("ThicknessName_t_tenstiff")),
        ThicknessName_t_webstiff=_mm_or_none(row.get("ThicknessName_t_webstiff")),
        ThicknessName_t_BeamHcompstiff=_mm_or_none(row.get("ThicknessName_t_BeamHcompstiff")),
        AnguloCantonera=_num(row.get("AnguloCantonera")),
        FactorCantoCantonera=_num(row.get("FactorCantoCantonera"), 1.0),
        Lh=lh,
        ww=_num(row.get("ww")),
        wf=_num(row.get("wf")),
        wh=_num(row.get("wh")),
        wswc=_num(row.get("w3")),   # SUPUESTO: soldadura rigidizador-alma del pilar; confirmar w3/w4
        WeldFactor=weld_factor,
        Electrode=electrode,
        Pr=pr,
        MzEd_Porcen=_num(row.get("MzEd_Porcen")),
        FyEd_Porcen=_num(row.get("FyEd_Porcen")),
        FxEd_Porcen=_num(row.get("FxEd_Porcen")),
        MzEd_PanelZone=_num(row.get("MzEd_PanelZone")),
        TipoCombiEnve="LRFD",
        yieldmethod=_bool(row.get("yieldmethod"), True),
        Traccion=_bool(row.get("Traccion"), False),
        SisUniPerMetri=_bool(row.get("SisUniBoltSI"), True),
        SisUniPlaMetri=_bool(row.get("SisUniPlateSI"), True),
    )


def calcular_ratios_row(row: Dict[str, Any]) -> Tuple[Dict[str, float], CalculadorAISC36010]:
    """Via rapida: solo los ratios ({descripcion: valor}) y el calculador (para sus residuos)."""
    calc = CalculadorAISC36010(inp=row_to_conexion_input(row))
    return calc.calcular_ratios(), calc


def check_single_row(row: Dict[str, Any]) -> Dict[str, Any]:
    conexion_input = row_to_conexion_input(row)
    resultado = CalculadorAISC36010(inp=conexion_input).calcular_dict()

    res_dict = dict(row)
    res_dict["cumple"] = resultado["cumple"]
    res_dict["ratio_pesimo"] = round(resultado["ratio_pesimo"], 4)
    res_dict["descripcion_ratio_pesimo"] = resultado["descripcion_ratio_pesimo"]
    res_dict["ratios"] = resultado["ratios"]
    res_dict["complementarios"] = resultado["complementarios"]
    return res_dict


def _ratio_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _ratio_quote(descripcion: Any) -> str:
    text = str(descripcion or "").strip()
    return f"«{text}»" if text else "una comprobación sin descripción"


def _ratio_list(items: List[Dict[str, Any]], render) -> str:
    visibles = items[:3]
    texto = ", ".join(render(item) for item in visibles)
    if len(items) > len(visibles):
        texto += f" y {len(items) - len(visibles)} más"
    return texto


def resumen_comprobacion(check: Dict[str, Any]) -> Dict[str, Any]:
    """Resumen compacto y redactado de una comprobacion AISC para el informe MCP."""
    ratios: List[Dict[str, Any]] = []
    for item in check.get("ratios") or []:
        if not isinstance(item, dict):
            continue
        ratio = _ratio_number(item.get("ratio"))
        if ratio is None:
            continue
        ratios.append({
            "indice": item.get("indice"),
            "descripcion": str(item.get("descripcion") or "").strip(),
            "ratio": round(ratio, 4),
        })

    fallos_geometricos: List[Dict[str, Any]] = []
    for item in check.get("complementarios") or []:
        if not isinstance(item, dict):
            continue
        resultado = str(item.get("resultado") or "").strip()
        if resultado.upper().startswith("REVISAR"):
            fallos_geometricos.append({
                "indice": item.get("indice"),
                "descripcion": str(item.get("descripcion") or "").strip(),
                "resultado": resultado,
            })

    excedidos = sorted(
        [item for item in ratios if item["ratio"] > 1.0 + 1e-9],
        key=lambda item: item["ratio"],
        reverse=True,
    )

    if ratios:
        peor_ratio = max(ratios, key=lambda item: item["ratio"])
        ratio_max_todos = peor_ratio["ratio"]
        descripcion_ratio_max_todos = peor_ratio["descripcion"]
    else:
        ratio_max_todos = _ratio_number(check.get("ratio_max_todos"))
        if ratio_max_todos is not None:
            ratio_max_todos = round(ratio_max_todos, 4)
        descripcion_ratio_max_todos = str(
            check.get("descripcion_ratio_max_todos") or ""
        ).strip()

    ratio_pesimo = _ratio_number(check.get("ratio_pesimo"))
    if ratio_pesimo is not None:
        ratio_pesimo = round(ratio_pesimo, 4)
    descripcion_ratio_pesimo = str(check.get("descripcion_ratio_pesimo") or "").strip()
    cumple = bool(check.get("cumple"))

    if cumple:
        estado = "CUMPLE"
    elif not ratios and not fallos_geometricos:
        if "cumple" in check and descripcion_ratio_pesimo:
            estado = "NO_CUMPLE_OTRO"
        else:
            estado = "SIN_DATOS"
    elif excedidos and fallos_geometricos:
        estado = "NO_CUMPLE_AMBOS"
    elif excedidos:
        estado = "NO_CUMPLE_RESISTENCIA"
    elif fallos_geometricos:
        estado = "NO_CUMPLE_GEOMETRIA"
    else:
        estado = "NO_CUMPLE_OTRO"

    if estado == "SIN_DATOS":
        detalle = descripcion_ratio_pesimo or "sin datos de comprobación"
        dictamen = f"Sin datos de comprobación AISC: {detalle}."
    elif estado == "CUMPLE":
        partes: List[str] = []
        if ratio_pesimo is not None:
            partes.append(
                f"ratio pésimo {ratio_pesimo:.2f} en "
                f"{_ratio_quote(descripcion_ratio_pesimo or descripcion_ratio_max_todos)}"
            )
        if (
            ratio_max_todos is not None
            and ratio_pesimo is not None
            and abs(ratio_max_todos - ratio_pesimo) > 1e-9
        ):
            partes.append(
                f"máximo global {ratio_max_todos:.2f} en "
                f"{_ratio_quote(descripcion_ratio_max_todos)}"
            )
        dictamen = "Cumple AISC 360-10"
        if partes:
            dictamen += ": " + "; ".join(partes)
        dictamen += "."
    elif estado == "NO_CUMPLE_OTRO":
        dictamen = (
            "No cumple AISC 360-10: "
            f"{descripcion_ratio_pesimo or 'comprobación no conforme'}."
        )
    else:
        partes = []
        if excedidos:
            partes.append(
                "supera 1.00: "
                + _ratio_list(
                    excedidos,
                    lambda item: f"{_ratio_quote(item['descripcion'])} = {item['ratio']:.2f}",
                )
            )
        if fallos_geometricos:
            partes.append(
                "geometría: "
                + _ratio_list(
                    fallos_geometricos,
                    lambda item: f"{item['descripcion']} ({item['resultado']})",
                )
            )
        dictamen = "No cumple AISC 360-10: " + "; ".join(partes) + "."
        if (
            ratio_max_todos is not None
            and ratio_pesimo is not None
            and abs(ratio_max_todos - ratio_pesimo) > 1e-9
            and not any(abs(item["ratio"] - ratio_max_todos) < 1e-9 for item in excedidos)
        ):
            dictamen += (
                f" El ratio pésimo clásico es {ratio_pesimo:.2f} en "
                f"{_ratio_quote(descripcion_ratio_pesimo)}; el máximo global es "
                f"{ratio_max_todos:.2f} en {_ratio_quote(descripcion_ratio_max_todos)}."
            )

    return {
        "cumple": cumple,
        "estado": estado,
        "ratio_pesimo": ratio_pesimo,
        "descripcion_ratio_pesimo": descripcion_ratio_pesimo,
        "ratio_max_todos": ratio_max_todos,
        "descripcion_ratio_max_todos": descripcion_ratio_max_todos,
        "ratios_superan_1": excedidos,
        "fallos_geometricos": fallos_geometricos,
        "n_ratios_superan_1": len(excedidos),
        "n_fallos_geometricos": len(fallos_geometricos),
        "dictamen": dictamen,
    }


def _estado_complementario(resultado: str) -> str:
    """Clasifica una comprobacion complementaria: REVISAR, OK o INFO (dato)."""
    texto = str(resultado or "").strip().upper()
    if texto.startswith("REVISAR"):
        return "REVISAR"
    if texto == "OK" or texto.startswith("OK "):
        return "OK"
    return "INFO"


def detalle_comprobacion(check: Dict[str, Any]) -> Dict[str, Any]:
    """Comprobacion completa para la ficha MCP: todos los ratios y complementarios.

    Cada ratio lleva `valor`, `limite` (1.0) y `estado` (OK/EXCEDE/SIN_DATO). Cada
    comprobacion complementaria lleva `valor`, `limite` (None: el umbral va en el texto)
    y `estado` (OK/REVISAR/INFO).
    """
    detalle = resumen_comprobacion(check)

    ratios: List[Dict[str, Any]] = []
    for item in check.get("ratios") or []:
        if not isinstance(item, dict):
            continue
        ratio = _ratio_number(item.get("ratio"))
        if ratio is None:
            estado = "SIN_DATO"
            valor = None
        else:
            estado = "EXCEDE" if ratio > 1.0 + 1e-9 else "OK"
            valor = round(ratio, 4)
        ratios.append({
            "indice": item.get("indice"),
            "descripcion": str(item.get("descripcion") or "").strip(),
            "valor": valor,
            "limite": 1.0,
            "estado": estado,
        })

    complementarios: List[Dict[str, Any]] = []
    for item in check.get("complementarios") or []:
        if not isinstance(item, dict):
            continue
        resultado = str(item.get("resultado") or "").strip()
        complementarios.append({
            "indice": item.get("indice"),
            "descripcion": str(item.get("descripcion") or "").strip(),
            "valor": resultado,
            "limite": None,
            "estado": _estado_complementario(resultado),
        })

    detalle["ratios"] = ratios
    detalle["complementarios"] = complementarios
    return detalle


def check_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [check_single_row(r) for r in rows]
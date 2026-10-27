"""Esquema canonico de resultados CODAME (`Tabla_MOMENT_M1`).

Los resultados del servidor son SIEMPRE estas columnas, independientemente del modelo.
Se omite `Estado_IA` (no se usa en este flujo). Cada modelo las produce con su `to_codame()`;
el servidor solo reordena y, como red de seguridad, rellena con `CODAME_DEFAULTS` las que
falten (nunca falla por culpa del modelo). Referencia: `doc/codame/mapeo_columnas_codame.md`.
"""

# Orden de las columnas de Tabla_MOMENT_M1.
CODAME_COLUMNS = [
    "Unicode",
    "Normativas",
    "BEAM",
    "COLUMN",
    "MzEd_Porcen",
    "FyEd_Porcen",
    "FxEd_Porcen",
    "Pr_Porcen",
    "Traccion",
    "MzEd_PanelZone",
    "Vgra",
    "Havg",
    "BoltName_ep",
    "p2_ep",
    "ThicknessName_t_webstiff",
    "Bws",
    "Lws",
    "b_ep",
    "Lp",
    "ThicknessName_t_ep",
    "Lh",
    "AnguloCantonera",
    "dev",
    "p12_Outer",
    "pfo",
    "pfi",
    "pb",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "p9",
    "p10",
    "p11",
    "p12",
    "ThicknessName_t_compstiff",
    "ThicknessName_t_BeamHcompstiff",
    "ThicknessName_t_tenstiff",
    "wf",
    "ww",
    "w3",
    "w4",
    "wh",
    "FactorCantoCantonera",
    "ntensionrows",
    "fy",
    "fu",
    "fyp",
    "fup",
    "E",
    "BoltGrade",
    "BoltThreads",
    "Electrode",
    "yieldmethod",
    "WeldFactor",
    "BoltsInTensionFactor",
    "SisUniBoltSI",
    "SisUniPlateSI",
    "Source",
    "Designation",
]

# Valor simbolico por defecto (red de seguridad / columnas que el cliente no rellena).
CODAME_DEFAULTS = {c: 0.0 for c in CODAME_COLUMNS}
CODAME_DEFAULTS.update({
    # Valores por defecto de la tabla (Cambios_a_aplicar.txt)
    "wf": 6.0, "ww": 6.0, "w3": 6.0, "w4": 6.0, "wh": 6.0,
    "Pr_Porcen": 50,
    "Traccion": True,
    "E": 200000.0,
    "Electrode": "E49XX",
    "SisUniBoltSI": True,
    "SisUniPlateSI": True,
    "Designation": "Tekla19",
    "Unicode": "",
    "Normativas": "",
    "BEAM": "",
    "COLUMN": "",
    "BoltName_ep": "",
    "ThicknessName_t_webstiff": "0mm",
    "ThicknessName_t_ep": "0mm",
    "ThicknessName_t_compstiff": "0mm",
    "ThicknessName_t_BeamHcompstiff": "0mm",
    "ThicknessName_t_tenstiff": "0mm",
    "ntensionrows": 2,
    "BoltGrade": "",
    "BoltThreads": True,
    "yieldmethod": True,
    "WeldFactor": 1,
    "BoltsInTensionFactor": 2,
    "Source": "",
})


def fill_missing(row: dict) -> dict:
    """Rellena con los defaults las columnas que falten, en el orden canonico."""
    return {c: row.get(c, CODAME_DEFAULTS[c]) for c in CODAME_COLUMNS}
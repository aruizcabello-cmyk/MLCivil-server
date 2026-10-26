RAW_INPUT_COLUMNS = [
    "UNICODE",
    "BEAM",
    "COLUMN",
    "ratio_moment",
    "ratio_shear",
    "ratio_axial",
    "grado_acero",
    "clase_tornillo",
]
CATEGORICAL_FEATURES = ["grado_acero", "clase_tornillo"]
NUMERIC_FEATURES = [
    "BEAM_d_mm",
    "COLUMN_d_mm",
    "fy_acero_mpa",
    "fu_acero_mpa",
    "fy_tornillo_mpa",
    "fu_tornillo_mpa",
    "fv_tornillo_mpa",
    "ratio_moment",
    "ratio_shear",
    "ratio_axial",
]
TARGET_CLS_N_ALMA = "n_alma"
TARGET_CLS_N_CANTONERA = "n_cantonera"
N_Y = 12
P_COLUMNS = [f"p{i}" for i in range(1, N_Y + 1)]
P_TRANSITION_COLUMNS = [f"p{i}_transition" for i in range(1, N_Y + 1)]
N_ALMA_CLASSES = 7
N_CANTONERA_CLASSES = 6
REGRESSION_ALWAYS = ["g", "e", "d", "Bp", "tp"]
STIFFENER_COLUMNS = ["tws", "ts", "tbs", "tst"]
MASKED_COLUMNS = P_COLUMNS + STIFFENER_COLUMNS
CATALOGO_BOLT = [16, 20, 22, 24, 27, 30, 36]
CATALOGO_g = list(range(80, 171, 5))
CATALOGO_e = list(range(35, 76, 5))
CATALOGO_d = None
CATALOGO_Bp = list(range(140, 400))
CATALOGO_tp = [12, 15, 20, 25, 30, 35, 40, 45, 50, 60]
CATALOGO_tws = [0, 6, 8, 10, 12, 15, 20]
CATALOGO_ts = [0, 10, 12, 15, 20, 25, 30, 35, 40]
CATALOGO_tbs = CATALOGO_ts
CATALOGO_tst = [0, 10, 12, 15, 20, 25, 30, 35, 40]
Y_GRID_STEP = 5
UNUSED_TRANSITION = "UNUSED"


def etiquetas_zona(z, n):
    n = int(n)
    if n <= 0:
        return []
    if n == 1:
        return [f"{z} centro"]
    if n == 2:
        return [f"{z} primero", f"{z} inferior"]
    out = [f"{z} primero"]
    for i in range(1, n - 1):
        r = i / (n - 1)
        out.append(
            f"{z} "
            + (
                "centro superior"
                if r < 0.5
                else "centro"
                if r == 0.5
                else "centro inferior"
            )
        )
    return out + [f"{z} inferior"]


def secuencia_zonas_tornillos(na, nc):
    return (
        ["superior"]
        + etiquetas_zona("viga", na)
        + etiquetas_zona("cantonera", nc)
        + ["inferior"]
    )


def secuencia_transiciones(na, nc):
    z = secuencia_zonas_tornillos(na, nc)
    return [f"{a}__{b}" for a, b in zip(z[:-1], z[1:])]


def crear_catalogo_transiciones():
    s = {UNUSED_TRANSITION}
    for a in range(N_ALMA_CLASSES):
        for c in range(N_CANTONERA_CLASSES):
            s.update(secuencia_transiciones(a, c))
    return [UNUSED_TRANSITION] + sorted(x for x in s if x != UNUSED_TRANSITION)


P_TRANSITION_CLASSES = crear_catalogo_transiciones()
P_TRANSITION_TO_IDX = {x: i for i, x in enumerate(P_TRANSITION_CLASSES)}
P_TRANSITION_UNUSED_IDX = 0
N_P_TRANSITION_CLASSES = len(P_TRANSITION_CLASSES)

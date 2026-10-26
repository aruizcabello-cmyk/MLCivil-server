import math, os, threading
import pandas as pd
import config, codame
from catalogo import ACERO_TABLE, get_profile

MASTER = config.MODELS_DIR / "V11" / "data" / "datos_modelo.parquet"
LOCK = threading.Lock()
TORN = {"8.8": (640., 800., 480.), "10.9": (900., 1000., 500.), "A325M": (660., 830., 372.)}
NORMAS = {"EN 1993-1-8", "AISC-360-05/10", "CSA S16-09"}
BOLTS = {16, 20, 22, 24, 27, 30, 36}
COLS = (["UNICODE", "BEAM", "COLUMN", "ratio_moment", "ratio_shear", "ratio_axial", "grado_acero",
         "clase_tornillo", "BEAM_type", "COLUMN_type", "BEAM_d_mm", "COLUMN_d_mm", "standard", "Bolt",
         "SlipCritical", "n_alma", "n_cantonera", "fy_acero_mpa", "fu_acero_mpa", "fy_tornillo_mpa",
         "fu_tornillo_mpa", "fv_tornillo_mpa", "g", "e", "d", "Bp", "tp", "tws", "ts", "tbs", "tst"]
        + [f"p{i}" for i in range(1, 13)] + [f"p{i}_transition" for i in range(1, 13)]
        + ["geometria_dudosa"])


def _n(v, d=0.0):
    try:
        f = float(str(v).lower().replace("mm", "").replace(",", ".").strip())
        return d if math.isnan(f) else f
    except (TypeError, ValueError):
        return d


def _z(z, n):
    if n <= 0: return []
    if n == 1: return [f"{z} centro"]
    if n == 2: return [f"{z} primero", f"{z} inferior"]
    o = [f"{z} primero"]
    for i in range(1, n - 1):
        r = i / (n - 1)
        o.append(f"{z} " + ("centro superior" if r < .5 else "centro" if r == .5 else "centro inferior"))
    return o + [f"{z} inferior"]


def _trans(na, nc):
    z = ["superior"] + _z("viga", na) + _z("cantonera", nc) + ["inferior"]
    return [f"{a}__{b}" for a, b in zip(z, z[1:])]


def to_row(r):
    uid = str(r.get("Unicode") or r.get("UNICODE") or "").strip()
    if not uid: raise ValueError("sin Unicode")
    beam, col = str(r.get("BEAM", "")).strip(), str(r.get("COLUMN", "")).strip()
    pb, pc = get_profile(beam), get_profile(col)
    if pb is None or pc is None: raise ValueError("perfil fuera de catálogo")
    h, tf = _n(pb.get("d_mm")), _n(pb.get("tf_mm"))
    e, pfo, pfi = _n(r.get("dev")), _n(r.get("pfo")), _n(r.get("pfi"))
    if min(h, tf, e, pfo, pfi) <= 0: raise ValueError("geometría incompleta")
    c = codame._redond5(pfo + pfi + tf)
    seps = [_n(r.get(k)) for k in ("pb", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12")]
    k = 0
    while k < len(seps) and seps[k] > 0: k += 1
    if any(s > 0 for s in seps[k:]): raise ValueError("separaciones no consecutivas")
    seps = seps[:k]
    ys = [e, e + c]
    for s in seps: ys.append(ys[-1] + s)
    top = e + pfo; bot = top + h
    na = sum(1 for y in ys[1:] if top < y < bot)
    nc = len(ys) - 2 - na
    if not (1 <= na <= 6 and 0 <= nc <= 5): raise ValueError(f"topología fuera de rango (na={na}, nc={nc})")
    if (_n(r.get("Lh")) > 0) != (nc > 0):
        raise ValueError(f"cantonera/filas incoherentes Lh={r.get('Lh')!r} na={na} nc={nc} "
                         f"ys={[round(y) for y in ys]} top={top:.0f} bot={bot:.0f} h={h} tf={tf}")
    p = [c] + seps
    if any(x % 5 for x in p): raise ValueError("p fuera de rejilla 5 mm")
    p += [0.0] * (12 - len(p))
    fy = _n(r.get("fy"))
    grado = next((g for g, v in ACERO_TABLE.items() if abs(v["fy_acero_mpa"] - fy) < 1e-6), None)
    if not grado: raise ValueError(f"fy={fy} sin grado")
    clase = str(r.get("BoltGrade", "")).strip()
    if clase not in TORN: raise ValueError(f"clase tornillo {clase}")
    bolt = int(_n("".join(ch for ch in str(r.get("BoltName_ep", "")) if ch.isdigit())))
    if bolt not in BOLTS: raise ValueError(f"tornillo {bolt}")
    norma = str(r.get("Normativas", "")).strip()
    if norma not in NORMAS: raise ValueError(f"normativa '{norma}'")
    tr = _trans(na, nc)
    o = {"UNICODE": uid, "BEAM": beam, "COLUMN": col,
         "ratio_moment": _n(r.get("MzEd_Porcen")) / 100, "ratio_shear": _n(r.get("FyEd_Porcen")) / 100,
         "ratio_axial": _n(r.get("FxEd_Porcen")) / 100, "grado_acero": grado, "clase_tornillo": clase,
         "BEAM_type": str(pb.get("type")), "COLUMN_type": str(pc.get("type")),
         "BEAM_d_mm": h, "COLUMN_d_mm": _n(pc.get("d_mm")), "standard": norma, "Bolt": str(bolt),
         "SlipCritical": str(r.get("SlipCritical") or "NO").upper().replace("TRUE", "YES").replace("FALSE", "NO"),
         "n_alma": na, "n_cantonera": nc, "fy_acero_mpa": fy, "fu_acero_mpa": ACERO_TABLE[grado]["fu_acero_mpa"],
         "fy_tornillo_mpa": TORN[clase][0], "fu_tornillo_mpa": TORN[clase][1], "fv_tornillo_mpa": TORN[clase][2],
         "g": _n(r.get("p2_ep")), "e": e, "d": pfo, "Bp": _n(r.get("b_ep")), "tp": _n(r.get("ThicknessName_t_ep")),
         "tws": _n(r.get("ThicknessName_t_webstiff")), "ts": _n(r.get("ThicknessName_t_compstiff")),
         "tbs": _n(r.get("ThicknessName_t_BeamHcompstiff")), "tst": _n(r.get("ThicknessName_t_tenstiff")),
         "geometria_dudosa": False}
    for i in range(12):
        o[f"p{i+1}"] = float(p[i])
        o[f"p{i+1}_transition"] = tr[i] if i < len(tr) else "UNUSED"
    return o


def append(rows):
    ok, rej = [], []
    for i, r in enumerate(rows, 1):
        try: ok.append(to_row(r))
        except Exception as ex: rej.append(f"fila {i} ({r.get('Unicode')}): {ex}")
    if not ok:
        return {"added": 0, "updated": 0, "rejected": rej, "total_rows": 0}
    new = pd.DataFrame(ok)
    with LOCK:
        cur = pd.read_parquet(MASTER) if MASTER.exists() else pd.DataFrame(columns=COLS)
        cols = list(cur.columns) if len(cur) else COLS
        new = new.reindex(columns=cols)
        ex_ids = set(cur["UNICODE"].astype(str))
        ids = set(new["UNICODE"])
        comb = pd.concat([cur, new], ignore_index=True).drop_duplicates("UNICODE", keep="last")
        tmp = MASTER.with_suffix(".tmp")
        comb.to_parquet(tmp, index=False)
        os.replace(tmp, MASTER)
    return {"added": len(ids - ex_ids), "updated": len(ids & ex_ids), "rejected": rej, "total_rows": len(comb)}
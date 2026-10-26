"""
Dependencias EXTERNAS de M_C_AISC36010.cs.
Implementación completa de las tablas y métodos auxiliares para AISC 360-10.
Unidades: mm, MPa, kN, kN*m.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


class tablapernos:
    """Tabla de tornillos según AISC 360-10."""

    DATOS_PERNOS = {
        16: {"Le_min": 22.0, "pitch_min": 50.0, "clearance": 30.0, "Tb": 91.0, "dh": 18.0},
        20: {"Le_min": 28.0, "pitch_min": 60.0, "clearance": 35.0, "Tb": 142.0, "dh": 22.0},
        22: {"Le_min": 30.0, "pitch_min": 65.0, "clearance": 40.0, "Tb": 176.0, "dh": 24.0},
        24: {"Le_min": 34.0, "pitch_min": 70.0, "clearance": 45.0, "Tb": 205.0, "dh": 26.0},
        27: {"Le_min": 38.0, "pitch_min": 80.0, "clearance": 50.0, "Tb": 267.0, "dh": 30.0},
        30: {"Le_min": 44.0, "pitch_min": 85.0, "clearance": 55.0, "Tb": 326.0, "dh": 33.0},
        36: {"Le_min": 52.0, "pitch_min": 100.0, "clearance": 65.0, "Tb": 475.0, "dh": 39.0},
    }

    RESISTENCIA_GRADO = {
        "8.8": {"Fnt": 600.0, "Fnv": 375.0},
        "10.9": {"Fnt": 830.0, "Fnv": 500.0},
        "A325": {"Fnt": 620.0, "Fnv": 372.0},
        "A325M": {"Fnt": 620.0, "Fnv": 372.0},
        "A490": {"Fnt": 780.0, "Fnv": 457.0},
        "A490M": {"Fnt": 780.0, "Fnv": 457.0},
    }

    def __init__(self) -> None:
        self.lambdaM0AME: str = "0.75"
        self.nombretornillo: str = "M24"
        self.threadscondition: Any = None
        self.gradotornillo: str = "10.9"

    @property
    def dmetrico(self) -> float:
        s = str(self.nombretornillo).strip().upper().replace("M", "")
        try:
            return float(s)
        except ValueError:
            return 24.0

    @property
    def dhmetrico(self) -> float:
        d = int(self.dmetrico)
        return float(self.DATOS_PERNOS.get(d, {}).get("dh", d + 2.0))

    @property
    def Area_b(self) -> float:
        return math.pi * (self.dmetrico ** 2) / 4.0

    @property
    def FntmetAME(self) -> float:
        grado = str(self.gradotornillo).strip().upper()
        fnt_mpa = self.RESISTENCIA_GRADO.get(grado, {}).get("Fnt", 620.0)
        return (fnt_mpa * self.Area_b) / 1000.0

    @property
    def FnvmetAME(self) -> float:
        grado = str(self.gradotornillo).strip().upper()
        fnv_mpa = self.RESISTENCIA_GRADO.get(grado, {}).get("Fnv", 372.0)
        return (fnv_mpa * self.Area_b) / 1000.0

    @property
    def FntimpAME(self) -> float:
        return self.FntmetAME

    @property
    def FnvimpAME(self) -> float:
        return self.FnvmetAME

    @property
    def dimperial(self) -> float:
        return self.dmetrico / 25.4

    @property
    def dhimperial(self) -> float:
        return self.dhmetrico / 25.4

    @property
    def disborminmet(self) -> float:
        d = int(self.dmetrico)
        return float(self.DATOS_PERNOS.get(d, {}).get("Le_min", 1.25 * d))

    @property
    def disborminimp(self) -> float:
        return self.disborminmet / 25.4

    @property
    def pitchminmet(self) -> float:
        d = int(self.dmetrico)
        return float(self.DATOS_PERNOS.get(d, {}).get("pitch_min", 2.67 * d))

    @property
    def pitchminimp(self) -> float:
        return self.pitchminmet / 25.4

    @property
    def dislibmet(self) -> float:
        d = int(self.dmetrico)
        return float(self.DATOS_PERNOS.get(d, {}).get("clearance", 35.0))

    @property
    def Tb_Me(self) -> float:
        d = int(self.dmetrico)
        return float(self.DATOS_PERNOS.get(d, {}).get("Tb", 205.0))

    @property
    def pitchmin(self) -> float:
        return self.pitchminmet

    def NombreTornilloAnterior(self, actual: str) -> str:
        catalogo = [16, 20, 22, 24, 27, 30, 36]
        d = int(self.dmetrico)
        if d in catalogo:
            idx = catalogo.index(d)
            if idx > 0:
                return f"M{catalogo[idx-1]}"
        return "M16"


class tablaplaca:
    """Tabla de espesores de chapa."""

    def __init__(self) -> None:
        self.nombreplaca: Optional[str] = None

    @property
    def tmetrico(self) -> float:
        if not self.nombreplaca or str(self.nombreplaca).strip() in ("0", "None", ""):
            return 0.0
        s = str(self.nombreplaca).replace("mm", "").replace("t", "").strip()
        try:
            return float(s)
        except ValueError:
            return 0.0

    @property
    def timperial(self) -> float:
        return self.tmetrico / 25.4

    @property
    def disbormaxmet(self) -> float:
        t = self.tmetrico
        return min(12.0 * t, 150.0) if t > 0 else 150.0

    @property
    def pitchmaxmet(self) -> float:
        t = self.tmetrico
        return min(24.0 * t, 305.0) if t > 0 else 305.0

    @property
    def disbormaximp(self) -> float:
        return self.disbormaxmet / 25.4

    @property
    def pitchmaximp(self) -> float:
        return self.pitchmaxmet / 25.4


class SlipCritical:
    def Rn(self, EsStandardHole, EsOverSizeHole, EsLongSlottedHole, EsOneFiller,
           EsClassASurface, EsClassBSurface, EsValornuTest, EsValorDuTest,
           nu, Du, Tb, Tu, BoltsInTension, ns) -> float:
        mu = 0.50 if EsClassBSurface else 0.30
        if EsValornuTest and nu:
            mu = float(nu)

        du_val = 1.13
        if EsValorDuTest and Du:
            du_val = float(Du)

        hf = 1.0 if EsOneFiller else 0.85
        ks = 0.85 if EsOverSizeHole else (0.70 if EsLongSlottedHole else 1.0)

        k_sc = 1.0
        if Tu > 0 and Tb > 0:
            denom = du_val * Tb * (BoltsInTension if BoltsInTension > 0 else 1.0)
            k_sc = max(0.0, 1.0 - (Tu / denom))

        rn = mu * du_val * hf * ks * Tb * (ns if ns else 1.0) * k_sc
        return float(rn)


class moduloauxiliar:
    @staticmethod
    def minimo2(a: float, b: float) -> float:
        return a if a < b else b

    @staticmethod
    def redond_mult(valor: float, multiplo: float) -> float:
        if multiplo == 0:
            return valor
        return round(valor / multiplo) * multiplo

    @staticmethod
    def fuweldtabla(electrodo: Any) -> float:
        if not electrodo or str(electrodo).strip() in ("0", "None", ""):
            return 485.0
        s = str(electrodo).upper().strip()
        if "E80" in s:
            return 550.0
        if "E70" in s:
            return 485.0
        try:
            return float(s)
        except ValueError:
            return 485.0

    @staticmethod
    def momentores_AME(Z_b: float, fy: float) -> float:
        return (Z_b * fy) / 1000.0

    @staticmethod
    def ResisCortanteEjeFuerte_AME(h_b: float, tw_b: float, fy: float) -> float:
        return (0.6 * fy * h_b * tw_b) / 1000.0

    @staticmethod
    def traccionres_AME(A_b: float, fy: float) -> float:
        return (A_b * fy) / 1000.0

    @staticmethod
    def ResisCortanteEjeDebil_AME(b_b: float, tf_b: float, fy: float) -> float:
        return (0.6 * fy * 2.0 * b_b * tf_b) / 1000.0

    @staticmethod
    def Espaciamiento(array: Sequence[float], EsMinimo: bool, valor: float, ref_proximas: Any = None) -> Any:
        valid = [x for x in array if x > 0]
        incumplimientos = []
        for i in range(len(valid) - 1):
            dist = valid[i+1] - valid[i]
            if EsMinimo and dist < valor:
                incumplimientos.append((i + 1, i + 2))
            elif not EsMinimo and dist > valor:
                incumplimientos.append((i + 1, i + 2))
        return incumplimientos if incumplimientos else None

    @staticmethod
    def distanciaObstaculo(array: Sequence[float], pos_obstaculo: float, ref_dist: Any = None, ref_tornillo: Any = None, BuscarFilaPosterior: bool = False) -> Tuple[float, int]:
        dist_min = 1e9
        tornillo_cercano = 1
        for i, pos in enumerate(array):
            if pos <= 0:
                continue
            dist = abs(pos - pos_obstaculo)
            if dist < dist_min:
                dist_min = dist
                tornillo_cercano = i + 1
        return dist_min, tornillo_cercano

    @staticmethod
    def PorcentajeDistanciaMinima(dist: float, minimo: float) -> float:
        if minimo == 0:
            return 100.0
        return (dist / minimo) * 100.0


@dataclass
class Externos:
    tablapernos: type = tablapernos
    tablaplaca: type = tablaplaca
    moduloauxiliar: Any = moduloauxiliar
    slip_critical: Any = SlipCritical
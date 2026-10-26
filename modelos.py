"""Modelos de entrada/salida. Los nombres de campo son los de las propiedades del C#
(para poder cruzarlos con M_C_AISC36010.cs). Unidades: mm, MPa, kN, kN*m."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Perfil(BaseModel):
    """Lo que devuelve la BD de perfiles (ya con los factores de unidades aplicados)."""
    nombre: str = ""
    h: float = Field(..., description="CANTO  (h_b / h_c)")
    b: float = Field(..., description="ANCHO  (b_b / b_c)")
    tf: float = Field(..., description="ESPESORALA")
    tw: float = Field(..., description="ESPESORALMA")
    r: float = Field(..., description="RADIOACUERDO")
    A: float = Field(..., description="AREA")
    Z: float = Field(..., description="MODURESISPLATICO")
    Prmin: float = 0.0   # solo columnas
    Prmax: float = 0.0   # solo columnas


class ConexionInput(BaseModel):
    # ---- perfiles: o bien nombre (se consulta la BD) o bien datos directos ----
    nombre_viga: Optional[str] = None
    nombre_pilar: Optional[str] = None
    viga: Optional[Perfil] = None
    pilar: Optional[Perfil] = None
    NombreFuente: str = ""
    ProntuarioNombrePerfiles: str = ""

    # ---- materiales ----
    E: float = 200000.0
    fy: float
    fu: float
    fyp: float
    fup: float
    fyL: Optional[float] = None

    # ---- tornillos ----
    BoltName_ep: str                       # "M24"
    BoltGrade: str                         # "A325", "10.9", ...
    BoltThreads: Any = None
    BoltsInTensionFactor: float = 2.0      # tornillos por fila
    nbrtension_ep_AME: int                 # filas traccionadas (2 o 3 en tus patrones)
    NumeTotalFilasTornillos: Optional[int] = None   # == num_filas_tornillos de tu modelo

    # ---- geometria de chapa y tornillos ----
    b_ep: float                            # Bp
    p2_ep: float                           # g
    pfo: float                             # d
    pfi: Optional[float] = None            # si no viene: p1 - tf_b - pfo
    dev: float                             # e
    pb: float                              # p2
    p1: Optional[float] = None             # solo para derivar pfi
    p3: float = 0.0
    p4: float = 0.0
    p5: float = 0.0
    p6: float = 0.0
    p7: float = 0.0
    p8: float = 0.0
    p9: float = 0.0
    p10: float = 0.0
    p11: float = 0.0
    p12: float = 0.0
    p12_Outer: float = 0.0

    # ---- espesores (NOMBRES de tablaplaca; None/"" = no existe) ----
    ThicknessName_t_ep: str
    ThicknessName_t_tenstiff: Optional[str] = None
    ThicknessName_t_tenstiff_p23: Optional[str] = None
    ThicknessName_t_webstiff: Optional[str] = None
    ThicknessName_t_BeamHcompstiff: Optional[str] = None
    ThicknessName_t_OuterStiff: Optional[str] = None   # no se usa en el C#

    # ---- cartela ----
    AnguloCantonera: float = 0.0
    FactorCantoCantonera: float = 1.0

    # ---- soldaduras ----
    ww: float = 0.0
    wf: float = 0.0
    wh: float = 0.0
    wswc: float = 0.0
    WeldFactor: float = 1.0
    Electrode: Any = None
    Fillet_Weld_Type: bool = False

    # ---- solicitaciones (absolutas, o en % si se rellenan *_Porcen) ----
    MzEd: float = 0.0
    FyEd: float = 0.0
    FxEd: float = 0.0
    FzEd: float = 0.0
    MzEd_Porcen: Optional[float] = None
    FyEd_Porcen: Optional[float] = None
    FxEd_Porcen: Optional[float] = None
    FzEd_Porcen: Optional[float] = None
    MzEd_PanelZone: float = 0.0
    Pr: float = 0.0
    Vgra: float = 0.0
    Havg: float = 1.0
    Lh: float = 1.0
    OMF_SMF: int = 0

    # ---- opciones ----
    TipoCombiEnve: str = "LRFD"
    ConnectionTop: str = "NO"
    yieldmethod: bool = True
    Traccion: bool = False
    SisUniPerMetri: bool = True
    SisUniPlaMetri: bool = True
    SeismicDoublerPlate: bool = False

    # ---- slip-critical ----
    EsStandardHole: bool = True
    EsOverSizedHole: bool = False
    EsLongSlottedHole: bool = False
    EsOneFiller: bool = True
    EsClassASurface: bool = True
    EsClassBSurface: bool = False
    EsValornuTest: bool = False
    EsValorDuTest: bool = False
    nu: float = 0.3
    Du: float = 1.13
    ns: float = 1.0

    # ---- opcional: contraste con tu modelo ----
    Hole_modelo: Optional[float] = None

    @field_validator("TipoCombiEnve", "ConnectionTop")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class FilaRatio(BaseModel):
    indice: int
    descripcion: str
    ratio: Optional[float]      # None si NaN/inf (ver `avisos`)


class Complementario(BaseModel):
    indice: int
    descripcion: str
    resultado: str


class ResultadoConexion(BaseModel):
    ratios: List[FilaRatio]
    ratio_pesimo: float
    descripcion_ratio_pesimo: str
    ratio_max_todos: Optional[float] = None    # incluye 15, 16, 17 (el C# los ignora en RatioPesimo)
    complementarios: List[Complementario]
    intermedios: Dict[str, Any]
    avisos: List[str]
    cumple: bool                               # max(ratios) <= 1 (criterio habitual, no esta en el C#)

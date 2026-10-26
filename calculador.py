"""
Motor de cálculo para uniones atornilladas con chapa frontal (AISC 360-10).
Traducción directa de M_C_AISC36010.cs a Python.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Any

from modelos import ConexionInput, ResultadoConexion, FilaRatio, Complementario, Perfil
from externos import Externos, tablapernos, tablaplaca
from perfiles import RepositorioPerfiles

log = logging.getLogger("ai-server")

# Externos solo contiene referencias a clases/estaticos: es seguro compartirlo.
_DEFAULT_EXT = Externos()


class CalculadorAISC36010:
    solo_ratios: bool = False
    solo_dict: bool = False
    ratios_dict: Dict[str, float] = {}
    resultado_dict: Dict[str, Any] = {}

    def __init__(self, inp: ConexionInput, ext: Optional[Externos] = None, repo_perfiles: Optional[RepositorioPerfiles] = None) -> None:
        self.inp = inp
        self.ext = ext or _DEFAULT_EXT
        self.repo_perfiles = repo_perfiles

        # Variables internas correspondientes a M_C_AISC36010.cs
        self.lambdaM0AME: float = 0.75
        self.NumeRatios: int = 18
        self.NumeResulComplemen: int = 33

        # Datos de perfiles
        self.viga: Optional[Perfil] = inp.viga
        self.pilar: Optional[Perfil] = inp.pilar

        # Geometría y propiedades iniciales
        self.E = inp.E
        self.fy = inp.fy
        self.fu = inp.fu
        self.fyp = inp.fyp
        self.fup = inp.fup
        self.fyL = inp.fyL if inp.fyL is not None else inp.fy

        # Pernos y chapas
        self.BoltName_ep = inp.BoltName_ep
        self.BoltGrade = inp.BoltGrade
        self.BoltThreads = inp.BoltThreads
        self.BoltsInTensionFactor = inp.BoltsInTensionFactor
        self.nbrtension_ep_AME = inp.nbrtension_ep_AME
        self.ExisteNumeTotalFilasTornillos = inp.NumeTotalFilasTornillos is not None
        self.NumeTotalFilasTornillos = inp.NumeTotalFilasTornillos or 0

        # Cotas de chapa y tornillos
        self.b_ep = inp.b_ep
        self.p2_ep = inp.p2_ep
        self.pfo = inp.pfo
        self.dev = inp.dev
        self.pb = inp.pb
        self.pfi = inp.pfi if inp.pfi is not None else (inp.p1 - (self.viga.tf if self.viga else 0.0) - inp.pfo if inp.p1 else 0.0)
        self.p3 = inp.p3
        self.p4 = inp.p4
        self.p5 = inp.p5
        self.p6 = inp.p6
        self.p7 = inp.p7
        self.p8 = inp.p8
        self.p9 = inp.p9
        self.p10 = inp.p10
        self.p11 = inp.p11
        self.p12 = inp.p12
        self.p12_Outer = inp.p12_Outer

        # Espesores y materiales
        self.ThicknessName_t_ep = inp.ThicknessName_t_ep
        self.ThicknessName_t_tenstiff = inp.ThicknessName_t_tenstiff
        self.ThicknessName_t_tenstiff_p23 = inp.ThicknessName_t_tenstiff_p23
        self.ThicknessName_t_webstiff = inp.ThicknessName_t_webstiff
        self.ThicknessName_t_BeamHcompstiff = inp.ThicknessName_t_BeamHcompstiff

        # Cartela y Soldaduras
        self.AnguloCantonera = inp.AnguloCantonera
        self.FactorCantoCantonera = inp.FactorCantoCantonera
        self.ww = inp.ww
        self.wf = inp.wf
        self.wh = inp.wh
        self.wswc = inp.wswc
        self.WeldFactor = inp.WeldFactor
        self.Electrode = inp.Electrode
        self.Fillet_Weld_Type = inp.Fillet_Weld_Type

        # Solicitaciones y Opciones
        self.MzEd = inp.MzEd
        self.FyEd = inp.FyEd
        self.FxEd = inp.FxEd
        self.FzEd = inp.FzEd
        self.MzEd_PanelZone = inp.MzEd_PanelZone
        self.Pr = inp.Pr
        self.Vgra = inp.Vgra
        self.Havg = inp.Havg
        self.Lh = inp.Lh
        self.OMF_SMF = inp.OMF_SMF

        self.TipoCombiEnve = inp.TipoCombiEnve
        self.ConnectionTop = inp.ConnectionTop
        self.yieldmethod = inp.yieldmethod
        self.Traccion = inp.Traccion
        self.SisUniPerMetri = inp.SisUniPerMetri
        self.SisUniPlaMetri = inp.SisUniPlaMetri

        # Slip critical parameters
        self.EsStandardHole = inp.EsStandardHole
        self.EsOverSizedHole = inp.EsOverSizedHole
        self.EsLongSlottedHole = inp.EsLongSlottedHole
        self.EsOneFiller = inp.EsOneFiller
        self.EsClassASurface = inp.EsClassASurface
        self.EsClassBSurface = inp.EsClassBSurface
        self.EsValornuTest = inp.EsValornuTest
        self.EsValorDuTest = inp.EsValorDuTest
        self.nu = inp.nu
        self.Du = inp.Du
        self.ns = inp.ns

        # Salidas calculadas
        self.FntAME_: float = 0.0
        self.FnvAME_: float = 0.0
        self.nbol_: float = 0.0
        self.h_h_: float = 0.0
        self.De_: float = 0.0
        self.Lp_: float = 0.0
        self.c_: float = 0.0
        self.deh_: float = 0.0
        self.h0_: float = 0.0
        self.h1_: float = 0.0
        self.h2_: float = 0.0
        self.h3_: float = 0.0
        self.Mr_: float = 0.0
        self.Mfy_: float = 0.0
        self.Tfb_: float = 0.0
        self.nbol_shear_: float = 0.0
        self.Vfb_: float = 0.0
        self.Vr1_: float = 0.0
        self.Vr2_: float = 0.0
        self.Vr3_: float = 0.0
        self.lc3_: float = 0.0
        self.Vr4_: float = 0.0
        self.lc4_: float = 0.0
        self.VrminAME_: float = 0.0
        self.Y_ep_: float = 0.0
        self.treq_ep_: float = 0.0
        self.tmin_ep_: float = 0.0
        self.Mpl_: float = 0.0
        self.Pt_ep_: float = 0.0
        self.Qmax0_ep_: float = 0.0
        self.Qmaxi_ep_: float = 0.0
        self.Tb_ep_: float = 0.0
        self.Mq_ep_: float = 0.0
        self.Ff_: float = 0.0
        self.Ap_: float = 0.0
        self.RnyAME_: float = 0.0
        self.Apn_: float = 0.0
        self.RnrAME_: float = 0.0
        self.Y_cf_: float = 0.0
        self.treq_cf_: float = 0.0
        self.tmin_cf_: float = 0.0
        self.Mcf_: float = 0.0
        self.Pt_cf_: float = 0.0
        self.Qmax0_cf_: float = 0.0
        self.Qmaxi_cf_: float = 0.0
        self.Tb_cf_: float = 0.0
        self.Mq_cf_: float = 0.0
        self.N_: float = 0.0
        self.k1_: float = 0.0
        self.kc_: float = 0.0
        self.Br1_: float = 0.0
        self.k3_: float = 0.0
        self.Br2_: float = 0.0
        self.Br3_: float = 0.0
        self.BrminAME_: float = 0.0
        self.VrAME_: float = 0.0
        self.Ffts_: float = 0.0
        self.bs_: float = 0.0
        self.clip_s_: float = 0.0
        self.tts_min_: float = 0.0
        self.A_seff_: float = 0.0
        self.KL_: float = 0.0
        self.Fe_: float = 0.0
        self.Fcr_: float = 0.0
        self.CrAME_: float = 0.0
        self.Vfdp_: float = 0.0
        self.Ldp_: float = 0.0
        self.bdp_: float = 0.0
        self.tdp_min_: float = 0.0
        self.VrdpAME_: float = 0.0
        self.C1_: float = 0.0
        self.C2_: float = 0.0
        self.C3_: float = 0.0
        self.N_beam_: float = 0.0
        self.Br1_beam_: float = 0.0
        self.Br2_beam_: float = 0.0
        self.Br3_beam_: float = 0.0
        self.BrminAME_beam_: float = 0.0
        self.C1wsAME_: float = 0.0
        self.bs_beam_: float = 0.0
        self.clip_s_beam_: float = 0.0
        self.tts_min_beam_: float = 0.0
        self.A_seff_beam_: float = 0.0
        self.KL_beam_: float = 0.0
        self.Fe_beam_: float = 0.0
        self.Fcr_beam_: float = 0.0
        self.CrAME_beam_: float = 0.0
        self.Lw_w_: float = 0.0
        self.Aw_w_: float = 0.0
        self.Vr1_w_: float = 0.0
        self.Vr2_w_: float = 0.0
        self.Vr3_w_: float = 0.0
        self.Vr_min_w_: float = 0.0
        self.Lw_f_: float = 0.0
        self.Aw_f_: float = 0.0
        self.Vr1_f_: float = 0.0
        self.Vr2_f_: float = 0.0
        self.Vr3_f_: float = 0.0
        self.Vr_min_f_: float = 0.0
        self.wf_min_: float = 0.0
        self.Lw_h_: float = 0.0
        self.Aw_h_: float = 0.0
        self.Vr1_h_: float = 0.0
        self.Vr2_h_: float = 0.0
        self.Vr3_h_: float = 0.0
        self.Vr_min_h_: float = 0.0
        self.bep_min_: float = 0.0
        self.BaseMatFailFlange: float = 0.0
        self.BaseMatFailWeb: float = 0.0
        self.BaseMatFailHaunch: float = 0.0
        self.RatioPesimo_: float = 0.0

        self.MINedge: float = 0.0
        self.MAXedge: float = 0.0
        self.MINspacing: float = 0.0
        self.MAXspacing: float = 0.0
        self.MINenter: float = 0.0
        self.MINpretension: float = 0.0

    def _cargar_perfiles(self) -> None:
        if not self.viga and self.inp.nombre_viga and self.repo_perfiles:
            self.viga, _ = self.repo_perfiles.consultar(self.inp.nombre_viga, es_columna=False)
        if not self.pilar and self.inp.nombre_pilar and self.repo_perfiles:
            self.pilar, _ = self.repo_perfiles.consultar(self.inp.nombre_pilar, es_columna=True)

        if not self.viga or not self.pilar:
            raise ValueError("Faltan los datos geométricos de la viga o del pilar.")

    def CalculaEsfuerzos(self) -> None:
        mod_aux = self.ext.moduloauxiliar
        if self.inp.MzEd_Porcen is not None:
            self.MzEd = self.inp.MzEd_Porcen / 100.0 * mod_aux.momentores_AME(self.viga.Z, self.fy) / 1000.0
        if self.inp.FyEd_Porcen is not None:
            self.FyEd = self.inp.FyEd_Porcen / 100.0 * mod_aux.ResisCortanteEjeFuerte_AME(self.viga.h, self.viga.tw, self.fy)  # ya en kN
        if self.inp.FxEd_Porcen is not None:
            self.FxEd = self.inp.FxEd_Porcen / 100.0 * mod_aux.traccionres_AME(self.viga.A, self.fy)  # ya en kN
        if self.inp.FzEd_Porcen is not None:
            self.FzEd = self.inp.FzEd_Porcen / 100.0 * mod_aux.ResisCortanteEjeDebil_AME(self.viga.b, self.viga.tf, self.fy)  # ya en kN

    def calcular_ratios(self) -> Dict[str, float]:
        """{descripcion del ratio: valor}. Mismo calculo que `calcular_conexion`, sin construir el
        resultado completo (complementarios, objetos pydantic): para comprobar muchas veces."""
        self.solo_ratios = True
        try:
            self.calcular_conexion()
        finally:
            self.solo_ratios = False
        return self.ratios_dict

    def calcular_dict(self) -> Dict[str, Any]:
        """Mismo calculo que `calcular_conexion` pero sin crear objetos Pydantic."""
        self.solo_dict = True
        try:
            self.calcular_conexion()
        finally:
            self.solo_dict = False
        return self.resultado_dict

    def calcular_conexion(self) -> ResultadoConexion:
        self._cargar_perfiles()
        self.CalculaEsfuerzos()

        self.lambdaM0AME = 0.75
        ratios = [0.0] * self.NumeRatios
        ratios_desc = [""] * self.NumeRatios
        resul_comp = [""] * self.NumeResulComplemen
        desc_resul_comp = [""] * self.NumeResulComplemen

        dP = self.pilar.h - 2.0 * self.pilar.tf - 2.0 * self.pilar.r
        dV = self.viga.h - 2.0 * self.viga.tf - 2.0 * self.viga.r

        tbl_pernos = self.ext.tablapernos()
        tbl_pernos.lambdaM0AME = str(self.lambdaM0AME)
        if self.TipoCombiEnve == "ASD":
            tbl_pernos.lambdaM0AME = "0.5"
        tbl_pernos.nombretornillo = self.BoltName_ep
        tbl_pernos.threadscondition = self.BoltThreads
        tbl_pernos.gradotornillo = self.BoltGrade

        if self.SisUniPerMetri:
            self.FntAME_ = tbl_pernos.FntmetAME
            self.FnvAME_ = tbl_pernos.FnvmetAME
            db = tbl_pernos.dmetrico
            dh = tbl_pernos.dhmetrico
            self.MINedge = tbl_pernos.disborminmet
            self.MINspacing = tbl_pernos.pitchminmet
            self.MINenter = tbl_pernos.dislibmet
            self.MINpretension = tbl_pernos.Tb_Me
        else:
            self.FntAME_ = tbl_pernos.FntimpAME
            self.FnvAME_ = tbl_pernos.FnvimpAME
            db = tbl_pernos.dimperial
            dh = tbl_pernos.dhimperial
            self.MINedge = tbl_pernos.disborminimp
            self.MINspacing = tbl_pernos.pitchminimp
            self.MINenter = tbl_pernos.dislibmet
            self.MINpretension = tbl_pernos.Tb_Me

        self.h_h_ = 0.0
        if self.AnguloCantonera != 0.0:
            self.h_h_ = self.FactorCantoCantonera * self.viga.h / math.sin(math.radians(self.AnguloCantonera))

        self.De_ = self.viga.h + self.h_h_
        self.Lp_ = math.ceil((self.De_ + 2.0 * self.pfo + 2.0 * self.dev) / 10.0) * 10.0
        self.c_ = self.ext.moduloauxiliar.redond_mult(self.pfo + self.pfi + self.viga.tf, 5.0)
        self.deh_ = (self.b_ep - self.p2_ep) / 2.0
        self.h0_ = self.pfo + self.De_ - self.viga.tf / 2.0
        self.h1_ = self.h0_ - self.c_
        self.h2_ = 0.0
        self.h3_ = 0.0

        if self.BoltsInTensionFactor * self.nbrtension_ep_AME == 6.0:
            self.h2_ = self.h1_ - self.pb
        elif self.BoltsInTensionFactor * self.nbrtension_ep_AME == 8.0:
            self.h3_ = self.h1_ - self.pb
            self.h2_ = self.h1_
            self.h1_ = self.h0_
            self.h0_ += self.p12_Outer

        arr_pos = [0.0] * 13
        arr_pos[0] = self.dev
        arr_pos[1] = self.c_ + arr_pos[0]
        arr_pos[2] = self.pb + arr_pos[1]
        arr_pos[3] = self.p3 + arr_pos[2]
        arr_pos[4] = self.p4 + arr_pos[3]
        arr_pos[5] = self.p5 + arr_pos[4]
        arr_pos[6] = self.p6 + arr_pos[5]
        arr_pos[7] = self.p7 + arr_pos[6]
        arr_pos[8] = self.p8 + arr_pos[7]
        arr_pos[9] = self.p9 + arr_pos[8]
        arr_pos[10] = self.p10 + arr_pos[9]
        arr_pos[11] = self.p11 + arr_pos[10]
        arr_pos[12] = self.p12 + arr_pos[11]

        if self.ExisteNumeTotalFilasTornillos:
            num_filas = self.NumeTotalFilasTornillos
        else:
            num_filas = 1
            for i in range(12):
                if arr_pos[i] != arr_pos[i + 1]:
                    num_filas += 1

        self.nbol_ = 2.0 * num_filas
        nbrinner_ep_AME = int(round((self.nbol_ - 2.0 * self.BoltsInTensionFactor * self.nbrtension_ep_AME) / 2.0))
        self.nbol_shear_ = self.nbol_ - self.BoltsInTensionFactor * self.nbrtension_ep_AME

        posObstaculo1 = self.dev + self.pfo
        posObstaculo2 = self.dev + self.pfo + self.viga.tf
        posObstaculo3 = self.dev + self.pfo + self.viga.h - self.viga.tf
        posObstaculo4 = self.dev + self.pfo + self.viga.h
        posObstaculo5 = self.dev + self.pfo + self.De_ - self.viga.tf
        posObstaculo6 = self.dev + self.pfo + self.De_
        pos_mitad_viga = self.dev + self.pfo + self.viga.h / 2.0

        # Tablas de chapa
        tp_obj = self.ext.tablaplaca()
        tp_obj.nombreplaca = self.ThicknessName_t_ep
        if self.SisUniPlaMetri:
            tp = float(tp_obj.tmetrico)
            self.MAXedge = float(tp_obj.disbormaxmet)
            self.MAXspacing = float(tp_obj.pitchmaxmet)
        else:
            tp = float(tp_obj.timperial)
            self.MAXedge = float(tp_obj.disbormaximp)
            self.MAXspacing = float(tp_obj.pitchmaximp)

        t_tenstiff_obj = self.ext.tablaplaca()
        t_tenstiff_obj.nombreplaca = self.ThicknessName_t_tenstiff
        esp_t_s = float(t_tenstiff_obj.tmetrico if self.SisUniPlaMetri else t_tenstiff_obj.timperial)

        t_p23_obj = self.ext.tablaplaca()
        t_p23_obj.nombreplaca = self.ThicknessName_t_tenstiff_p23
        ts2 = float(t_p23_obj.tmetrico if self.SisUniPlaMetri else t_p23_obj.timperial)

        t_web_obj = self.ext.tablaplaca()
        t_web_obj.nombreplaca = self.ThicknessName_t_webstiff
        esp_w_d_p = float(t_web_obj.tmetrico if self.SisUniPlaMetri else t_web_obj.timperial)

        t_hcomp_obj = self.ext.tablaplaca()
        t_hcomp_obj.nombreplaca = self.ThicknessName_t_BeamHcompstiff
        esp_h_w_s = float(t_hcomp_obj.tmetrico if self.SisUniPlaMetri else t_hcomp_obj.timperial)

        fuweld = self.ext.moduloauxiliar.fuweldtabla(self.Electrode)
        num8 = abs(self.MzEd / ((self.De_ - self.viga.tf) / 1000.0))
        num9 = num8
        if self.OMF_SMF in (1, 2):
            if self.OMF_SMF == 1:
                num9 = abs(self.MzEd_PanelZone / ((self.De_ - self.viga.tf) / 1000.0))
            else:
                num9 = abs(self.MzEd_PanelZone / ((self.De_ - self.viga.tf) / 1000.0)) - self.MzEd_PanelZone / (self.Havg / 1000.0)

        if self.OMF_SMF == 1:
            self.FyEd = 2.0 * self.MzEd / self.Lh
        elif self.OMF_SMF == 2:
            self.FyEd = 2.0 * self.MzEd / self.Lh + self.Vgra

        self.Ff_ = num8
        if self.Traccion:
            self.Ff_ += abs(0.5 * self.FxEd)

        num10 = 0.0
        if self.Traccion:
            self.Tfb_ = abs(self.MzEd / (2.0 * (self.h0_ + self.h1_ + self.h2_ + self.h3_) / 1000.0)) + self.FxEd / self.nbol_
            num10 = (self.FxEd / self.nbol_) * self.nbol_shear_
        else:
            self.Tfb_ = abs(self.MzEd / (2.0 * (self.h0_ + self.h1_ + self.h2_ + self.h3_) / 1000.0)) - self.FxEd / self.nbol_

        if self.Tfb_ < 0.0:
            self.Tfb_ = 0.0

        self.BaseMatFailWeb = 0.0
        self.BaseMatFailFlange = 0.0
        self.BaseMatFailHaunch = 0.0

        # Step 1: Bolts in Tension
        ratios[0], ratios_desc[0] = self.step1(self.FntAME_, self.Tfb_)

        # Step 2: Bolts in Shear
        (self.Vfb_, self.Vr1_, self.Vr2_, self.Vr3_, self.lc3_, self.Vr4_, self.lc4_,
         self.VrminAME_, ratios[1], ratios_desc[1]) = self.step2(
            self.TipoCombiEnve, self.fy, self.fu, self.fup, self.pilar.tf, self.FnvAME_,
            int(round(self.nbol_shear_)), db, dh, tp, self.c_, self.pb, self.dev, self.FyEd
        )

        # Step 1_2: Bolts in combined tension/shear
        ratios_desc[2] = "Bolts in combined tension/shear"
        ratios[2] = self.step1_2(self.TipoCombiEnve, num10, self.FyEd, db, int(round(self.nbol_shear_)), self.FntAME_, self.FnvAME_)
        # Interaccion sobre los tornillos traccionados (cortante repartido entre todos): siempre da valor.
        nbol_all = int(round(self.nbol_))
        ratios[2] = max(
            ratios[2],
            self.step1_2(self.TipoCombiEnve, self.Tfb_ * nbol_all, self.FyEd, db, nbol_all, self.FntAME_, self.FnvAME_),
        )

        # Step 3: End Plate / Yield Method
        if not self.yieldmethod:
            ratios_desc[3] = "Bolt prying action for tensile bolts(E-P)"
            ratios[3] = 0.0
        else:
            self.Mr_ = self.FntAME_ * 2.0 * (self.h0_ + self.h1_ + self.h2_ + self.h3_) / 1000.0
            if self.Traccion:
                self.Mfy_ = self.MzEd + self.FxEd * (0.5 * self.viga.h + self.h_h_ - self.viga.tf / 2.0) / 1000.0
            else:
                self.Mfy_ = self.MzEd

            (self.Y_ep_, self.treq_ep_, self.tmin_ep_, self.Mpl_, self.Pt_ep_, self.Qmax0_ep_,
             self.Qmaxi_ep_, self.Tb_ep_, self.Mq_ep_, ratios[3], ratios_desc[3]) = self.step3_yieldmethod(
                self.TipoCombiEnve, int(round(self.BoltsInTensionFactor * self.nbrtension_ep_AME)),
                self.h0_, self.h1_, self.h2_, self.h3_, self.b_ep, self.p2_ep, self.pfo, self.c_,
                self.viga.tf, self.p12_Outer, self.pb, self.fyp, self.Mr_, self.Mfy_, tp,
                self.BoltGrade, self.pfi, db, dh, self.dev, self.FntAME_, self.MzEd
            )

        # Step 4: Shear yielding
        self.Ap_, self.RnyAME_, ratios[4], ratios_desc[4] = self.step4(self.TipoCombiEnve, self.fyp, self.b_ep, tp, self.Ff_)

        # Step 5: Shear rupture
        self.Apn_, self.RnrAME_, ratios[5], ratios_desc[5] = self.step5(self.TipoCombiEnve, self.fup, self.b_ep, tp, dh, self.Ff_)

        # Step 6: Column flange
        if not self.yieldmethod:
            ratios_desc[6] = "Bolt prying action for tensile bolts(Col)"
            ratios[6] = 0.0
        else:
            (self.Y_cf_, self.treq_cf_, self.tmin_cf_, self.Mcf_, self.Pt_cf_, self.Qmax0_cf_,
             self.Qmaxi_cf_, self.Tb_cf_, self.Mq_cf_, ratios[6], ratios_desc[6]) = self.step6_yieldmethod(
                self.TipoCombiEnve, int(round(self.BoltsInTensionFactor * self.nbrtension_ep_AME)),
                self.h0_, self.h1_, self.h2_, self.h3_, self.pilar.b, self.p2_ep, self.pfo, self.c_,
                self.viga.tf, self.p12_Outer, self.pb, esp_t_s, ts2, self.Mr_, self.fy, self.Mfy_,
                self.pilar.tf, self.BoltGrade, self.pfi, db, dh, self.dev, self.FntAME_, self.MzEd
            )

        # Step 7: Column web crippling, yielding & buckling
        (self.N_, self.k1_, self.kc_, self.Br1_, self.k3_, self.Br2_, self.Br3_, self.BrminAME_,
         ratios[7], ratios_desc[7]) = self.step7(
            self.TipoCombiEnve, self.viga.tf, self.viga.r, self.pilar.tf, self.pilar.tw, self.pilar.r,
            self.pilar.h, dP, tp, self.WeldFactor * self.wf, self.fy, self.Ff_, self.E, esp_t_s, self.ConnectionTop
        )

        # Step 8: Panel zone shear
        self.VrAME_, ratios[8], ratios_desc[8] = self.step8(
            self.TipoCombiEnve, self.pilar.A, self.pilar.h, self.pilar.tw, self.pilar.tf,
            self.pilar.b, self.viga.h, num9, self.fy, esp_w_d_p, self.Pr
        )

        # Step 9: Transverse stiffeners
        (self.Ffts_, self.bs_, self.clip_s_, self.tts_min_, self.A_seff_, self.KL_, self.Fe_,
         self.Fcr_, self.CrAME_, ratios[9], ratios_desc[9]) = self.step9(
            self.TipoCombiEnve, self.BrminAME_, self.Ff_, self.pilar.b, self.pilar.tw, self.pilar.r,
            self.pilar.h, self.pilar.tf, self.viga.tf, esp_t_s, self.ConnectionTop, self.E, self.fyp
        )

        # Step 10: Doubler plate
        (self.Vfdp_, self.Ldp_, self.bdp_, self.tdp_min_, self.VrdpAME_, ratios[10], ratios_desc[10]) = self.step10(
            self.TipoCombiEnve, self.VrAME_, num9, self.Ffts_, self.clip_s_, esp_w_d_p, self.De_,
            self.pilar.tf, self.pilar.r, tp, self.pilar.h, self.fyp, esp_t_s, self.Fillet_Weld_Type
        )

        # Step 11: Beam web yielding/crippling
        (self.C1_, self.C2_, self.C3_, self.N_beam_, self.Br1_beam_, self.Br2_beam_, self.Br3_beam_,
         self.BrminAME_beam_, ratios[11], ratios_desc[11]) = self.step11(
            self.TipoCombiEnve, self.AnguloCantonera, self.De_, self.viga.h, dV, self.viga.tf,
            self.viga.tw, self.viga.r, abs(self.MzEd), esp_h_w_s, self.fy, self.E
        )

        # Step 12: Haunch web stiffeners
        (self.C1wsAME_, self.bs_beam_, self.clip_s_beam_, self.tts_min_beam_, self.A_seff_beam_,
         self.KL_beam_, self.Fe_beam_, self.Fcr_beam_, self.CrAME_beam_, ratios[12], ratios_desc[12]) = self.step12(
            self.TipoCombiEnve, self.BrminAME_beam_, self.C1_, self.viga.b, self.viga.tw, self.viga.r,
            self.viga.h, self.viga.tf, esp_h_w_s, self.E, self.fyp
        )

        # Step 13: Weld beam web to end plate
        (self.Lw_w_, self.Aw_w_, self.Vr1_w_, self.Vr2_w_, self.Vr3_w_, self.Vr_min_w_,
         ratios[13], ratios_desc[13]) = self.step13(
            self.TipoCombiEnve, tp, self.viga.h, self.viga.tf, self.viga.r, self.viga.tw,
            self.WeldFactor * self.ww, fuweld, self.FyEd, self.fyp, self.fy
        )
        if self.Vr_min_w_ != self.Vr3_w_:
            self.BaseMatFailWeb = 1.0

        # Step 14: Weld beam flange to end plate
        (self.Lw_f_, self.Aw_f_, self.Vr1_f_, self.Vr2_f_, self.Vr3_f_, self.Vr_min_f_,
         self.wf_min_, ratios[14], ratios_desc[14]) = self.step14(
            self.TipoCombiEnve, self.fy, self.viga.b, self.viga.tf, self.viga.tw, self.viga.r,
            tp, self.Ff_, self.WeldFactor * self.wf, fuweld, self.fu, self.fup
        )
        if self.Vr_min_f_ != self.Vr3_f_:
            self.BaseMatFailFlange = 1.0

        # Step 15: Weld haunch web to beam weld
        if self.AnguloCantonera != 0.0:
            (self.Lw_h_, self.Aw_h_, self.Vr1_h_, self.Vr2_h_, self.Vr3_h_, self.Vr_min_h_,
             ratios[15], ratios_desc[15]) = self.step15(
                self.TipoCombiEnve, self.viga.tf, self.viga.tw, self.WeldFactor * self.wh, self.h_h_,
                self.AnguloCantonera, fuweld, self.fu, self.C2_
            )
        else:
            ratios_desc[15] = "Weld - Haunch web to beam weld"
            ratios[15] = 0.0
            if self.Vr_min_h_ != self.Vr3_h_:
                self.BaseMatFailHaunch = 1.0

        # Step 16: Slip critical
        ratios[16], ratios_desc[16] = self.SlipCritical(
            self.TipoCombiEnve, self.MINpretension, self.FxEd / self.nbol_, self.FyEd / self.nbol_
        )

        # Step 17: Stiffener to column web weld
        ratios[17], ratios_desc[17] = self.StiffenerToColumnWebWeld(
            self.Ff_, self.TipoCombiEnve, self.wswc, self.pilar.h, self.pilar.b, self.pilar.tf,
            self.pilar.tw, self.pilar.r, esp_t_s, fuweld, self.fu, self.fy, self.fup
        )

        # Cálculo del ratio pésimo (pasos 0 a 14 como en C# original)
        self.RatioPesimo_ = 0.0
        num49 = 0
        for i in range(15):
            if ratios[i] > self.RatioPesimo_:
                self.RatioPesimo_ = ratios[i]
                num49 = i

        # Via rapida (calcular_ratios): solo se necesitan los ratios, sin complementarios ni pydantic.
        if self.solo_ratios:
            self.ratios_dict = {str(ratios_desc[i]).strip(): ratios[i]
                                for i in range(self.NumeRatios) if not math.isnan(ratios[i])}
            return None  # type: ignore[return-value]

        # Resultados complementarios
        desc_resul_comp[0] = "PRMIN"
        resul_comp[0] = "REVISAR: PERFIL NO APTO COMO COLUMNA" if self.pilar.Prmin == 0.0 else ("OK" if not (self.p2_ep < self.pilar.Prmin) else f"REVISAR: {self.p2_ep - self.pilar.Prmin} / prmin: {self.pilar.Prmin}")

        desc_resul_comp[1] = "PRMAX"
        resul_comp[1] = "REVISAR: PERFIL NO APTO COMO COLUMNA" if self.pilar.Prmax == 0.0 else ("OK" if not (self.p2_ep > self.pilar.Prmax) else f"REVISAR: {self.pilar.Prmax - self.p2_ep} / prmax: {self.pilar.Prmax}")

        desc_resul_comp[2] = "DEH_MIN"
        resul_comp[2] = "OK" if not (self.deh_ < self.MINedge) else f"REVISAR: {self.deh_ - self.MINedge} / MinEdgeDist: {self.MINedge}"

        desc_resul_comp[3] = "DEH_MAX"
        resul_comp[3] = "OK" if not (self.deh_ > self.MAXedge) else f"REVISAR: {self.MAXedge - self.deh_} / MaxEdgeDist: {self.MAXedge}"

        desc_resul_comp[4] = "DEV_MIN"
        resul_comp[4] = f"REVISAR: {self.dev - self.MINedge} / MinEdgeDist: {self.MINedge}" if self.dev < self.MINedge else "OK"

        desc_resul_comp[5] = "DEV_MAX"
        resul_comp[5] = f"REVISAR: {self.MAXedge - self.dev} / MaxEdgeDist: {self.MAXedge}" if self.dev > self.MAXedge else "OK"

        # Filas de tornillos activas (los pasos a 0 no añaden fila).
        filas_tornillos = [arr_pos[0]]
        for i in range(1, 13):
            if arr_pos[i] != arr_pos[i - 1]:
                filas_tornillos.append(arr_pos[i])

        # Espaciamiento mínimo / máximo entre filas consecutivas.
        seps = [arr_pos[i + 1] - arr_pos[i] for i in range(12) if arr_pos[i + 1] > arr_pos[i]]
        desc_resul_comp[6] = "SPACING_MIN"
        resul_comp[6] = (
            "OK" if (not seps or min(seps) >= self.MINspacing)
            else f"REVISAR: separación mínima {min(seps):g} < {self.MINspacing:g}"
        )
        desc_resul_comp[7] = "SPACING_MAX"
        resul_comp[7] = (
            "OK" if (not seps or max(seps) <= self.MAXspacing)
            else f"REVISAR: separación máxima {max(seps):g} > {self.MAXspacing:g}"
        )

        # Distancias a obstáculos: colisión de la fila de tornillos con las CARAS de las
        # alas (solo se comprueba el eje vertical). El margen es el espacio de herramienta
        # del tornillo (`clearance`), medido desde el CENTRO del agujero (no incluye el
        # radio). Las filas entre las alas (alma) o dentro del cuerpo de la cantonera NO
        # son conflicto por sí mismas: solo lo son si la herramienta toca una cara de ala.
        _margen = tablapernos.DATOS_PERNOS.get(int(round(db)), {}).get("clearance", 0.0)

        def _toca_al(pos_obs):
            for j, y in enumerate(filas_tornillos, start=1):
                if abs(y - pos_obs) < _margen:
                    return f"REVISAR: fila {j} a {abs(y - pos_obs):.1f} mm (< {_margen:.1f})"
            return "OK"

        for idx, pos_obs, key in [
            (8, posObstaculo1, "ENTER_TOP_FLANGE_TOP"),
            (9, posObstaculo2, "ENTER_TOP_FLANGE_LOW"),
            (10, posObstaculo3, "ENTER_LOW_FLANGE_TOP"),
            (11, posObstaculo4, "ENTER_LOW_FLANGE_LOW"),
        ]:
            desc_resul_comp[idx] = key
            resul_comp[idx] = _toca_al(pos_obs)

        desc_resul_comp[12] = "ENTER_HAUNCH_FLANGE_TOP"
        desc_resul_comp[13] = "ENTER_HAUNCH_FLANGE_LOW"
        if self.AnguloCantonera != 0.0:
            resul_comp[12] = _toca_al(posObstaculo5)
            resul_comp[13] = _toca_al(posObstaculo6)
        else:
            resul_comp[12] = "OK"
            resul_comp[13] = "OK"

        self.bep_min_ = math.ceil((self.viga.b + 20.0) / 5.0) * 5.0
        desc_resul_comp[14] = "bc&bb"
        resul_comp[14] = "OK" if not (self.pilar.b < self.viga.b) else f"REVISAR: {self.pilar.b - self.viga.b} / bc: {self.pilar.b}"

        desc_resul_comp[15] = "bep&bb"
        resul_comp[15] = "OK" if not (self.b_ep < self.viga.b) else f"REVISAR: {self.b_ep - self.viga.b} / bb: {self.viga.b}"

        desc_resul_comp[16] = "bep&bepmin"
        resul_comp[16] = "OK" if not (self.b_ep < self.bep_min_) else f"REVISAR: {self.b_ep - self.bep_min_} / bepmin: {self.bep_min_}"

        desc_resul_comp[17] = "bep&bc"
        resul_comp[17] = "OK" if not (self.b_ep > self.pilar.b) else f"REVISAR: {self.pilar.b - self.b_ep} / bep: {self.b_ep}"

        desc_resul_comp[18] = "Zb/Zc"
        resul_comp[18] = str(self.viga.Z / self.pilar.Z) if self.pilar.Z else "N/A"
        desc_resul_comp[19] = "BMFailWeb"
        resul_comp[19] = str(self.BaseMatFailWeb)
        desc_resul_comp[20] = "BMFailFlange"
        resul_comp[20] = str(self.BaseMatFailFlange)
        desc_resul_comp[21] = "BMFailHaunch"
        resul_comp[21] = str(self.BaseMatFailHaunch)
        desc_resul_comp[22] = "wfMIN"
        resul_comp[22] = str(self.wf_min_)
        desc_resul_comp[23] = "RatioPesimo"
        resul_comp[23] = str(self.RatioPesimo_)
        desc_resul_comp[24] = "DescripRatioPesimo"
        resul_comp[24] = ratios_desc[num49]

        desc_resul_comp[25] = "CoincidenFilas"
        resul_comp[25] = "OK" if float(num_filas) == self.nbol_ / 2.0 else "El número de filas de tornillos introducido por distancias NO coincide con el introducido como brinner"

        desc_resul_comp[26] = "NoHaunchPerfilCentrado"
        desc_resul_comp[27] = "NoHaunchTornilloImparCentrado"
        desc_resul_comp[28] = "NoHaunchSimetriaFilasTornillos"
        resul_comp[26] = "OK"
        resul_comp[27] = "OK"
        resul_comp[28] = "OK"

        if self.AnguloCantonera == 0.0:
            if abs((pos_mitad_viga - self.Lp_ / 2.0) / pos_mitad_viga) > 0.001:
                resul_comp[26] = "El perfil no está centrado respecto de la chapa"

        desc_resul_comp[29] = "Número de filas de tornillos interiores"
        resul_comp[29] = str(nbrinner_ep_AME)
        desc_resul_comp[30] = "Connection Capacity Axial [kN]"
        resul_comp[30] = str(self.FxEd)
        desc_resul_comp[31] = "Connection Capacity Shear [kN]"
        resul_comp[31] = str(self.FyEd)
        desc_resul_comp[32] = "Connection Capacity Moment [kN*m]"
        resul_comp[32] = str(self.MzEd)

        hay_fallo_geometrico = any(
            str(resultado).startswith("REVISAR") for resultado in resul_comp
        )
        ratio_max_todos = max(ratios) if ratios else 0.0
        cumple = (ratio_max_todos <= 1.0) and not hay_fallo_geometrico

        if self.solo_dict:
            self.resultado_dict = {
                "ratio_pesimo": self.RatioPesimo_,
                "descripcion_ratio_pesimo": ratios_desc[num49],
                "ratios": [
                    {"indice": i, "descripcion": ratios_desc[i],
                     "ratio": ratios[i] if not math.isnan(ratios[i]) else None}
                    for i in range(self.NumeRatios)
                ],
                "complementarios": [
                    {"indice": i, "descripcion": desc_resul_comp[i], "resultado": resul_comp[i]}
                    for i in range(self.NumeResulComplemen)
                ],
                "cumple": cumple,
            }
            return None  # type: ignore[return-value]

        # Construcción de la respuesta Pydantic
        filas_ratios = [
            FilaRatio(indice=i, descripcion=ratios_desc[i], ratio=ratios[i] if not math.isnan(ratios[i]) else None)
            for i in range(self.NumeRatios)
        ]
        comp_objs = [
            Complementario(indice=i, descripcion=desc_resul_comp[i], resultado=resul_comp[i])
            for i in range(self.NumeResulComplemen)
        ]

        intermedios = {
            "FntAME": self.FntAME_, "FnvAME": self.FnvAME_, "nbol": self.nbol_,
            "Tfb": self.Tfb_, "Ff": self.Ff_, "De": self.De_, "Lp": self.Lp_, "c": self.c_
        }

        return ResultadoConexion(
            ratios=filas_ratios,
            ratio_pesimo=self.RatioPesimo_,
            descripcion_ratio_pesimo=ratios_desc[num49],
            ratio_max_todos=ratio_max_todos,
            complementarios=comp_objs,
            intermedios=intermedios,
            avisos=[],
            cumple=cumple
        )

    def calcular_geometria(self) -> Dict[str, str]:
        """Via rapida para el refinado: solo comprobaciones geometricas.

        Evita los 17 pasos de resistencia y la construccion de objetos Pydantic.
        Devuelve {clave: texto} solo para las comprobaciones con REVISAR.
        """
        self._cargar_perfiles()
        self.CalculaEsfuerzos()
        self.lambdaM0AME = 0.75

        tbl_pernos = self.ext.tablapernos()
        tbl_pernos.lambdaM0AME = str(self.lambdaM0AME)
        if self.TipoCombiEnve == "ASD":
            tbl_pernos.lambdaM0AME = "0.5"
        tbl_pernos.nombretornillo = self.BoltName_ep
        tbl_pernos.threadscondition = self.BoltThreads
        tbl_pernos.gradotornillo = self.BoltGrade

        if self.SisUniPerMetri:
            db = tbl_pernos.dmetrico
            dh = tbl_pernos.dhmetrico
            self.MINedge = tbl_pernos.disborminmet
            self.MINspacing = tbl_pernos.pitchminmet
            self.MINenter = tbl_pernos.dislibmet
        else:
            db = tbl_pernos.dimperial
            dh = tbl_pernos.dhimperial
            self.MINedge = tbl_pernos.disborminimp
            self.MINspacing = tbl_pernos.pitchminimp
            self.MINenter = tbl_pernos.dislibmet

        self.h_h_ = 0.0
        if self.AnguloCantonera != 0.0:
            self.h_h_ = self.FactorCantoCantonera * self.viga.h / math.sin(math.radians(self.AnguloCantonera))

        self.De_ = self.viga.h + self.h_h_
        self.Lp_ = math.ceil((self.De_ + 2.0 * self.pfo + 2.0 * self.dev) / 10.0) * 10.0
        self.c_ = self.ext.moduloauxiliar.redond_mult(self.pfo + self.pfi + self.viga.tf, 5.0)
        self.deh_ = (self.b_ep - self.p2_ep) / 2.0

        arr_pos = [0.0] * 13
        arr_pos[0] = self.dev
        arr_pos[1] = self.c_ + arr_pos[0]
        arr_pos[2] = self.pb + arr_pos[1]
        arr_pos[3] = self.p3 + arr_pos[2]
        arr_pos[4] = self.p4 + arr_pos[3]
        arr_pos[5] = self.p5 + arr_pos[4]
        arr_pos[6] = self.p6 + arr_pos[5]
        arr_pos[7] = self.p7 + arr_pos[6]
        arr_pos[8] = self.p8 + arr_pos[7]
        arr_pos[9] = self.p9 + arr_pos[8]
        arr_pos[10] = self.p10 + arr_pos[9]
        arr_pos[11] = self.p11 + arr_pos[10]
        arr_pos[12] = self.p12 + arr_pos[11]

        if self.ExisteNumeTotalFilasTornillos:
            num_filas = self.NumeTotalFilasTornillos
        else:
            num_filas = 1
            for i in range(12):
                if arr_pos[i] != arr_pos[i + 1]:
                    num_filas += 1
        self.nbol_ = 2.0 * num_filas

        posObstaculo1 = self.dev + self.pfo
        posObstaculo2 = self.dev + self.pfo + self.viga.tf
        posObstaculo3 = self.dev + self.pfo + self.viga.h - self.viga.tf
        posObstaculo4 = self.dev + self.pfo + self.viga.h
        posObstaculo5 = self.dev + self.pfo + self.De_ - self.viga.tf
        posObstaculo6 = self.dev + self.pfo + self.De_

        tp_obj = self.ext.tablaplaca()
        tp_obj.nombreplaca = self.ThicknessName_t_ep
        if self.SisUniPlaMetri:
            self.MAXedge = float(tp_obj.disbormaxmet)
            self.MAXspacing = float(tp_obj.pitchmaxmet)
        else:
            self.MAXedge = float(tp_obj.disbormaximp)
            self.MAXspacing = float(tp_obj.pitchmaximp)

        filas_tornillos = [arr_pos[0]]
        for i in range(1, 13):
            if arr_pos[i] != arr_pos[i - 1]:
                filas_tornillos.append(arr_pos[i])

        seps = [arr_pos[i + 1] - arr_pos[i] for i in range(12) if arr_pos[i + 1] > arr_pos[i]]

        fallos: Dict[str, str] = {}

        def _rev(key: str, texto: str) -> None:
            if texto.upper().startswith("REVISAR"):
                fallos[key] = texto

        if self.pilar.Prmin == 0.0:
            _rev("PRMIN", "REVISAR: PERFIL NO APTO COMO COLUMNA")
        else:
            _rev("PRMIN", "OK" if not (self.p2_ep < self.pilar.Prmin)
                 else f"REVISAR: {self.p2_ep - self.pilar.Prmin} / prmin: {self.pilar.Prmin}")

        if self.pilar.Prmax == 0.0:
            _rev("PRMAX", "REVISAR: PERFIL NO APTO COMO COLUMNA")
        else:
            _rev("PRMAX", "OK" if not (self.p2_ep > self.pilar.Prmax)
                 else f"REVISAR: {self.pilar.Prmax - self.p2_ep} / prmax: {self.pilar.Prmax}")

        _rev("DEH_MIN", "OK" if not (self.deh_ < self.MINedge)
             else f"REVISAR: {self.deh_ - self.MINedge} / MinEdgeDist: {self.MINedge}")
        _rev("DEH_MAX", "OK" if not (self.deh_ > self.MAXedge)
             else f"REVISAR: {self.MAXedge - self.deh_} / MaxEdgeDist: {self.MAXedge}")
        _rev("DEV_MIN", f"REVISAR: {self.dev - self.MINedge} / MinEdgeDist: {self.MINedge}"
             if self.dev < self.MINedge else "OK")
        _rev("DEV_MAX", f"REVISAR: {self.MAXedge - self.dev} / MaxEdgeDist: {self.MAXedge}"
             if self.dev > self.MAXedge else "OK")

        _rev("SPACING_MIN", "OK" if (not seps or min(seps) >= self.MINspacing)
             else f"REVISAR: separacion minima {min(seps):g} < {self.MINspacing:g}")
        _rev("SPACING_MAX", "OK" if (not seps or max(seps) <= self.MAXspacing)
             else f"REVISAR: separacion maxima {max(seps):g} > {self.MAXspacing:g}")

        _margen = tablapernos.DATOS_PERNOS.get(int(round(db)), {}).get("clearance", 0.0)

        def _toca_al(pos_obs):
            for j, y in enumerate(filas_tornillos, start=1):
                if abs(y - pos_obs) < _margen:
                    return f"REVISAR: fila {j} a {abs(y - pos_obs):.1f} mm (< {_margen:.1f})"
            return "OK"

        for key, pos_obs in (
            ("ENTER_TOP_FLANGE_TOP", posObstaculo1),
            ("ENTER_TOP_FLANGE_LOW", posObstaculo2),
            ("ENTER_LOW_FLANGE_TOP", posObstaculo3),
            ("ENTER_LOW_FLANGE_LOW", posObstaculo4),
        ):
            _rev(key, _toca_al(pos_obs))

        if self.AnguloCantonera != 0.0:
            _rev("ENTER_HAUNCH_FLANGE_TOP", _toca_al(posObstaculo5))
            _rev("ENTER_HAUNCH_FLANGE_LOW", _toca_al(posObstaculo6))
        else:
            _rev("ENTER_HAUNCH_FLANGE_TOP", "OK")
            _rev("ENTER_HAUNCH_FLANGE_LOW", "OK")

        self.bep_min_ = math.ceil((self.viga.b + 20.0) / 5.0) * 5.0
        _rev("bc&bb", "OK" if not (self.pilar.b < self.viga.b)
             else f"REVISAR: {self.pilar.b - self.viga.b} / bc: {self.pilar.b}")
        _rev("bep&bb", "OK" if not (self.b_ep < self.viga.b)
             else f"REVISAR: {self.b_ep - self.viga.b} / bb: {self.viga.b}")
        _rev("bep&bepmin", "OK" if not (self.b_ep < self.bep_min_)
             else f"REVISAR: {self.b_ep - self.bep_min_} / bepmin: {self.bep_min_}")
        _rev("bep&bc", "OK" if not (self.b_ep > self.pilar.b)
             else f"REVISAR: {self.pilar.b - self.b_ep} / bep: {self.b_ep}")

        return fallos

    # --- MÉTODOS DE CÁLCULO DE PASOS (STEPS) ---

    def step1(self, FntAME: float, Tfb: float) -> Tuple[float, str]:
        return (Tfb / FntAME if FntAME != 0 else 0.0, "Bolts in Tension")

    def step1_2(self, TipoCarga: str, T: float, V: float, db: float, NumeBoltsCortante: float, FntAME: float, FnvAME: float) -> float:
        num = 1000.0
        num2 = math.pi * (db ** 2) / 4.0
        num3 = V * num / (NumeBoltsCortante * num2) if (NumeBoltsCortante * num2) != 0 else 0.0
        left = self.lambdaM0AME
        if TipoCarga == "ASD":
            left = 0.5
        fnt_stress = (FntAME / (left * num2)) * num if (left * num2) != 0 else 0.0
        num4 = 1.3 * fnt_stress - (FntAME / (left * FnvAME)) * num3 if FnvAME != 0 else fnt_stress
        num4 = max(min(num4, fnt_stress), 0.0)   # AISC J3-3a: F'nt <= Fnt
        num5 = (left * num2 * num4 * NumeBoltsCortante) / num
        return T / num5 if num5 != 0 else 0.0

    def step2(self, TipoCarga: str, Fy: float, Fu: float, Fup: float, tfP: float, FnvAME: float,
              nbolt_shear: int, db: float, dh: float, tp: float, c: float, pb: float, dev: float, Vf: float) -> Tuple:
        namrt2 = "Bolts in Shear"
        Vfb = Vf / float(nbolt_shear) if nbolt_shear != 0 else 0.0
        VrminAME = FnvAME
        num = 0.5 if TipoCarga == "ASD" else 0.75

        Vr1 = num * 2.4 * tp * db * Fu / 1000.0
        if Vr1 < VrminAME:
            VrminAME = Vr1

        Vr2 = num * 2.4 * tfP * db * Fu / 1000.0
        if Vr2 < VrminAME:
            VrminAME = Vr2

        lc3 = dev - 0.5 * dh
        Vr3 = num * 1.2 * lc3 * tp * Fup / 1000.0
        if Vr3 < VrminAME:
            VrminAME = Vr3

        lc4 = c - dh
        Vr4 = num * 1.2 * lc4 * tp * Fup / 1000.0
        if Vr4 < VrminAME:
            VrminAME = Vr4

        rtamem2 = Vfb / VrminAME if VrminAME != 0 else 0.0
        return Vfb, Vr1, Vr2, Vr3, lc3, Vr4, lc4, VrminAME, rtamem2, namrt2

    def step3_yieldmethod(self, TipoCarga: str, nbolt_tension: int, ho: float, hi: float, hi2: float, hi3: float,
                          bpl: float, g: float, pfo: float, c: float, tfv: float, p12_Outer: float, p23: float,
                          Fyp: float, Mr: float, Mfy: float, tp: float, boltgrade: str, pfi: float, db: float,
                          dh: float, dev: float, Fnt: float, Mf: float) -> Tuple:
        num = 0.5988023952095809 if TipoCarga == "ASD" else 0.9
        Y = self.Yp(nbolt_tension, ho, hi, hi2, hi3, bpl, g, pfo, c, tfv, p12_Outer, p23)

        treq_ep = math.sqrt(1.11 * Mr / (num * (Fyp / 1e6) * Y)) if (num * (Fyp / 1e6) * Y) > 0 else 0.0
        tmin_ep = math.sqrt(Mfy / (num * (Fyp / 1e6) * Y)) if (num * (Fyp / 1e6) * Y) > 0 else 0.0
        Mpl = num * Fyp / 1000.0 * (tp ** 2) * Y / 1000.0

        if (tp > treq_ep) or (tp < tmin_ep):
            mensaje3 = "End plate thickness by yield line method"
            ratio3 = Mfy / Mpl if Mpl != 0 else 0.0
            return Y, treq_ep, tmin_ep, Mpl, 0.0, 0.0, 0.0, 0.0, 0.0, ratio3, mensaje3
        else:
            Pt, Qmax0, Qmaxi, Tb, Mq, ratio3, mensaje3 = self.step3_BoltRupturewPryingAction(
                TipoCarga, boltgrade, pfo, pfi, bpl, Fyp, tp, db, dh, dev, Fnt, self.h0_, self.h1_, self.h2_, self.h3_, Mf
            )
            return Y, treq_ep, tmin_ep, Mpl, Pt, Qmax0, Qmaxi, Tb, Mq, ratio3, mensaje3

    def step3_BoltRupturewPryingAction(self, TipoCarga: str, boltgrade: str, pfo: float, pfi: float, bpl: float, Fyp: float,
                                       tp: float, db: float, dh: float, dev: float, Fnt: float, d0: float, d1: float,
                                       d2: float, d3: float, Mf: float) -> Tuple:
        num = 113.0 if boltgrade == "A490" else 90.0
        num2, num3, num4 = pfo / 25.4, pfi / 25.4, bpl / 25.4
        num5 = Fyp * 145.0 / 1000.0
        num6, num7 = tp / 25.4, db / 25.4
        num8 = 3.682 * ((num6 / num7) ** 3) - 0.085
        num9 = min(num8, dev / 25.4)
        num10 = (bpl / 2.0 - dh) / 25.4

        num11 = ((num6 ** 2) * num5 * (0.85 * num4 / 2.0 + 0.8 * num10) + math.pi * (num7 ** 3) * num / 8.0) / (4.0 * num3)
        num12 = num11 * (num3 / num2)

        Pt = (num7 ** 2) * math.pi / 4.0 * num
        Qmaxi = num10 * (num6 ** 2) / (4.0 * num8) * math.sqrt(max(0.0, (num5 ** 2) - 3.0 * ((num11 / num10 / num6) ** 2)))
        Qmax0 = num10 * (num6 ** 2) / (4.0 * num9) * math.sqrt(max(0.0, (num5 ** 2) - 3.0 * ((num12 / num10 / num6) ** 2)))

        Pt *= 4.448
        Qmax0 *= 4.448
        Qmaxi *= 4.448
        Tb = 0.7 * Fnt / 0.75

        Mq = 2.0 * (Pt - Qmax0) * d0 + 2.0 * (Pt - Qmaxi) * d1 + 2.0 * Tb * d2
        Mq = max(Mq, 2.0 * (Pt - Qmax0) * d0 + 2.0 * Tb * d1 + 2.0 * Tb * d2)
        Mq = max(Mq, 2.0 * (Pt - Qmaxi) * d1 + 2.0 * Tb * d0 + 2.0 * Tb * d2)
        Mq = max(Mq, 2.0 * Tb * (d0 + d1 + d2))

        num17 = 0.5 if TipoCarga == "ASD" else 0.75
        Mq = num17 * Mq / 1000.0
        ratio3 = Mf / Mq if Mq != 0 else 0.0
        return Pt, Qmax0, Qmaxi, Tb, Mq, ratio3, "End plate thickness by bolt rupture with prying action"

    def Yp(self, nbolt_tension: int, ho: float, hi: float, hi2: float, hi3: float,
           bpl: float, g: float, pfo: float, c: float, tfv: float, p12_Outer: float, p23: float) -> float:
        result = 0.0
        num = 0.5 * math.sqrt(bpl * g)
        num2 = min(c - pfo - tfv, num)
        num3 = nbolt_tension - 4

        if num3 == 0:
            result = bpl / 2.0 * (hi * (1.0 / num2 + 1.0 / num) + ho / pfo - 0.5) + 2.0 / g * (hi * (num2 + num))
        elif num3 == 2:
            result = bpl / 2.0 * (hi / num2 + hi2 / num + ho / pfo - 0.5) + 2.0 / g * (hi * (num2 + 0.75 * p23) + hi2 * (num + 0.25 * p23)) + g / 2.0
        elif num3 == 4:
            if self.dev <= num:
                result = bpl / 2.0 * (ho * (1.0 / (2.0 * self.dev)) + hi * (1.0 / pfo) + hi2 * (1.0 / num2) + hi3 + 1.0 / num) + 2.0 / g * (ho * (self.dev + 3.0 * p23 / 4.0) + hi * (pfo + p23 / 4.0) + hi2 * (num2 + 3.0 * p23 / 4.0) + hi3 * (num + p23 / 4.0)) + g
            else:
                result = bpl / 2.0 * (ho * (1.0 / num) + hi * (1.0 / pfo) + hi2 * (1.0 / num2) + hi3 + 1.0 / num) + 2.0 / g * (ho * (num + p23 / 4.0) + hi * (pfo + 3.0 * p23 / 4.0) + hi2 * (num2 + p23 / 4.0) + hi3 * (num + 3.0 * p23 / 4.0)) + g
        return result

    def step4(self, TipoCarga: str, Fyp: float, bp: float, tp: float, Ff: float) -> Tuple:
        namrt4 = "Shear yielding subject to flange tension force"
        Ap = tp * bp
        num = 2.0 / 3.0 if TipoCarga == "ASD" else 1.0
        RnyAME = num * 0.6 * Fyp * Ap / 1000.0
        rtamem4 = Ff / 2.0 / RnyAME if RnyAME != 0 else 0.0
        return Ap, RnyAME, rtamem4, namrt4

    def step5(self, TipoCarga: str, Fup: float, bp: float, tp: float, dh: float, Ff: float) -> Tuple:
        namrt5 = "Shear rupture subject to flange tension force"
        Apn = tp * (bp - 2.0 * (dh + 2.0))
        num = 0.5 if TipoCarga == "ASD" else 0.75
        RnrAME = num * 0.6 * Fup * Apn / 1000.0
        rtamem5 = Ff / 2.0 / RnrAME if RnrAME != 0 else 0.0
        return Apn, RnrAME, rtamem5, namrt5

    def step6_yieldmethod(self, TipoCarga: str, nbolt_tension: int, ho: float, hi: float, hi2: float, hi3: float,
                          bfc: float, g: float, pfo: float, c: float, tfv: float, p12_Outer: float, p23: float,
                          t_s: float, ts2: float, Mr: float, fy: float, Mfy: float, tfc: float, boltgrade: str,
                          pfi: float, db: float, dh: float, dev: float, Fnt: float, Mf: float) -> Tuple:
        num = 0.5988023952095809 if TipoCarga == "ASD" else 0.9
        Y = self.Yp2(nbolt_tension, ho, hi, hi2, hi3, bfc, g, p23, c, pfo, tfv, t_s, ts2)

        treq_cf = math.sqrt(1.11 * Mr / (num * (fy / 1e6) * Y)) if (num * (fy / 1e6) * Y) > 0 else 0.0
        tmin_cf = math.sqrt(Mfy / (num * (fy / 1e6) * Y)) if (num * (fy / 1e6) * Y) > 0 else 0.0
        Mcf = num * fy / 1000.0 * (tfc ** 2) * Y / 1000.0

        if (tfc > treq_cf) or (tfc < tmin_cf):
            mensaje6 = "Column flange thickness by yield line method"
            ratio6 = Mfy / Mcf if Mcf != 0 else 0.0
            return Y, treq_cf, tmin_cf, Mcf, 0.0, 0.0, 0.0, 0.0, 0.0, ratio6, mensaje6
        else:
            Pt, Qmax0, Qmaxi, Tb, Mq, ratio6, mensaje6 = self.step6_BoltRupturewPryingAction(
                TipoCarga, boltgrade, pfo, pfi, bfc, fy, tfc, db, dh, dev, Fnt, self.h0_, self.h1_, self.h2_, self.h3_, Mf
            )
            return Y, treq_cf, tmin_cf, Mcf, Pt, Qmax0, Qmaxi, Tb, Mq, ratio6, mensaje6

    def step6_BoltRupturewPryingAction(self, TipoCarga: str, boltgrade: str, pfo: float, pfi: float, bfc: float, Fy: float,
                                       tfb: float, db: float, dh: float, dev: float, Fnt: float, d0: float, d1: float,
                                       d2: float, d3: float, Mf: float) -> Tuple:
        num = 113.0 if boltgrade == "A490" else 90.0
        num2, num3, num4 = pfo / 25.4, pfi / 25.4, bfc / 25.4
        num5 = Fy * 145.0 / 1000.0
        num6, num7 = tfb / 25.4, db / 25.4
        num8 = 3.682 * ((num6 / num7) ** 3) - 0.085
        num9 = min(num8, dev / 25.4)
        num10 = (bfc / 2.0 - dh) / 25.4

        num11 = ((num6 ** 2) * num5 * (0.85 * num4 / 2.0 + 0.8 * num10) + math.pi * (num7 ** 3) * num / 8.0) / (4.0 * num3)
        num12 = num11 * (num3 / num2)

        Pt = (num7 ** 2) * math.pi / 4.0 * num
        Qmaxi = num10 * (num6 ** 2) / (4.0 * num8) * math.sqrt(max(0.0, (num5 ** 2) - 3.0 * ((num11 / num10 / num6) ** 2)))
        Qmax0 = num10 * (num6 ** 2) / (4.0 * num9) * math.sqrt(max(0.0, (num5 ** 2) - 3.0 * ((num12 / num10 / num6) ** 2)))

        Pt *= 4.448
        Qmax0 *= 4.448
        Qmaxi *= 4.448
        Tb = 0.7 * Fnt / 0.75

        Mq = 2.0 * (Pt - Qmax0) * d0 + 2.0 * (Pt - Qmaxi) * d1 + 2.0 * Tb * d2
        Mq = max(Mq, 2.0 * (Pt - Qmax0) * d0 + 2.0 * Tb * d1 + 2.0 * Tb * d2)
        Mq = max(Mq, 2.0 * (Pt - Qmaxi) * d1 + 2.0 * Tb * d0 + 2.0 * Tb * d2)
        Mq = max(Mq, 2.0 * Tb * (d0 + d1 + d2))

        num17 = 0.5 if TipoCarga == "ASD" else 0.75
        Mq = num17 * Mq / 1000.0
        ratio6 = Mf / Mq if Mq != 0 else 0.0
        return Pt, Qmax0, Qmaxi, Tb, Mq, ratio6, "Column flange thickness by bolt rupture with prying action"

    def Yp2(self, nbolt_tension: int, ho: float, hi: float, hi2: float, hi3: float,
            bfc: float, g: float, p23: float, c: float, pfo: float, tfv: float, t_s: float, ts2: float) -> float:
        result = 0.0
        num = 0.5 * math.sqrt(bfc * g)
        num2 = pfo + tfv / 2.0 - t_s / 2.0
        num3 = min(c - num2 - t_s, num)
        num4 = nbolt_tension - 4

        if num4 == 0:
            if t_s != 0.0:
                result = bfc / 2.0 * (hi * (1.0 / num + 1.0 / num3) + ho * (1.0 / num + 1.0 / num2)) + 2.0 / g * (hi * (num + num3) + ho * (num + num2))
            else:
                result = bfc / 2.0 * (hi / num + ho / num) + 2.0 / g * (hi * (num + 0.75 * c) + ho * (num + 0.25 * c) + (c ** 2) / 2.0) + g / 2.0
        elif num4 == 2:
            if t_s == 0.0:
                result = bfc / 2.0 * (ho / num + hi2 / num) + 2.0 / g * (ho * (num + c + 0.5 * p23) + hi2 * (num + 0.5 * p23)) + g / 2.0
            elif ts2 == 0.0:
                result = bfc / 2.0 * (hi / num3 + hi2 / num + ho * (1.0 / num2 + 1.0 / num)) + 2.0 / g * (hi * (num3 + 0.75 * p23) + hi2 * (num + 0.25 * p23) + ho * (num + num2)) + g / 2.0
            else:
                num5 = (p23 - ts2) / 2.0
                result = bfc / 2.0 * (hi2 / num + hi2 / num5) + 2.0 / g * (hi2 * num + hi2 * num5) + bfc / 2.0 * (ho / num + ho / num2) + 2.0 / g * (ho * num + ho * num2) + bfc / 2.0 * (hi / num3 + hi / num5) + 2.0 / g * (hi * num3 + hi * num5)
        elif num4 == 4:
            if t_s != 0.0:
                result = bfc / 2.0 * (ho * (1.0 / num) + hi * (1.0 / num2) + hi2 * (1.0 / num3) + hi3 * (1.0 / num)) + 2.0 / g * (ho * (num + self.pb / 4.0) + hi * (num2 + 3.0 * self.pb / 4.0) + hi2 * (num3 + self.pb / 4.0) + hi3 * (num + 3.0 * self.pb / 4.0) + (self.pb ** 2)) + g
            else:
                result = bfc / 2.0 * (ho * (1.0 / num) + hi3 * (1.0 / num)) + 2.0 / g * (ho * (self.pb + c / 2.0 + num) + hi * (self.pb / 2.0 + c / 4.0) + hi2 * (self.pb / 2.0 + c / 2.0) + hi3 * num) + g / 2.0
        return result

    def step7(self, TipoCarga: str, tfV: float, rV: float, tfP: float, twP: float, rP: float, hP: float,
              dP: float, tp: float, wf: float, Fy: float, Ff: float, E: float, esp_t_s: float, connectiontop: str) -> Tuple:
        namrt7 = "Column web crippling, yielding & buckling"
        num = 2.0 / 3.0 if TipoCarga == "ASD" else 1.0
        k1 = 1.0 if connectiontop == "YES" else 2.0
        N = tfV + 2.0 * wf
        kc = tfP + rP

        Br1 = num * twP * (k1 * (3.0 * kc + tp + 0.5 * N)) * Fy / 1000.0
        if connectiontop == "YES" and N < (tfV + rV):
            N = tfV + rV
        BrminAME = Br1

        num = 0.5 if TipoCarga == "ASD" else 0.75
        if connectiontop == "YES" and (N / hP) > 0.2:
            k3 = (4.0 * N / hP - 0.2) * ((twP / tfP) ** 1.5)
        else:
            k3 = 3.0 * (N / hP) * ((twP / tfP) ** 1.5)

        Br2 = num * 0.4 * (twP ** 2) * k1 * (1.0 + k3) * math.sqrt(E * Fy * tfP / twP) / 1000.0
        BrminAME = min(BrminAME, Br2)

        num = 0.5988023952095809 if TipoCarga == "ASD" else 0.9
        Br3 = k1 * num * 12.0 * (twP ** 3) * math.sqrt(E * Fy) / dP / 1000.0
        BrminAME = min(BrminAME, Br3)

        rtamem7 = Ff / BrminAME if BrminAME != 0 else 0.0
        if esp_t_s != 0.0:
            rtamem7 = 0.0
        return N, k1, kc, Br1, k3, Br2, Br3, BrminAME, rtamem7, namrt7

    def step8(self, TipoCarga: str, Ac: float, hP: float, twP: float, tfP: float, bp: float,
              hV: float, Ff: float, Fy: float, esp_w_d_p: float, Pr: float) -> Tuple:
        namrtm8 = "Column panel zone web shear"
        num = 0.5988023952095809 if TipoCarga == "ASD" else 0.9
        num2 = Ac * Fy / 1000.0
        if TipoCarga == "ASD":
            num2 *= 0.6

        factor_pr = 1.0 if Pr <= 0.75 * num2 else (1.9 - 1.2 * Pr / num2)
        VrAME = num * 0.6 * Fy * hP * twP * (1.0 + 3.0 * bp * (tfP ** 2) / (hV * hP * twP)) * factor_pr / 1000.0

        rtamem8 = Ff / VrAME if VrAME != 0 else 0.0
        if esp_w_d_p != 0.0:
            rtamem8 = 0.0
        return VrAME, rtamem8, namrtm8

    def step9(self, TipoCarga: str, BrminAME: float, Ff: float, bP: float, twP: float, rP: float,
              hP: float, tfP: float, tfV: float, esp_t_s: float, connectiontop: str, E: float, Fyp: float) -> Tuple:
        namrtm9 = "Transverse stiffeners"
        Ffts = Ff - BrminAME
        # Si la fuerza requerida es <= 0 o no hay espesor asignado, no hace falta rigidizador (medidas a 0)
        if esp_t_s == 0.0 or Ffts <= 0.0:
            log.debug(f"[LOG CALCULADOR] Ratio negativo/nulo en '{namrtm9}' (Ffts={Ffts:.2f} kN). Anulando medidas -> bs=0.0, clip=0.0, tts_min=0.0, A_seff=0.0")
            return max(Ffts, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, namrtm9

        bs = math.floor((bP - twP) / 2.0)
        bs = bs - (bs % 5)

        clip = math.ceil(rP)
        clip = clip + ((5 - (clip % 5)) % 5)

        num4 = hP - 2.0 * tfP
        tts_min = max(tfV / 2.0, (bs - clip) / 15.0)
        num9 = 1000.0 * Ffts / (0.54 * Fyp * (num4 - 2.0 * clip) * 2.0)
        tts_min = max(tts_min, num9)

        num10 = 12.0 if connectiontop == "YES" else 25.0
        num11 = 2.0 * bs + num10 * twP
        num12 = (1.0 / 12.0) * num11 * (esp_t_s ** 3)
        A_seff = num11 * esp_t_s

        num13 = hP - 2.0 * tfP - 2.0 * rP
        num15 = 0.75 * num13
        num16 = math.sqrt(num12 / A_seff) if A_seff != 0 else 1.0
        KL = num15 / num16
        Fe = (math.pi ** 2) * E / (KL ** 2)

        if KL <= 25.0:
            Fcr = Fyp
        elif Fyp / Fe < 2.25:
            Fcr = (0.658 ** (Fyp / Fe)) * Fyp
        else:
            Fcr = 0.877 * Fe

        num17 = 0.5988023952095809 if TipoCarga == "ASD" else 0.9
        CrAME = num17 * Fcr * A_seff / 1000.0
        rtamem9 = Ffts / CrAME if CrAME != 0 else 0.0
        return Ffts, bs, clip, tts_min, A_seff, KL, Fe, Fcr, CrAME, rtamem9, namrtm9

    def step10(self, TipoCarga: str, VrAME: float, Ff: float, Ffts: float, clip_s: float, esp_w_d_p: float,
               De: float, tfP: float, rP: float, tp: float, hP: float, Fyp: float, esp_t_s: float, Fillet_Weld_Type: bool) -> Tuple:
        namrtm10 = "Web doubler plate"
        num = 0.5988023952095809 if TipoCarga == "ASD" else 0.9
        Vfdp = Ff - VrAME
        # Si la solicitación sobre la chapa de refuerzo es <= 0 o no existe espesor asignado, se devuelven dimensiones a 0
        if esp_w_d_p == 0.0 or Vfdp <= 0.0:
            log.debug(f"[LOG CALCULADOR] Ratio negativo/nulo en '{namrtm10}' (Vfdp={Vfdp:.2f} kN). Anulando medidas -> Ldp=0.0, bdp=0.0, tdp_min=0.0, VrdpAME=0.0")
            return max(Vfdp, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, namrtm10

        Ldp = round((De + 2.0 * (3.0 * (tfP + rP + tp))) / 10.0) * 10.0
        bdp = math.floor(hP - 2.0 * (tfP + rP))
        bdp = bdp - (bdp % 5)

        tdp_min = math.sqrt(Fyp) * (hP - 2.0 * (tfP + rP)) / 1001.7584539199058
        if Fillet_Weld_Type:
            num4 = 3.175 if rP < 7.9375 else (4.7625 if rP < 12.7 else (7.9375 if rP >= 20.6375 else 6.35))
            tdp_min = max(tdp_min, rP - num4)

        if esp_t_s != 0.0 and Ffts > 0.0:
            num5 = hP - 2.0 * tfP
            if tdp_min < (1000.0 * Ffts / (num * 0.6 * Fyp * hP * 2.0)):
                tdp_min = 1000.0 * Ff / (0.54 * Fyp * hP * 2.0)
            elif tdp_min < (1000.0 * Ffts / (num * 0.6 * Fyp * (num5 - 2.0 * clip_s) * 2.0)):
                tdp_min = 1000.0 * Ffts / (num * 0.6 * Fyp * (num5 - 2.0 * clip_s) * 2.0)

        VrdpAME = num * 0.6 * hP * esp_w_d_p * Fyp / 1000.0
        rtamem10 = Vfdp / VrdpAME if (esp_w_d_p > tdp_min and VrdpAME != 0) else 1.0
        return Vfdp, Ldp, bdp, tdp_min, VrdpAME, rtamem10, namrtm10

    def step11(self, TipoCarga: str, angle: float, De: float, hV: float, dV: float, tfV: float,
               twV: float, rV: float, Mf: float, esp_h_w_s: float, Fy: float, E: float) -> Tuple:
        namrtm11 = "Beam web yielding, crippling & buckling"
        if angle == 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, namrtm11

        rad = math.radians(angle)
        num = Mf * 1000.0 / (De - tfV)
        C1 = num / math.tan(rad)
        C3 = Mf * 1000.0 / (hV - tfV)
        C2 = C3 - num
        num2 = 2.0 / 3.0 if TipoCarga == "ASD" else 1.0
        N = tfV / math.cos(rad)

        Br1 = num2 * twV * (5.0 * (tfV + rV) + N) * Fy / 1000.0
        BrminAME = Br1

        num2 = 0.5 if TipoCarga == "ASD" else 0.75
        num4 = 3.0 * (N / hV) * ((twV / tfV) ** 1.5)
        Br2 = num2 * 0.8 * (twV ** 2) * (1.0 + num4) * math.sqrt(E * Fy * tfV / twV) / 1000.0
        BrminAME = min(BrminAME, Br2)

        num2 = 0.5988023952095809 if TipoCarga == "ASD" else 0.9
        Br3 = num2 * 24.0 * (twV ** 3) * math.sqrt(E * Fy) / dV / 1000.0
        BrminAME = min(BrminAME, Br3)

        rtamem11 = C1 / BrminAME if (BrminAME != 0 and esp_h_w_s == 0.0) else 0.0
        return C1, C2, C3, N, Br1, Br2, Br3, BrminAME, rtamem11, namrtm11

    def step12(self, TipoCarga: str, BrminAME_beam: float, C1: float, bbV: float, twV: float,
               rV: float, hV: float, tfV: float, esp_h_w_s: float, E: float, Fyp: float) -> Tuple:
        namrtm12 = "Haunch web stiffeners"
        C1wsAME = C1 - BrminAME_beam
        # Si la fuerza sobre el rigidizador de la cartela es <= 0 o no hay espesor asignado, se eliminan sus dimensiones
        if esp_h_w_s == 0.0 or C1wsAME <= 0.0:
            log.debug(f"[LOG CALCULADOR] Ratio negativo/nulo en '{namrtm12}' (C1wsAME={C1wsAME:.2f} kN). Anulando medidas -> bs=0.0, clip=0.0, tts_min=0.0, A_seff=0.0")
            return max(C1wsAME, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, namrtm12

        bs = math.floor((bbV - twV) / 2.0)
        bs = bs - (bs % 5)
        clip = math.ceil(rV)
        clip = clip + ((5 - (clip % 5)) % 5)

        num4 = hV - 2.0 * tfV
        num5 = (1.0 / 12.0) * bs * (esp_h_w_s ** 3)
        num6 = math.sqrt(num5 / (bs * esp_h_w_s))
        num7 = bs - clip
        num8 = 2.0 * num7 * esp_h_w_s

        num9 = (1.0 / 12.0) * ((2.0 * num7 + twV) ** 3) * esp_h_w_s
        A_seff = num8 + 25.0 * (twV ** 2)
        num11 = math.sqrt(num9 / A_seff) if A_seff != 0 else 1.0
        num14 = 0.75 * (hV - 2.0 * tfV - 2.0 * clip)

        Fe = (math.pi ** 2) * E / ((num14 / num11) ** 2)
        tts_min = max(tfV / 2.0, bs / 16.0)

        if (num14 / num11) <= 25.0:
            Fcr = Fyp
        elif Fyp / Fe < 2.25:
            Fcr = (0.658 ** (Fyp / Fe)) * Fyp
        else:
            Fcr = 0.877 * Fe

        num15 = 0.5 if TipoCarga == "ASD" else 0.9
        CrAME = num15 * Fcr * A_seff / 1000.0
        rtamem12 = C1wsAME / CrAME if CrAME != 0 else 0.0
        return C1wsAME, bs, clip, tts_min, A_seff, num14 / num11, Fe, Fcr, CrAME, rtamem12, namrtm12

    def step13(self, TipoCarga: str, tp: float, hV: float, tfV: float, rV: float, twV: float,
               ww: float, FuWeld: float, Vf: float, Fyp: float, Fy: float) -> Tuple:
        namrtm13 = "Weld - Beam web to end plate weld "
        Lw = 2.0 * (hV - 2.0 * tfV - 2.0 * rV)
        Aw = Lw * ww / math.sqrt(2)
        num = 0.5 if TipoCarga == "ASD" else 0.75

        Vr1 = 0.6 * num * (hV - 2.0 * tfV - 2.0 * rV) * twV * Fy / 1000.0
        Vr2 = 0.6 * num * Lw * tp * Fyp / 1000.0
        Vr3 = 0.6 * num * Aw * FuWeld / 1000.0
        Vr_min = min(Vr1, Vr2, Vr3)

        rtamem13 = Vf / Vr_min if Vr_min != 0 else 0.0
        rtamem13 = 0.0
        return Lw, Aw, Vr1, Vr2, Vr3, Vr_min, rtamem13, namrtm13

    def step14(self, TipoCarga: str, Fy: float, bbV: float, tfV: float, twV: float, rV: float,
               t_ep: float, Ff: float, wf: float, FuWeld: float, Fu: float, Fup: float) -> Tuple:
        namrtm14 = "Weld - Beam flange to end plate weld "
        Lf = 2.0 * bbV - twV - 2.0 * rV
        Af = Lf * wf / math.sqrt(2)
        num = 0.5 if TipoCarga == "ASD" else 0.75

        Vr1 = 0.6 * num * Lf * tfV * Fy / 1000.0
        Vr2 = 0.6 * num * Lf * t_ep * Fup / 1000.0
        Vr3 = 0.6 * num * Af * FuWeld * 1.5 / 1000.0
        Vr_min = min(Vr1, Vr2, Vr3)

        rtamem14 = Ff / Vr3 if Vr3 != 0 else 0.0
        rtamem14 = 0.0
        wf_min = Ff * wf / Vr3 if Vr3 != 0 else 0.0
        return Lf, Af, Vr1, Vr2, Vr3, Vr_min, wf_min, rtamem14, namrtm14

    def step15(self, TipoCarga: str, tfV: float, twV: float, wh: float, Hh: float, angle: float,
               FuWeld: float, Fu: float, C2: float) -> Tuple:
        namrtm15 = "Weld - Haunch web to beam weld"
        Lw = 2.0 * (Hh * math.tan(math.radians(angle)) - tfV)
        Aw = Lw * wh / math.sqrt(2)
        num = 0.5 if TipoCarga == "ASD" else 0.75

        Vr1 = 0.6 * num * 2.0 * Lw * tfV * Fu / 1000.0
        Vr2 = 0.6 * num * Lw * twV * Fu / 1000.0
        Vr3 = 0.6 * num * Aw * FuWeld / 1000.0
        Vr_min = min(Vr1, Vr2, Vr3)

        rtamem15 = C2 / Vr_min if Vr_min != 0 else 0.0
        rtamem15 = 0.0
        return Lw, Aw, Vr1, Vr2, Vr3, Vr_min, rtamem15, namrtm15

    def StiffenerToColumnWebWeld(self, fF: float, TipoCarga: str, wscw: float, h_c: float, b_c: float,
                                 tf_c: float, tw_c: float, r_c: float, t_s: float, fuweld: float,
                                 fu: float, fy: float, fup: float) -> Tuple[float, str]:
        left = 0.5 if TipoCarga == "ASD" else 0.75
        num = h_c - 2.0 * tf_c - 2.0 * r_c
        num2 = (b_c - tw_c - 2.0 * r_c) / 2.0
        num3 = (left * num2 * t_s * fu) / 1000.0
        num4 = (left * 0.6 * num * t_s * fy) / 1000.0
        num5 = (left * 0.6 * num * tw_c * self.fyp) / 1000.0
        num6 = min(num3, num4, num5)

        num7 = (left * 0.6 * 2.0 * num * wscw * 0.707 * fuweld) / 1000.0
        ratio = num6 / num7 if num7 != 0 else 0.0
        ratio = 0.0
        return ratio, "Weld between transverse stiffener and column web"

    def SlipCritical(self, TipoCarga: str, Tb: float, Tu: float, Vb: float) -> Tuple[float, str]:
        num, num2, num3, num4 = 0.3, 1.13, 1.0, 1.0
        left = 2.0 / 3.0 if TipoCarga == "ASD" else 1.0
        num5 = 1.0 - (Tu / (left * num2 * Tb)) if (left * num2 * Tb) != 0 else 1.0
        right = left * num * num2 * num3 * Tb * num4 * num5
        rtamem16 = Vb / right if right != 0 else 0.0
        return rtamem16, "Slip Resistance"
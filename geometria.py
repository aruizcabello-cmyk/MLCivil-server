# geometria.py

import math
from externos import tablapernos, tablaplaca

def recalcular_geometria_fallos(inp, pfo_pred, pfi_pred, pb_pred, dev_pred):
    """
    Recalcula las distancias entre tornillos según el algoritmo ConnectionDesign de C#
    únicamente cuando la comprobación geométrica inicial no cumple.
    """
    tbl_pernos = tablapernos()
    tbl_pernos.nombretornillo = inp.BoltName_ep
    
    tbl_placa = tablaplaca()
    tbl_placa.nombreplaca = inp.ThicknessName_t_ep
    
    max_edge = tbl_placa.disbormaxmet
    max_spacing = tbl_placa.pitchmaxmet
    
    h_b = inp.viga.h if inp.viga else 0.0
    tf_b = inp.viga.tf if inp.viga else 0.0
    
    if inp.AnguloCantonera != 0.0:
        h_h = inp.FactorCantoCantonera * h_b / math.sin(math.radians(inp.AnguloCantonera))
    else:
        h_h = 0.0
        
    num_canto = h_b + h_h
    num3 = 0.35
    text_bolt = inp.BoltName_ep
    
    flag2 = True
    num8 = 2.0
    flag_m20_limit = False
    
    min_spacing = 0.0
    min_enter = 0.0
    pb = pb_pred
    
    while flag2:
        tbl_pernos.nombretornillo = text_bolt
        min_spacing = tbl_pernos.pitchminmet
        min_enter = tbl_pernos.dislibmet
        
        num4 = math.ceil(min_spacing / 5.0) * 5.0
        num5 = math.ceil(min_enter)
        pb = num4
        
        num6 = num_canto + 2.0 * num5
        num7 = num5 + tf_b + num5 + num4
        
        if (num7 / num6) < num3:
            num8 = 3.0
            flag2 = False
            continue
            
        num7 = num5 + tf_b + num5
        if (num7 / num6) < num3:
            num8 = 2.0
            flag2 = False
        elif text_bolt.upper() == "M20":
            flag_m20_limit = True
            flag2 = False
        else:
            text_bolt = tbl_pernos.NombreTornilloAnterior(text_bolt)

    arr = [0.0] * 12
    pfi_calc = pfi_pred
    pfo_calc = pfo_pred

    if not flag_m20_limit and inp.AnguloCantonera == 0.0:
        if num8 == 2.0:
            arr[1] = round((h_b - 2.0 * num5 - 2.0 * tf_b) / 5.0) * 5.0
            pfi_calc = (h_b - 2.0 * tf_b - arr[1]) / 2.0
            arr[0] = round((pfi_calc + num5 + tf_b) / 5.0) * 5.0
            arr[2] = arr[0]
            pfo_calc = arr[0] - tf_b - pfi_calc
        elif num8 == 3.0:
            arr[2] = round((h_b - 2.0 * num5 - 2.0 * num4 - 2.0 * tf_b) / 5.0) * 5.0
            arr[1] = num4
            arr[3] = num4
            pfi_calc = (h_b - 2.0 * tf_b - 2.0 * arr[1] - arr[2]) / 2.0
            arr[0] = round((pfi_calc + num5 + tf_b) / 5.0) * 5.0
            arr[4] = arr[0]
            pfo_calc = arr[0] - tf_b - pfi_calc

    return {
        "pfo": pfo_calc,
        "pfi": pfi_calc,
        "pb": pb,
        "p_separations": arr,
        "bolt_used": text_bolt
    }
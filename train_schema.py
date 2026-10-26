"""Esquema del dataset de reentreno (subconjunto de `data/data_final.csv`).

Se excluyen las columnas derivadas (`BEAM_*`, `COLUMN_*`, `fy/fu/fv`) y las que el modelo
derivara/adaptara despues (`standard`, `n_alma`, `n_cantonera`).
`TRAIN_COLUMNS` es el orden exacto del parquet unificado; el tipo de cada columna se usa
para coercionar al traducir.
"""

TRAIN_COLUMNS = [
    # demanda
    "Capacity_Moment_Percentage", "Capacity_Moment",
    "Capacity_Shear_Percentage", "Capacity_Shear",
    "Capacity_Axial_Percentage", "Capacity_Axial",
    # diseno (chapa / pasos / rigidizadores)
    "g", "tws", "Bp", "Lp", "tp", "Lh", "e", "d",
    "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12",
    "ts", "tbs", "tst", "Hole",
    # identificacion / metadatos
    "Unicode", "BEAM", "COLUMN", "DrawingCode", "Bolt", "SlipCritical",
    "Normativas", "AceroEstructural", "EstandarTornillos", "Source", "National_Annex",
]

TEXT_COLUMNS = {
    "Unicode", "BEAM", "COLUMN", "DrawingCode",
    "Normativas", "AceroEstructural", "EstandarTornillos", "Source", "National_Annex",
}

BOOL_COLUMNS = {"SlipCritical"}

NUMERIC_COLUMNS = [
    c for c in TRAIN_COLUMNS if c not in TEXT_COLUMNS and c not in BOOL_COLUMNS
]

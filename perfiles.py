"""
Equivalente de ConsultaPerfilesSQLServer(object[] ArgsBBDD) del C#.

Lo que hace el C# original:
  1. Mira la cache en memoria (DataTable `dt`, clave = nombre del perfil).
  2. Si no esta, abre SqlConnection y llama a
        bdperfiles.PerfilesBD.GetDatosBDPerfilesOptimizada(
            conexion, ProntuarioNombrePerfiles, ["Arcelor"], NombreFuente,
            perfil, campos[9], ref valores)
     con los campos, en este orden:
        CANTO, ANCHO, ESPESORALA, ESPESORALMA, RADIOACUERDO, AREA,
        MODURESISPLATICO, PRMIN, PRMAX
     y multiplica cada valor por su FACTOR (interBDperfiles.<campo>.FACTOR;
     MODURESISPLATICO siempre x1.0).
  3. BEAM  -> h_b,b_b,tf_b,tw_b,r_b,A_b,Z_b
     COLUMN-> h_c,b_c,tf_c,tw_c,r_c,A_c,Z_c,Prmin,Prmax

El SELECT esta DENTRO de la DLL `bdperfiles` (no en el .cs), por eso
`PerfilesBD.GetDatosBDPerfilesOptimizada` es un stub.

Diferencia deliberada: el C# nunca cierra la SqlConnection; aqui se cierra siempre.
"""
from __future__ import annotations

import os
from contextlib import closing
from typing import Dict, List, Optional, Sequence

from modelos import Perfil

# Cadena de conexion: el C# la tiene fija y con autenticacion de Windows
# (Server='TR-SQLISO-YEC01\AMS';Database='PerfilesBD';Trusted_Connection=True).
# Desde Linux/contenedor lo normal es un usuario SQL de solo lectura via variable de entorno.
CADENA_POR_DEFECTO = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=TR-SQLISO-YEC01\\AMS;DATABASE=PerfilesBD;"
    "Trusted_Connection=yes;TrustServerCertificate=yes;"
)

CAMPOS = ["CANTO", "ANCHO", "ESPESORALA", "ESPESORALMA", "RADIOACUERDO",
          "AREA", "MODURESISPLATICO", "PRMIN", "PRMAX"]


class interBDperfiles:
    """NAME/FACTOR de cada campo. Aqui NAME = nombre de columna y FACTOR = 1.0 por defecto.
    VERIFICAR contra la clase real: si la BD guarda pulgadas/mm2, hay que poner el factor."""
    FACTOR: Dict[str, float] = {c: 1.0 for c in CAMPOS}
    NAME: Dict[str, str] = {c: c for c in CAMPOS}


class PerfilesBD:
    """Stub de bdperfiles.PerfilesBD (DLL externa)."""

    def GetDatosBDPerfilesOptimizada(self, conexion, prontuario_nombre_perfiles: str,
                                     prontuarios: Sequence[str], nombre_fuente: str,
                                     perfil: str, campos: Sequence[str]) -> List[float]:
        """Debe devolver los 9 valores (sin factor) en el orden de CAMPOS."""
        raise NotImplementedError(
            "El SELECT esta en la DLL bdperfiles. Implementar aqui la consulta o "
            "pasar los datos del perfil directamente (ConexionInput.viga / .pilar)."
        )


class RepositorioPerfiles:
    """Una instancia por proceso (cache compartida), no una por peticion."""

    def __init__(self, nombre_fuente: str = "", prontuario_nombre_perfiles: str = "",
                 cadena_conexion: Optional[str] = None, bd: Optional[PerfilesBD] = None) -> None:
        self.nombre_fuente = nombre_fuente
        self.prontuario_nombre_perfiles = prontuario_nombre_perfiles
        self.cadena = cadena_conexion or os.environ.get("PERFILESBD_CONN", CADENA_POR_DEFECTO)
        self.bd = bd or PerfilesBD()
        self._cache: Dict[str, Perfil] = {}          # equivalente a `dt`

    def consultar(self, nombre: str, es_columna: bool) -> tuple[Perfil, str]:
        """Devuelve (perfil, origen) con origen 'CACHE' o 'BASE DE DATOS'."""
        if nombre in self._cache:
            return self._cache[nombre], "CACHE"

        import pyodbc  # import perezoso: no hace falta si se pasan los datos directos

        with closing(pyodbc.connect(self.cadena)) as conn:
            v = self.bd.GetDatosBDPerfilesOptimizada(
                conn, self.prontuario_nombre_perfiles, ["Arcelor"],
                self.nombre_fuente, nombre, CAMPOS)

        f = interBDperfiles.FACTOR
        vals = [float(v[i]) * (1.0 if CAMPOS[i] == "MODURESISPLATICO" else f[CAMPOS[i]])
                for i in range(9)]
        perfil = Perfil(nombre=nombre, h=vals[0], b=vals[1], tf=vals[2], tw=vals[3],
                        r=vals[4], A=vals[5], Z=vals[6], Prmin=vals[7], Prmax=vals[8])
        self._cache[nombre] = perfil
        return perfil, "BASE DE DATOS"

"""Cliente de terminal: manda cada fila de un CSV a POST /scenarios y guarda los resultados (CSV con las columnas CODAME + JSON completo).

Solo usa la libreria estandar (no hace falta instalar nada).

Uso:
  python enviar_escenarios.py entrada.csv
  python enviar_escenarios.py entrada.csv --url http://localhost:8080 --salida resultado.csv
  python enviar_escenarios.py entrada.csv --modelo V11 --incremento 0.05 --max-pasos 40

CSV de entrada (separador , o ; detectado solo; coma decimal admitida). Cabecera minima:
  UNICODE,BEAM,COLUMN,ratio_moment,ratio_shear,ratio_axial
y opcionales: Normativas, grado_acero, clase_tornillo. Una fila = una conexion.
"""
import argparse
import csv
import json
import sys
import urllib.error
import urllib.request


def leer_csv(ruta):
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        muestra = f.read(4096)
        f.seek(0)
        try:
            dialecto = csv.Sniffer().sniff(muestra, delimiters=",;\t")
        except csv.Error:
            dialecto = csv.excel
        filas = [{k.strip(): (v or "").strip() for k, v in r.items() if k}
                 for r in csv.DictReader(f, dialect=dialecto)]
    return [f for f in filas if any(f.values())]


def llamar(url, cuerpo, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(cuerpo).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detalle = e.read().decode("utf-8", "replace")[:600]
        raise RuntimeError(f"HTTP {e.code}: {detalle}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"no se pudo conectar con {url}: {e.reason}") from None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--url", default="http://localhost:8080", help="servidor (por defecto %(default)s)")
    ap.add_argument("--salida", default="escenarios_resultado.csv")
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--incremento", type=float, default=0.05)
    ap.add_argument("--max-pasos", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=600, help="segundos por fila")
    a = ap.parse_args()

    filas = leer_csv(a.csv)
    if not filas:
        sys.exit("El CSV no tiene filas.")
    endpoint = a.url.rstrip("/") + "/scenarios"
    salida, columnas, errores, completos = [], ["Unicode origen", "Escenario"], 0, []

    for i, fila in enumerate(filas, 1):
        uid = fila.get("UNICODE") or fila.get("Unicode") or f"fila{i}"
        print(f"[{i}/{len(filas)}] {uid} ...", end=" ", flush=True)
        cuerpo = {"row": fila, "incremento": a.incremento, "max_pasos": a.max_pasos}
        if a.modelo:
            cuerpo["model"] = a.modelo
        try:
            res = llamar(endpoint, cuerpo, a.timeout)
        except RuntimeError as e:
            errores += 1
            print("ERROR")
            print(f"    {e}", file=sys.stderr)
            salida.append({"Unicode origen": uid, "Escenario": "ERROR", "Ratio pésimo": str(e)})
            continue
        print(f"ok (parada: {res['parada']})")
        print(res["tabla_markdown"])
        for aviso in res["avisos"]:
            print(f"  ! {aviso}")
        print()
        completos.append({"Unicode origen": uid, "respuesta": res})
        meta = {e["id"]: e for e in res["escenarios"]}
        # Una linea por escenario encontrado: las columnas CODAME de /predict (mismos nombres)
        # + Escenario, Paso, Factor carga, cumple, ratio_pesimo.
        for row, chk in zip(res["rows"], res["checks"]):
            m = meta.get(str(row.get("Escenario", ""))[:2], {})
            linea = {"Unicode origen": uid, "Escenario": row.get("Escenario"), "Paso": m.get("paso"),
                     "Factor carga": m.get("factor_carga"), "cumple": chk.get("cumple"),
                     "ratio_pesimo": chk.get("ratio_pesimo"),
                     "descripcion_ratio_pesimo": chk.get("descripcion_ratio_pesimo"),
                     **{k: v for k, v in row.items() if k != "Escenario"}}
            salida.append(linea)
            columnas += [c for c in linea if c not in columnas]
        for e in res["escenarios"]:
            if not e["encontrado"]:
                salida.append({"Unicode origen": uid, "Escenario": f"{e['id']} - {e['escenario']}",
                               "cumple": "No encontrado", "descripcion_ratio_pesimo": e.get("motivo")})
                columnas += [c for c in ("cumple", "descripcion_ratio_pesimo") if c not in columnas]

    if errores and "Ratio pésimo" not in columnas:
        columnas.append("Ratio pésimo")
    with open(a.salida, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel abre bien los acentos
        w = csv.DictWriter(f, fieldnames=columnas, delimiter=",", extrasaction="ignore")
        w.writeheader()
        w.writerows(salida)
    ruta_json = a.salida.rsplit(".", 1)[0] + ".json"
    with open(ruta_json, "w", encoding="utf-8") as f:       # respuesta completa (rows, checks, rows_raw, historial...)
        json.dump(completos, f, ensure_ascii=False, indent=1)
    print(f"Guardado: {ruta_json}")
    print(f"Guardado: {a.salida}  ({len(filas) - errores} ok, {errores} con error)")
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    main()
"""
parse_enr3.py
--------------
Convierte el texto combinado de ENR 3 (generado por explore_pdf_batch.py,
el archivo all_text.txt) en los dos CSV que ya usa network.py:
 
    data/processed/waypoints.csv   (waypoint_id, lat, lon, tipo)
    data/processed/segments.csv    (from_id, to_id, airway_id)
 
Cómo funciona (basado en el formato real de ENR 3.1 Colombia):
  - Cada archivo de ruta trae una lista de puntos significativos marcados
    con "▲", seguidos de sus coordenadas en formato DD°MM'SS''N/S y
    DDD°MM'SS''W/E, ej:  ▲OROSA 14°18'42''N 074°00'000''W
  - Los puntos aparecen EN ORDEN dentro del archivo: ese orden define la
    secuencia de la ruta, así que generamos un segmento entre cada punto
    y el siguiente.
  - El identificador de la ruta (ej. A319) se obtiene del nombre de archivo.
  - Las distancias entre puntos se recalculan con la fórmula de Haversine
    en network.py (no dependemos del número impreso en el PDF, que además
    queda desalineado por el formato de tabla al extraer texto plano).
 
Uso:
    # Un solo archivo:
    python src/parse_enr3.py data/raw/enr3/all_text.txt
 
    # Varios archivos explícitos:
    python src/parse_enr3.py data/raw/enr3/enr3.1/all_text.txt data/raw/enr3/enr3.2/all_text.txt
 
    # TODOS los all_text.txt dentro de una carpeta, sin importar en qué
    # subcarpeta estén (recomendado si generaste uno por subsección):
    python src/parse_enr3.py --search-root data/raw/enr3
"""
 
import argparse
import glob
import os
import re
import pandas as pd
 
 
FILE_SPLIT_RE = re.compile(r"===== ARCHIVO: (.+?) =====")
 
WAYPOINT_RE = re.compile(
    r"▲([A-ZÀ-Ú]{3,8})\s+"
    r"(\d{1,3})°(\d{2})'(\d{1,3})''([NS])\s+"
    r"(\d{1,3})°(\d{2})'(\d{1,3})''([EW])"
)
 
ROUTE_ID_RE = re.compile(
    r"(A\d{2,3}|B\d{2,3}|G\d{2,3}|L\d{2,3}|M\d{2,3}|N\d{2,3}|"
    r"P\d{2,3}|R\d{2,3}|U[A-Z]\d{2,3}|W\d{2,3})",
    re.IGNORECASE,
)
 
 
def dms_to_decimal(deg, minutes, seconds, hemisphere):
    """Convierte grados/minutos/segundos a grados decimales."""
    value = float(deg) + float(minutes) / 60 + float(seconds) / 3600
    if hemisphere in ("S", "W"):
        value = -value
    return round(value, 6)
 
 
def route_id_from_filename(filename):
    """Extrae el designador de ruta (ej. A319) del nombre del archivo."""
    match = ROUTE_ID_RE.search(os.path.basename(filename))
    if match:
        return match.group(1).upper()
    return os.path.splitext(os.path.basename(filename))[0]
 
 
def split_into_files(full_text):
    """Divide el texto combinado en bloques por archivo, usando el
    separador '===== ARCHIVO: ... =====' que genera explore_pdf_batch.py."""
    parts = FILE_SPLIT_RE.split(full_text)
    # parts queda como: ['', 'nombre1', 'contenido1', 'nombre2', 'contenido2', ...]
    files = {}
    for i in range(1, len(parts), 2):
        name = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        files[name] = content
    return files
 
 
def parse_one_file(all_text_path: str, waypoints: dict, segments: list, empty_files: list):
    """Parsea un solo archivo all_text.txt y acumula resultados en las
    estructuras compartidas waypoints/segments/empty_files (se modifican
    en el lugar para poder combinar varios archivos fácilmente)."""
    with open(all_text_path, encoding="utf-8", errors="ignore") as f:
        full_text = f.read()
 
    files = split_into_files(full_text)
 
    for filename, content in files.items():
        route_id = route_id_from_filename(filename)
        found = WAYPOINT_RE.findall(content)
 
        if not found:
            empty_files.append(f"{all_text_path} :: {filename}")
            continue
 
        sequence = []
        for name, d1, m1, s1, h1, d2, m2, s2, h2 in found:
            lat = dms_to_decimal(d1, m1, s1, h1)
            lon = dms_to_decimal(d2, m2, s2, h2)
            waypoints.setdefault(name, (lat, lon))
            sequence.append(name)
 
        for a, b in zip(sequence[:-1], sequence[1:]):
            segments.append((a, b, route_id))
 
 
def save_csv(waypoints, segments, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
 
    wp_df = pd.DataFrame(
        [{"waypoint_id": k, "lat": v[0], "lon": v[1], "tipo": "ordinario"}
         for k, v in waypoints.items()]
    )
    seg_df = pd.DataFrame(segments, columns=["from_id", "to_id", "airway_id"])
 
    wp_path = os.path.join(out_dir, "waypoints.csv")
    seg_path = os.path.join(out_dir, "segments.csv")
    wp_df.to_csv(wp_path, index=False)
    seg_df.to_csv(seg_path, index=False)
 
    return wp_path, seg_path, wp_df, seg_df
 
 
def main():
    parser = argparse.ArgumentParser(description="Parsear ENR 3 (rutas ATS) a CSV")
    parser.add_argument("all_text_paths", nargs="*",
                         help="Uno o varios archivos all_text.txt")
    parser.add_argument("--search-root", default=None,
                         help="En vez de listar archivos a mano, busca TODOS los "
                              "all_text.txt dentro de esta carpeta y subcarpetas")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()
 
    if args.search_root:
        paths = sorted(glob.glob(os.path.join(args.search_root, "**", "all_text*.txt"),
                                  recursive=True))
    else:
        paths = args.all_text_paths
 
    if not paths:
        print("No se encontraron archivos all_text.txt para procesar. "
              "Usa --search-root <carpeta> o indica las rutas directamente.")
        return
 
    print(f"Procesando {len(paths)} archivo(s):")
    for p in paths:
        print(f"  - {p}")
 
    waypoints = {}
    segments = []
    empty_files = []
 
    for path in paths:
        parse_one_file(path, waypoints, segments, empty_files)
 
    wp_path, seg_path, wp_df, seg_df = save_csv(waypoints, segments, args.out_dir)
 
    print(f"\nWaypoints únicos encontrados (combinando todos los archivos): {len(wp_df)}")
    print(f"Segmentos generados (combinando todos los archivos): {len(seg_df)}")
    print(f"Guardado en: {wp_path} y {seg_path}\n")
 
    if empty_files:
        print(f"Aviso: {len(empty_files)} archivo(s) individuales dentro de esos "
              f"all_text.txt sin waypoints detectados (revisar si son escaneados):")
        for f in empty_files[:15]:
            print(f"  - {f}")
        if len(empty_files) > 15:
            print(f"  ... y {len(empty_files) - 15} más")
 
 
if __name__ == "__main__":
    main()
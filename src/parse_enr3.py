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
 
# Un punto significativo en el texto extraído de los ENR 3 tiene este aspecto:
#     ▲OROSA                     (ENR 3.1: símbolo pegado al nombre)
#     ▲ LONAX                    (ENR 3.3: con espacio)
#     ∆ VUMBA                    (punto de notificación a solicitud: otro símbolo)
#     ▲PUERTO CARREÑO            (estaciones VOR: nombre de varias palabras ...)
#     VOR/DME ' PTC'             (... y una línea con el tipo e identificador)
#     06°10'53''N
#     067°29'44''W
# El patrón anterior solo aceptaba "▲NOMBRE" de 3 a 8 letras seguido de la
# coordenada, y por eso descartaba ~2/3 de los puntos (ENR 3.3 completo,
# los puntos con ∆ y las estaciones VOR).
WAYPOINT_RE = re.compile(
    r"[▲∆]\s*([A-ZÀ-ÚÑ0-9][A-ZÀ-ÚÑ0-9 .\-]*?"
    r"(?:[ \t]*\n[A-ZÀ-ÚÑ][A-ZÀ-ÚÑ .\-]*?)?)[ \t]*\n"      # nombre en 1 o 2 líneas
    r"(?:[A-Z/]{2,12}[ \t]*'[^'\n]*'[^\n]*\n)?"
    r"(\d{1,3})°(\d{2})'(\d{1,3})''([NS])\s+"
    r"(\d{1,3})°(\d{2})'(\d{1,3})''([EW])"
)

# Encabezado repetido en cada página (termina en la fila de numeración
# "1 2 3 4 5 6 7") y pie de página. Cuando una página corta un punto a la
# mitad, el nombre queda antes del pie y las coordenadas después del
# encabezado de la página siguiente; quitarlos los vuelve a juntar.
PAGE_SPLIT_RE = re.compile(r"--- página \d+ ---")
PAGE_HEADER_RE = re.compile(r"\A[\s\S]*?^\s*1 2 3 4 5 6 7\s*$\n?", re.MULTILINE)
PAGE_FOOTER_RE = re.compile(r"^AIS COLOMBIA.*$\n?", re.MULTILINE)

ROUTE_ID_RE = re.compile(
    r"(A\d{1,3}|B\d{1,3}|G\d{1,3}|L\d{1,3}|M\d{1,3}|N\d{1,3}|"
    r"P\d{1,3}|R\d{1,3}|U[A-Z]\d{1,3}|V\d{1,3}|W\d{1,3})",
    re.IGNORECASE,
)


def clean_pages(content):
    """Quita encabezados y pies de página del texto de un archivo de ruta y
    une las páginas, para que cada punto significativo quede contiguo."""
    pages = PAGE_SPLIT_RE.split(content)
    cleaned = []
    for page in pages:
        page = PAGE_HEADER_RE.sub("", page, count=1)
        page = PAGE_FOOTER_RE.sub("", page)
        cleaned.append(page.strip("\n"))
    return "\n".join(cleaned) + "\n"
 
 
def dms_to_decimal(deg, minutes, seconds, hemisphere):
    """Convierte grados/minutos/segundos a grados decimales."""
    value = float(deg) + float(minutes) / 60 + float(seconds) / 3600
    if hemisphere in ("S", "W"):
        value = -value
    return round(value, 6)
 
 
def route_id_from_filename(filename):
    """Designador de ruta (ej. A319, UL201, Z001) = último elemento del
    nombre del archivo ('ENR 3.3 UL201.pdf'). La lista fija de prefijos que
    había antes no reconocía UM, UN, UP, UQ, UZ ni Z."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split()[-1].upper() if stem.split() else stem


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
        found = WAYPOINT_RE.findall(clean_pages(content))

        if not found:
            empty_files.append(f"{all_text_path} :: {filename}")
            continue

        sequence = []
        for name, d1, m1, s1, h1, d2, m2, s2, h2 in found:
            name = re.sub(r"\s+", " ", name).strip()
            lat = dms_to_decimal(d1, m1, s1, h1)
            lon = dms_to_decimal(d2, m2, s2, h2)
            if name in waypoints and (abs(waypoints[name][0] - lat) > 0.02
                                      or abs(waypoints[name][1] - lon) > 0.02):
                # Mismo nombre, otra ubicación (>~2 km): no se sobrescribe
                # en silencio; se avisa para revisarlo a mano.
                name_collisions.append((name, waypoints[name], (lat, lon), route_id))
            waypoints.setdefault(name, (lat, lon))
            if not sequence or sequence[-1] != name:
                sequence.append(name)

        for a, b in zip(sequence[:-1], sequence[1:]):
            segments.append((a, b, route_id))


name_collisions = []
 
 
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
 
    if name_collisions:
        print(f"Aviso: {len(name_collisions)} nombre(s) de punto aparecen con coordenadas "
              f"distintas (se conserva la primera; revisar):")
        for n, c1, c2, r in name_collisions[:10]:
            print(f"  - {n}: {c1} vs {c2} (ruta {r})")
 
    if empty_files:
        print(f"Aviso: {len(empty_files)} archivo(s) individuales dentro de esos "
              f"all_text.txt sin waypoints detectados (revisar si son escaneados):")
        for f in empty_files[:15]:
            print(f"  - {f}")
        if len(empty_files) > 15:
            print(f"  ... y {len(empty_files) - 15} más")
 
 
if __name__ == "__main__":
    main()
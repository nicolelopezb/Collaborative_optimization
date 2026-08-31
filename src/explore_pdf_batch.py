
"""
explore_pdf_batch.py
---------------------
Como cada sección del AIP (ENR 3, ENR 4, ENR 5) trae ~100 PDFs individuales,
este script recorre TODA una carpeta y te da:
 
  1. Un resumen: nombre de archivo, número de páginas, si tiene texto o es
     escaneado, y si pdfplumber detectó tablas.
  2. Una muestra de texto de los primeros N archivos (para que identifiques
     el patrón sin tener que abrir cada PDF a mano).
  3. Un archivo combinado (all_text.txt) con el texto completo de todos los
     PDFs, con encabezados que dicen de qué archivo viene cada parte -- útil
     para buscar con Ctrl+F o para pegarme fragmentos aquí en el chat.
 
Ahora busca de forma RECURSIVA: si tienes subcarpetas como
data/raw/enr3/enr3.1/, data/raw/enr3/enr3.2/, etc., las recorre todas
en una sola corrida y usa la ruta relativa (ej. "enr3.1/UL332.pdf") como
identificador, para que no pierdas de qué subsección viene cada archivo.
 
Uso:
    python src/explore_pdf_batch.py data/raw/enr3
    python src/explore_pdf_batch.py data/raw/enr3 --sample 3
    python src/explore_pdf_batch.py data/raw/enr4 --output data/processed/enr4_all_text.txt
"""
 
import argparse
import glob
import os
import pdfplumber
 
 
def summarize_pdf(path, label):
    """Devuelve un resumen corto de un PDF: páginas, si tiene texto, si tiene tablas.
    `label` es la ruta relativa (ej. 'enr3.1/UL332.pdf') para identificar la subsección."""
    try:
        with pdfplumber.open(path) as pdf:
            n_pages = len(pdf.pages)
            has_text = False
            has_tables = False
            first_page_text = ""
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    has_text = True
                    if i == 0:
                        first_page_text = text
                if page.extract_tables():
                    has_tables = True
            return {
                "archivo": label,
                "paginas": n_pages,
                "tiene_texto": has_text,
                "tiene_tablas": has_tables,
                "muestra": first_page_text[:300],
            }
    except Exception as e:
        return {
            "archivo": label,
            "paginas": None,
            "tiene_texto": False,
            "tiene_tablas": False,
            "muestra": f"[ERROR al abrir: {e}]",
        }
 
 
def extract_full_text(path):
    """Extrae todo el texto de un PDF, página por página."""
    chunks = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or "[sin texto en esta página]"
            chunks.append(f"--- página {i + 1} ---\n{text}")
    return "\n".join(chunks)
 
 
def main():
    parser = argparse.ArgumentParser(description="Explorar en lote una carpeta de PDFs del AIP")
    parser.add_argument("folder", help="Carpeta con los PDFs, ej: data/raw/enr3")
    parser.add_argument("--sample", type=int, default=5,
                         help="Cuántos archivos mostrar en detalle en la consola")
    parser.add_argument("--output", default=None,
                         help="Ruta del .txt combinado (por defecto: <folder>/all_text.txt)")
    args = parser.parse_args()
 
    pdf_paths = sorted(glob.glob(os.path.join(args.folder, "**", "*.pdf"), recursive=True))
    if not pdf_paths:
        print(f"No se encontraron PDFs en {args.folder} (ni en sus subcarpetas)")
        return
 
    labels = [os.path.relpath(p, args.folder) for p in pdf_paths]
 
    print(f"Encontrados {len(pdf_paths)} archivos PDF en {args.folder} (incluyendo subcarpetas)\n")
 
    # 1. Resumen de todos los archivos
    summaries = [summarize_pdf(p, lbl) for p, lbl in zip(pdf_paths, labels)]
 
    print(f"{'archivo':40s} {'páginas':8s} {'texto?':8s} {'tablas?':8s}")
    print("-" * 70)
    for s in summaries:
        print(f"{s['archivo']:40s} {str(s['paginas']):8s} "
              f"{str(s['tiene_texto']):8s} {str(s['tiene_tablas']):8s}")
 
    no_text = [s["archivo"] for s in summaries if not s["tiene_texto"]]
    if no_text:
        print(f"\nAviso: {len(no_text)} archivo(s) sin texto extraíble "
              f"(posiblemente escaneados como imagen): {no_text[:10]}")
 
    # 2. Muestra en detalle de los primeros N archivos
    print(f"\n\n===== MUESTRA DE TEXTO (primeros {args.sample} archivos) =====")
    for s in summaries[:args.sample]:
        print(f"\n--- {s['archivo']} ---")
        print(s["muestra"])
 
    # 3. Texto completo combinado, guardado en archivo
    output_path = args.output or os.path.join(args.folder, "all_text.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for path, lbl in zip(pdf_paths, labels):
            f.write(f"\n\n===== ARCHIVO: {lbl} =====\n")
            f.write(extract_full_text(path))
 
    print(f"\n\nTexto completo de los {len(pdf_paths)} archivos guardado en: {output_path}")
    print("Puedes abrir ese archivo y buscar (Ctrl+F) patrones, o pegarme aquí "
          "un par de bloques de ejemplo para diseñar el parser.")
 
 
if __name__ == "__main__":
    main()
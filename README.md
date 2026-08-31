# Optimización colaborativa de rutas aéreas — Colombia

Implementación por fases del artículo de Sui et al. (2026), adaptada al
espacio aéreo colombiano usando datos abiertos de Aerocivil / datos.gov.co.

## Cómo usar esto

1. Copia toda esta carpeta dentro de tu proyecto de VS Code (o ábrela
   directamente como carpeta del proyecto).
2. Crea y activa tu entorno virtual, luego instala dependencias:
   ```
   python -m venv venv
   # Windows: .\venv\Scripts\activate   |   Mac/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```
3. YA FUNCIONA con datos de ejemplo (8 aeropuertos principales de Colombia):
   ```
   python src/network.py
   python src/demand.py
   ```
   Esto te confirma que el pipeline corre correctamente antes de meter
   datos reales.

## Siguiente paso: datos reales

1. Descarga ENR 3 / ENR 4 / ENR 5 del eAIP Colombia (ver instrucciones que
   te di en el chat) y ponlos en `data/raw/`.
2. Corre `python src/explore_pdf.py data/raw/ENR3.pdf` para ver cómo está
   estructurado el texto — cada AIRAC puede variar un poco el formato.
3. Con eso, construimos juntos el parser específico (regex) para extraer
   waypoints y rutas reales, y los guardamos como
   `data/processed/waypoints.csv` y `data/processed/segments.csv`
   (mismo formato que los archivos `_ejemplo.csv`, que puedes usar de
   plantilla).
4. Para la demanda: descarga el CSV de "Operaciones aéreas acumuladas en
   Colombia" (datos.gov.co) y/o el Excel de "Oferta y Demanda" de Aerocivil,
   y ajusta el `column_map` en `demand.py` → `clean_columns()` según las
   columnas reales que tenga el archivo.

## Estructura

```
data/
  raw/          <- PDFs y CSVs originales tal como los descargas
  processed/    <- CSVs limpios: waypoints.csv, segments.csv, demand.csv
src/
  explore_pdf.py   <- Fase 1: inspeccionar estructura del PDF del AIP
  network.py       <- Fase 1: construir grafo G=(V,E) y graficarlo
  demand.py        <- Fase 2: cargar/generar demanda de vuelos (conjunto A)
outputs/        <- Mapas, gráficas, resultados de cada corrida
notebooks/      <- Para exploración y análisis visual
```

## Próximas fases (aún por construir)

- Fase 3: K-caminos más cortos (`networkx.shortest_simple_paths`)
- Fase 4: Modelo MILP de asignación (PuLP), ecuaciones (32)-(40) del artículo
- Fase 5: Crecimiento de red vía campos de potencial (APF), ecuaciones (1)-(19)
- Fase 6: Ciclo iterativo completo + índices de red, ecuaciones (41)-(49)

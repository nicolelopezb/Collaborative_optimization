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

## Estado de las fases

- Fases 1-4 (red, demanda, K-caminos, MILP del nivel inferior): hechas.
- Fase 5 (crecimiento de red + APF, Ec. 1-19): hecha. Los waypoints nuevos se
  conectan (vecino relativo, longitud, ángulo) y luego su posición se optimiza
  con las fuerzas de las Ec. 13-17 y el paso de las Ec. 18-19, manteniendo la
  topología fija. Lmin, el radio r, el ángulo mínimo y las zonas PRD son
  restricciones duras; se aplica además la perturbación estocástica del artículo.
- Fase 6 (ciclo iterativo): hecha, con los dos criterios de convergencia de la
  Sec. 3.2 (I: variación de exceso y distancia; II: exceso horario <= 5 % de la
  capacidad) y el borrado de waypoints sin tráfico (4.1.3).
- Pendiente: índices de red (Ec. 41-49), escenarios 1/2/3 comparados, análisis
  de sensibilidad, tiempos de paso por waypoint (el artículo los obtiene por
  simulación), varios niveles de vuelo y zonas PRD reales de Colombia
  (`--prd-zones` acepta un CSV lat,lon,radius_km).

## Uso del ciclo completo

```
python src/main.py --max-iterations 8 --wp-capacity 20 --m-new-nodes 3
python tests/test_apf_and_convergence.py
```

Notas: `--wp-capacity` es un supuesto (no hay dato público de capacidad); con
5 aeronaves/hora el Criterio II no se alcanza porque el exceso inicial es muy
grande. Los aeropuertos no cuentan como waypoints para la capacidad: su límite
son las capacidades Q de aeropuerto (Ec. 38-40). Requiere `pulp==2.9.0`.

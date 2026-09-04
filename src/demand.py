"""
demand.py
---------
Carga y limpia la demanda de vuelos (conjunto A del artículo) a partir de
los archivos que descargues de:
  - datos.gov.co -> "Operaciones aéreas acumuladas en Colombia" (CSV)
  - Aerocivil -> "Estadísticas de Oferta y Demanda - Transporte de Pasajeros" (Excel)

La idea es terminar con una tabla estándar, sin importar la fuente original:

    flight_id, origin, destination, dep_time, week_day

A partir de eso, build_hourly_demand() agrupa por hora para construir las
96 franjas horarias T (7 días x 24 horas) que usa el artículo (Ec. 1, T).

NOTA: estos datasets de Colombia normalmente NO traen la hora exacta de
cada vuelo individual (a diferencia de BTS en EE.UU.). Si el archivo que
bajaste solo trae frecuencias semanales por ruta (ej. "14 vuelos/semana
Bogotá-Medellín"), usa la función synthesize_weekly_flights() para generar
vuelos individuales distribuidos en la semana de forma razonable, dejando
clara la naturaleza sintética de esa distribución horaria en tu informe.
"""

import pandas as pd
import numpy as np


def load_raw_csv(path: str) -> pd.DataFrame:
    """Carga el CSV tal como venga de datos.gov.co, sin asumir nombres de columna.
    Imprime las columnas para que ajustes el mapeo en clean_columns()."""
    df = pd.read_csv(path)
    print("Columnas encontradas en el archivo:")
    print(list(df.columns))
    return df


def clean_columns(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Renombra columnas según el mapeo que definas tras inspeccionar el CSV.

    Ejemplo de uso:
        column_map = {
            "AEROPUERTO ORIGEN": "origin",
            "AEROPUERTO DESTINO": "destination",
            "FECHA": "dep_date",
        }
        df = clean_columns(df, column_map)
    """
    return df.rename(columns=column_map)


def synthesize_weekly_flights(routes: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Genera vuelos individuales sintéticos para una semana a partir de
    frecuencias semanales por ruta.

    routes: DataFrame con columnas [origin, destination, weekly_frequency]

    Distribuye los vuelos de cada ruta en horas "pico" realistas
    (mañana 6-9h, mediodía 12-14h, noche 17-20h) en vez de completamente
    al azar, para reflejar el patrón típico de demanda doméstica colombiana.
    Esto es una aproximación razonable cuando no existe el dato real por hora,
    y debe declararse como supuesto en el informe.
    """
    rng = np.random.default_rng(seed)
    peak_hours = [6, 7, 8, 9, 12, 13, 14, 17, 18, 19, 20]
    rows = []
    flight_counter = 0

    for _, r in routes.iterrows():
        n_flights = int(r["weekly_frequency"])
        for _ in range(n_flights):
            day = rng.integers(0, 7)          # 0=lunes ... 6=domingo
            hour = rng.choice(peak_hours)
            minute = rng.integers(0, 60)
            flight_counter += 1
            rows.append({
                "flight_id": f"SYN{flight_counter:05d}",
                "origin": r["origin"],
                "destination": r["destination"],
                "week_day": day,
                "dep_hour": hour,
                "dep_minute": minute,
            })

    return pd.DataFrame(rows)


def load_aerocivil_operations(path: str, year=None, month=None,
                               tipo_vuelo_filter=None,
                               tipo_operacion_filter="SALIDA"):
    """Carga el archivo real de Aerocivil/datos.gov.co ('Operaciones aéreas
    acumuladas en Colombia') y lo agrega a nivel de ruta (origen-destino)
    para un mes específico, devolviendo la frecuencia MENSUAL de vuelos
    por ruta.

    Columnas reales del archivo (confirmadas por la usuaria):
        ANIO, MES, AEROPUERTO_OPERACION, ORIGEN, DESTINO, EMPRESA,
        EXPLOTADOR, TIPO_VUELO, TRAFICO, TIPO_OPERACION, TotalOperaciones

    Valores reales confirmados de TIPO_OPERACION: 'SALIDA', 'LLEGADA',
    'TOQUE Y DESPEGUE'. Por defecto se filtra a SOLO 'SALIDA', porque cada
    vuelo real queda registrado dos veces en el archivo (una vez como
    SALIDA en el aeropuerto de origen, otra vez como LLEGADA en el de
    destino) -- si se sumaran ambos, cada vuelo se contaría el doble.
    'TOQUE Y DESPEGUE' se excluye siempre porque son maniobras de
    entrenamiento (el avión despega y aterriza en el mismo aeropuerto), no
    vuelos punto a punto reales.

    Valores reales confirmados de TIPO_VUELO: 'N', 'M', 'G', 'X', 'S' (y
    nulos). NO se encontró documentación pública oficial que confirme su
    significado exacto (probablemente Nacional/Internacional/Militar/
    General/etc., pero no está confirmado). Por eso tipo_vuelo_filter se
    deja en None por defecto (no se filtra por este campo). Si tu informe
    necesita distinguir tipos de vuelo, contacta a Aerocivil o revisa el
    diccionario de datos del dataset en datos.gov.co (la mayoría de sus
    datasets tienen un enlace "Ver diccionario de datos" en la página).

    year, month: si se especifican, filtra a ese mes exacto (recomendado,
    para tomar "una semana representativa" como hace el artículo). Si se
    deja en None, se usa el mes más reciente disponible en el archivo.

    Devuelve: DataFrame [origin, destination, monthly_frequency]
    """
    df = pd.read_csv(path, low_memory=False)

    if year is None or month is None:
        # Tomar el mes más reciente disponible
        latest = df[["ANIO", "MES"]].drop_duplicates().sort_values(["ANIO", "MES"]).iloc[-1]
        year, month = latest["ANIO"], latest["MES"]
        print(f"Usando el mes más reciente disponible: {year}-{month}")

    filtered = df[(df["ANIO"] == year) & (df["MES"] == month)]

    # Excluir siempre las maniobras de entrenamiento (no son vuelos punto a punto)
    filtered = filtered[filtered["TIPO_OPERACION"] != "TOQUE Y DESPEGUE"]

    if tipo_vuelo_filter is not None:
        filtered = filtered[filtered["TIPO_VUELO"] == tipo_vuelo_filter]
    if tipo_operacion_filter is not None:
        filtered = filtered[filtered["TIPO_OPERACION"] == tipo_operacion_filter]

    routes = (
        filtered.groupby(["ORIGEN", "DESTINO"])["TotalOperaciones"]
        .sum()
        .reset_index()
        .rename(columns={"ORIGEN": "origin", "DESTINO": "destination",
                         "TotalOperaciones": "monthly_frequency"})
    )

    print(f"Filtrado aplicado: TIPO_OPERACION='{tipo_operacion_filter}' "
          f"(excluyendo siempre TOQUE Y DESPEGUE), "
          f"TIPO_VUELO={'sin filtrar' if tipo_vuelo_filter is None else tipo_vuelo_filter}")
    print(f"Rutas únicas encontradas: {len(routes)}")

    return routes


def monthly_to_weekly_frequency(routes_monthly: pd.DataFrame, days_per_month=30.44):
    """Convierte la frecuencia mensual de cada ruta a una frecuencia semanal
    aproximada (dividiendo por ~4.345 semanas/mes), redondeando hacia el
    entero más cercano. Rutas que redondean a 0 vuelos/semana se descartan."""
    weeks_per_month = days_per_month / 7
    routes = routes_monthly.copy()
    routes["weekly_frequency"] = (routes["monthly_frequency"] / weeks_per_month).round().astype(int)
    before = len(routes)
    routes = routes[routes["weekly_frequency"] > 0]
    dropped = before - len(routes)
    if dropped:
        print(f"Aviso: se descartaron {dropped} rutas con menos de 0.5 vuelos/semana "
              f"(demasiado infrecuentes para el redondeo).")
    return routes[["origin", "destination", "weekly_frequency"]]


def build_hourly_demand(flights: pd.DataFrame) -> pd.DataFrame:
    """Agrupa vuelos en las 96 franjas horarias semanales (T en el artículo).

    Devuelve una tabla con conteo de vuelos por (origin, destination, time_slot),
    donde time_slot = week_day * 24 + dep_hour  (0 a 167 si usas 7*24=168,
    o ajusta a 96 si sigues la convención textual del artículo -- revisa
    si ellos usan slots de 1h * 7 días = 168, aunque el texto dice 96;
    confirma la convención exacta antes de fijar esto en el modelo final).
    """
    flights = flights.copy()
    flights["time_slot"] = flights["week_day"] * 24 + flights["dep_hour"]

    demand = (
        flights.groupby(["origin", "destination", "time_slot"])
        .size()
        .reset_index(name="n_flights")
    )
    return demand


if __name__ == "__main__":
    # Ejemplo de uso con datos sintéticos, para probar el pipeline hoy mismo
    example_routes = pd.DataFrame({
        "origin": ["SKBO", "SKBO", "SKBO", "SKMD"],
        "destination": ["SKRG", "SKCL", "SKBQ", "SKCL"],
        "weekly_frequency": [70, 56, 63, 21],
    })
    flights = synthesize_weekly_flights(example_routes)
    demand = build_hourly_demand(flights)
    print(demand.head(10))
    print(f"\nTotal de vuelos generados: {len(flights)}")

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
franjas horarias semanales que usa el artículo.
"""

import pandas as pd
import numpy as np


def load_raw_csv(path: str) -> pd.DataFrame:
    """Carga el CSV tal como venga de datos.gov.co, sin asumir nombres de columna.
    Imprime las columnas para que ajustes el mapeo en clean_columns()."""
    df = pd.read_csv(path, low_memory=False)
    print("Columnas encontradas en el archivo:")
    print(list(df.columns))
    return df


def clean_columns(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Renombra columnas según el mapeo que definas tras inspeccionar el CSV."""
    return df.rename(columns=column_map)


def synthesize_weekly_flights(routes: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Genera vuelos individuales sintéticos para una semana a partir de
    frecuencias semanales por ruta.

    routes: DataFrame con columnas [origin, destination, weekly_frequency]

    Distribuye los vuelos de cada ruta en horas "pico" realistas
    (mañana 6-9h, mediodía 12-14h, noche 17-20h).
    """
    rng = np.random.default_rng(seed)
    peak_hours = [6, 7, 8, 9, 12, 13, 14, 17, 18, 19, 20]
    rows = []
    flight_counter = 0

    for _, r in routes.iterrows():
        n_flights = int(r["weekly_frequency"])
        for _ in range(n_flights):
            day = rng.integers(0, 7)
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

    Columnas reales del archivo (confirmadas):
        ANIO, MES, AEROPUERTO_OPERACION, ORIGEN, DESTINO, EMPRESA,
        EXPLOTADOR, TIPO_VUELO, TRAFICO, TIPO_OPERACION, TotalOperaciones

    Valores reales confirmados de TIPO_OPERACION: 'SALIDA', 'LLEGADA',
    'TOQUE Y DESPEGUE'. Por defecto se filtra a SOLO 'SALIDA' (cada vuelo
    real queda registrado dos veces: SALIDA en origen, LLEGADA en destino;
    sumar ambos duplicaría el conteo). 'TOQUE Y DESPEGUE' se excluye
    siempre (maniobras de entrenamiento, no vuelos punto a punto).

    Valores reales confirmados de TIPO_VUELO: 'N','M','G','X','S' (sin
    documentación pública confirmada de su significado exacto) -- por eso
    tipo_vuelo_filter se deja en None por defecto.

    Devuelve: DataFrame [origin, destination, monthly_frequency]
    """
    df = pd.read_csv(path, low_memory=False)

    if year is None or month is None:
        latest = df[["ANIO", "MES"]].drop_duplicates().sort_values(["ANIO", "MES"]).iloc[-1]
        year, month = latest["ANIO"], latest["MES"]
        print(f"Usando el mes más reciente disponible: {year}-{month}")

    filtered = df[(df["ANIO"] == year) & (df["MES"] == month)]
    filtered = filtered[filtered["TIPO_OPERACION"] != "TOQUE Y DESPEGUE"]

    if tipo_vuelo_filter is not None:
        filtered = filtered[filtered["TIPO_VUELO"] == tipo_vuelo_filter]
    if tipo_operacion_filter is not None:
        filtered = filtered[filtered["TIPO_OPERACION"] == tipo_operacion_filter]

    # TotalOperaciones puede venir como texto (comas de miles, vacíos, u
    # otros caracteres) -- se limpia y convierte a número de forma segura
    # ANTES de agrupar/sumar, para no sumar strings por accidente.
    filtered = filtered.copy()
    filtered["TotalOperaciones"] = (
        filtered["TotalOperaciones"].astype(str).str.replace(",", "", regex=False)
    )
    filtered["TotalOperaciones"] = pd.to_numeric(filtered["TotalOperaciones"], errors="coerce")

    n_invalid = filtered["TotalOperaciones"].isna().sum()
    if n_invalid:
        print(f"Aviso: {n_invalid} fila(s) con TotalOperaciones no numérico fueron "
              f"descartadas.")
    filtered = filtered.dropna(subset=["TotalOperaciones"])

    # Excluir "rutas" donde origen == destino: no son vuelos punto a punto
    # reales, sino operaciones locales/de aviación general (ej. vuelos de
    # entrenamiento que despegan y permanecen en el área del mismo aeropuerto).
    n_self_loops = (filtered["ORIGEN"] == filtered["DESTINO"]).sum()
    if n_self_loops:
        print(f"Aviso: {n_self_loops} fila(s) con origen == destino fueron "
              f"excluidas (operaciones locales, no rutas punto a punto).")
    filtered = filtered[filtered["ORIGEN"] != filtered["DESTINO"]]

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
    print(f"Tipo de dato de monthly_frequency: {routes['monthly_frequency'].dtype}")

    return routes


def monthly_to_weekly_frequency(routes_monthly: pd.DataFrame, days_per_month=30.44):
    """Convierte la frecuencia mensual de cada ruta a una frecuencia semanal
    aproximada (dividiendo por ~4.345 semanas/mes), redondeando hacia el
    entero más cercano. Rutas que redondean a 0 vuelos/semana se descartan."""
    weeks_per_month = days_per_month / 7
    routes = routes_monthly.copy()

    # Segunda capa de seguridad: forzar numérico aquí también, por si
    # monthly_frequency llega como texto desde otra fuente (evita el
    # TypeError de dividir texto por un número).
    routes["monthly_frequency"] = pd.to_numeric(routes["monthly_frequency"], errors="coerce")
    n_bad = routes["monthly_frequency"].isna().sum()
    if n_bad:
        print(f"Aviso: {n_bad} ruta(s) con monthly_frequency no numérico "
              f"fueron descartadas en monthly_to_weekly_frequency().")
    routes = routes.dropna(subset=["monthly_frequency"])

    routes["weekly_frequency"] = (routes["monthly_frequency"] / weeks_per_month).round().astype(int)

    before = len(routes)
    routes = routes[routes["weekly_frequency"] > 0]
    dropped = before - len(routes)
    if dropped:
        print(f"Aviso: se descartaron {dropped} rutas con menos de 0.5 vuelos/semana.")
    return routes[["origin", "destination", "weekly_frequency"]]


def build_hourly_demand(flights: pd.DataFrame) -> pd.DataFrame:
    """Agrupa vuelos en franjas horarias semanales (T en el artículo)."""
    flights = flights.copy()
    flights["time_slot"] = flights["week_day"] * 24 + flights["dep_hour"]

    demand = (
        flights.groupby(["origin", "destination", "time_slot"])
        .size()
        .reset_index(name="n_flights")
    )
    return demand


if __name__ == "__main__":
    example_routes = pd.DataFrame({
        "origin": ["SKBO", "SKBO", "SKBO", "SKMD"],
        "destination": ["SKRG", "SKCL", "SKBQ", "SKCL"],
        "weekly_frequency": [70, 56, 63, 21],
    })
    flights = synthesize_weekly_flights(example_routes)
    demand = build_hourly_demand(flights)
    print(demand.head(10))
    print(f"\nTotal de vuelos generados: {len(flights)}")

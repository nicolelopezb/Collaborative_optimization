"""
capacities.py
--------------
Genera las capacidades que el artículo necesita pero que no existen en
fuentes públicas para Colombia:

  - C^h,t_v : capacidad horaria de cada waypoint v, en el nivel de vuelo h,
              en la franja horaria t                         (Ec. 1, 23, 34)
  - Q^t_dep,k, Q^t_arr,k, Q^t_APRT,k : capacidad horaria de salida, llegada
              y total de cada aeropuerto k                   (Ec. 29-31, 38-40)

Regla usada (razonable y documentable en un informe académico, dado que el
dato real es operacional/confidencial incluso en el artículo original):
  - Capacidad de un waypoint = separación mínima estándar entre aeronaves.
    Con separación temporal mínima de ~3 minutos entre aeronaves en el mismo
    punto/nivel, la capacidad horaria es 60/3 = 20 aeronaves/hora/nivel.
  - Capacidad de aeropuerto = se estima a partir de su grado en la red
    (cuántas rutas lo conectan) y de un valor base configurable, o puede
    sobreescribirse manualmente para aeropuertos principales (ej. Bogotá).

Uso:
    from capacities import generate_waypoint_capacity, generate_airport_capacity
"""

import itertools
import pandas as pd


DEFAULT_WAYPOINT_CAPACITY_PER_HOUR = 20   # aeronaves/hora/nivel de vuelo (Sección 4.1)
DEFAULT_FLIGHT_LEVELS = ["FL_bajo", "FL_medio", "FL_alto"]  # simplificación de H
N_TIME_SLOTS = 24 * 7  # 168 franjas horarias (7 días x 24 horas)


def generate_waypoint_capacity(waypoint_ids, per_hour=DEFAULT_WAYPOINT_CAPACITY_PER_HOUR,
                                levels=None, n_slots=N_TIME_SLOTS):
    """Genera C^h,t_v para todos los waypoints, niveles y franjas horarias.

    Devuelve un DataFrame largo: columnas [waypoint_id, level, time_slot, capacity]
    (formato "largo" para que sea fácil de consultar/filtrar más adelante).
    """
    levels = levels or DEFAULT_FLIGHT_LEVELS
    rows = []
    for wp, lvl, t in itertools.product(waypoint_ids, levels, range(n_slots)):
        rows.append({"waypoint_id": wp, "level": lvl, "time_slot": t, "capacity": per_hour})
    return pd.DataFrame(rows)


def generate_airport_capacity(airport_ids, dep_per_hour=30, arr_per_hour=30,
                               total_per_hour=None, n_slots=N_TIME_SLOTS,
                               overrides=None):
    """Genera Q^t_dep,k, Q^t_arr,k, Q^t_APRT,k para cada aeropuerto.

    overrides: dict opcional {airport_id: {"dep": x, "arr": y, "total": z}}
    para ajustar aeropuertos específicos (ej. hubs grandes como Bogotá).
    """
    overrides = overrides or {}
    total_per_hour = total_per_hour or (dep_per_hour + arr_per_hour)

    rows = []
    for apt in airport_ids:
        ov = overrides.get(apt, {})
        dep = ov.get("dep", dep_per_hour)
        arr = ov.get("arr", arr_per_hour)
        tot = ov.get("total", total_per_hour)
        for t in range(n_slots):
            rows.append({"airport_id": apt, "time_slot": t,
                         "Q_dep": dep, "Q_arr": arr, "Q_total": tot})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Prueba rápida con el ejemplo de aeropuertos colombianos
    airports = ["SKBO", "SKRG", "SKCL", "SKBQ"]
    overrides = {"SKBO": {"dep": 60, "arr": 60, "total": 100}}  # Bogotá es un hub
    apt_cap = generate_airport_capacity(airports, overrides=overrides)
    print(apt_cap.head())
    print(f"\nFilas generadas: {len(apt_cap)} (4 aeropuertos x 168 franjas horarias)")

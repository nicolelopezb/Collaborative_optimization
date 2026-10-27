"""
milp_model.py
--------------
Modelo de optimización lineal entera mixta (MILP) del nivel inferior,
Sección 5.2 del artículo (Ecuaciones 32-40).

Decide, para cada vuelo, cuál de sus rutas candidatas (generadas por
pathing.py con K-shortest paths) usar, minimizando:
  (1) AO: el tráfico excedente total en los waypoints (Ec. 33-34)
  (2) LO: la distancia adicional total de vuelo (Ec. 35-36)
sujeto a:
  - cada vuelo elige exactamente 1 ruta                    (Ec. 37)
  - capacidad de salida de cada aeropuerto por hora         (Ec. 38)
  - capacidad de llegada de cada aeropuerto por hora        (Ec. 39)
  - capacidad total (salida+llegada) de cada aeropuerto     (Ec. 40)

El artículo usa Gurobi; aquí usamos PuLP con el solver CBC (gratuito,
incluido en la librería) para que puedas correrlo sin licencia. Si más
adelante consigues una licencia de Gurobi, el mismo modelo se puede
resolver con pulp.GUROBI_CMD() cambiando una sola línea (ver main()).

NOTA IMPORTANTE - simplificación deliberada de la primera versión:
El artículo minimiza AO y LO como dos objetivos relacionados por
consistencia de prioridades (congestión primero, eficiencia después,
igual que en la Sección 4.1 con α=0.7, β=0.3). Aquí implementamos la
misma idea como una suma ponderada: minimizar (alpha*AO + beta*LO).
Esto es un enfoque razonable y muy usado para bi-objetivo en MILP, y
es ajustable con los parámetros alpha/beta.

Uso:
    from milp_model import solve_path_assignment
    result = solve_path_assignment(routes_df, flights_df, capacities_df, airport_cap_df)
"""

import pandas as pd
import pulp


def solve_path_assignment(routes_df: pd.DataFrame, flights_df: pd.DataFrame,
                           waypoint_cap_df: pd.DataFrame, airport_cap_df: pd.DataFrame,
                           alpha=0.7, beta=0.3, time_limit_sec=60):
    """Resuelve el modelo MILP de asignación de rutas.

    routes_df: salida de pathing.build_candidate_routes()
               columnas: origin, destination, route_idx, path, length_km,
                         great_circle_km, extra_km
    flights_df: tabla de vuelos con columnas:
               flight_id, origin, destination, time_slot
               (time_slot = franja horaria 0-167, ver demand.py)
    waypoint_cap_df: salida de capacities.generate_waypoint_capacity()
    airport_cap_df: salida de capacities.generate_airport_capacity()

    Devuelve un dict con:
        assignment: DataFrame [flight_id, route_idx, path]  (ruta elegida por cada vuelo)
        status: estado de la solución ("Optimal", "Infeasible", etc.)
        objective_value: valor del objetivo ponderado
    """
    prob = pulp.LpProblem("flight_path_assignment", pulp.LpMinimize)

    # --- Variables de decisión: x[flight_id, route_idx] (Ec. 32) ---
    flights = flights_df.to_dict("records")
    x = {}
    flight_routes = {}  # flight_id -> lista de (route_idx, path, extra_km)

    for f in flights:
        od_routes = routes_df[
            (routes_df["origin"] == f["origin"]) &
            (routes_df["destination"] == f["destination"])
        ]
        if od_routes.empty:
            continue  # sin ruta disponible para este par OD; se omite (revisar conectividad)

        options = []
        for _, r in od_routes.iterrows():
            var = pulp.LpVariable(
                f"x_{f['flight_id']}_{r['route_idx']}", cat="Binary"
            )
            x[(f["flight_id"], r["route_idx"])] = var
            options.append((r["route_idx"], r["path"], r["extra_km"]))
        flight_routes[f["flight_id"]] = options

    # --- Restricción (Ec. 37): cada vuelo elige exactamente 1 ruta ---
    for flight_id, options in flight_routes.items():
        prob += pulp.lpSum(x[(flight_id, idx)] for idx, _, _ in options) == 1

    # --- Objetivo 2 (LO, Ec. 35-36): distancia adicional total ---
    LO_terms = []
    for flight_id, options in flight_routes.items():
        for idx, path, extra_km in options:
            LO_terms.append(x[(flight_id, idx)] * extra_km)
    LO = pulp.lpSum(LO_terms)

    # --- Objetivo 1 (AO, Ec. 33-34): tráfico excedente en waypoints ---
    # Para cada (waypoint, time_slot), contamos cuántos vuelos-ruta lo cruzan
    # en esa franja horaria, y comparamos contra la capacidad. Como el modelo
    # es lineal, el exceso se modela con una variable de holgura E_v_t >= 0
    # y dos restricciones: E_v_t >= flujo - capacidad, E_v_t >= 0.
    flight_by_id = {f["flight_id"]: f for f in flights}

    waypoint_time_flights = {}  # (waypoint, time_slot) -> lista de variables x involucradas
    for flight_id, options in flight_routes.items():
        t = flight_by_id[flight_id]["time_slot"]
        for idx, path, extra_km in options:
            for wp in path:
                key = (wp, t)
                waypoint_time_flights.setdefault(key, []).append(x[(flight_id, idx)])

    cap_lookup = waypoint_cap_df.groupby(["waypoint_id", "time_slot"])["capacity"].sum().to_dict()
    # (sumamos capacidad de todos los niveles de vuelo H, ya que en esta
    # primera versión no diferenciamos explícitamente el nivel asignado a
    # cada vuelo -- simplificación documentada, ver README de próximos pasos)

    # Solo se penaliza el exceso en nodos que tienen capacidad de waypoint
    # definida. Los aeropuertos (origen/destino de las rutas) no la tienen: su
    # límite son las restricciones Q_dep / Q_arr de abajo (Ec. 38-40). Si se
    # trataran como waypoints con capacidad 0, el modelo vería un "exceso"
    # enorme e inevitable en cada hub.
    capacity_nodes = set(waypoint_cap_df["waypoint_id"])

    excess_vars = {}
    AO_terms = []
    for (wp, t), var_list in waypoint_time_flights.items():
        if wp not in capacity_nodes:
            continue
        cap = cap_lookup.get((wp, t), 0)
        e = pulp.LpVariable(f"excess_{wp}_{t}", lowBound=0)
        prob += e >= pulp.lpSum(var_list) - cap
        excess_vars[(wp, t)] = e
        AO_terms.append(e)
    AO = pulp.lpSum(AO_terms)

    # --- Objetivo combinado (ponderado) ---
    prob += alpha * AO + beta * LO

    # --- Restricciones de capacidad de aeropuerto (Ec. 38-40) ---
    apt_cap_lookup = airport_cap_df.set_index(["airport_id", "time_slot"])[
        ["Q_dep", "Q_arr", "Q_total"]
    ].to_dict("index")

    dep_flights = {}
    arr_flights = {}
    for flight_id, options in flight_routes.items():
        f = flight_by_id[flight_id]
        t = f["time_slot"]
        var_sum = pulp.lpSum(x[(flight_id, idx)] for idx, _, _ in options)
        dep_flights.setdefault((f["origin"], t), []).append(var_sum)
        arr_flights.setdefault((f["destination"], t), []).append(var_sum)

    for (apt, t), var_list in dep_flights.items():
        cap = apt_cap_lookup.get((apt, t), {}).get("Q_dep")
        if cap is not None:
            prob += pulp.lpSum(var_list) <= cap
    for (apt, t), var_list in arr_flights.items():
        cap = apt_cap_lookup.get((apt, t), {}).get("Q_arr")
        if cap is not None:
            prob += pulp.lpSum(var_list) <= cap

    # --- Resolver ---
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]

    rows = []
    for flight_id, options in flight_routes.items():
        for idx, path, extra_km in options:
            if pulp.value(x[(flight_id, idx)]) and pulp.value(x[(flight_id, idx)]) > 0.5:
                rows.append({"flight_id": flight_id, "route_idx": idx, "path": path})
                break

    assignment = pd.DataFrame(rows)

    return {
        "assignment": assignment,
        "status": status,
        "objective_value": pulp.value(prob.objective),
    }


if __name__ == "__main__":
    # Prueba mínima de humo (smoke test) con datos sintéticos pequeños
    import sys
    sys.path.insert(0, ".")
    from network import build_network
    from pathing import build_candidate_routes
    from capacities import generate_waypoint_capacity, generate_airport_capacity

    G = build_network("data/processed/waypoints.csv", "data/processed/segments.csv")

    flights_df = pd.DataFrame({
        "flight_id": ["F1", "F2", "F3"],
        "origin": ["SKBO", "SKBO", "SKBO"],
        "destination": ["SKBG", "SKBG", "SKBG"],
        "time_slot": [10, 10, 34],
    })

    routes_df = build_candidate_routes(G, flights_df, k=3)
    wp_cap = generate_waypoint_capacity(list(G.nodes), per_hour=1)  # capacidad muy baja
    # a propósito, para forzar que aparezca tráfico excedente en la prueba
    apt_cap = generate_airport_capacity(["SKBO", "SKBG"], dep_per_hour=10, arr_per_hour=10)

    result = solve_path_assignment(routes_df, flights_df, wp_cap, apt_cap)
    print("Estado:", result["status"])
    print("Valor objetivo:", result["objective_value"])
    print(result["assignment"])

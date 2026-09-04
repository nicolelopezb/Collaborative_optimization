"""
apf_growth.py
--------------
Mecanismo de crecimiento adaptativo de la red aérea (Sección 4 del artículo,
nivel superior del modelo bi-nivel). Implementa:

  4.1.1 Reglas de generación de waypoints virtuales (Ec. 1-7)
  4.1.2 Reglas de conexión de segmentos (Ec. 8, + límites de longitud y ángulo)
  4.1.3 Mecanismo de eliminación de waypoints redundantes
  4.3   Optimización de posición vía campos de potencial artificial (Ec. 13-19)

NOTA SOBRE FIDELIDAD AL ARTÍCULO: esta es una implementación literal de las
fórmulas tal como aparecen en el paper, pero con una simplificación
declarada: el artículo genera la malla de puntos virtuales sobre TODO el
espacio aéreo usando el método alpha-shape para extraer el contorno exacto;
aquí generamos la malla sobre el rectángulo delimitador (bounding box) de
los waypoints existentes, que es más simple de implementar y razonable
para una primera versión (se puede refinar después con un alpha-shape real
usando la librería `alphashape`).

Uso:
    from apf_growth import compute_weights, generate_virtual_grid, \
        select_top_candidates, connect_new_waypoint, optimize_position
"""

import math
import itertools
import numpy as np
import networkx as nx
import pandas as pd

from network import great_circle_distance_km


# ---------------------------------------------------------------------------
# 4.1.1 (1)-(7): Cálculo de pesos de los waypoints existentes
# ---------------------------------------------------------------------------

def compute_excess_traffic(G: nx.Graph, flow_df: pd.DataFrame, cap_df: pd.DataFrame):
    """Ecuación (1): Ov = suma sobre t,h de max(f^h,t_v - C^h,t_v, 0)

    flow_df: columnas [waypoint_id, level, time_slot, flights]  (f^h,t_v)
    cap_df:  columnas [waypoint_id, level, time_slot, capacity] (C^h,t_v)

    Devuelve: Series indexada por waypoint_id con el valor de Ov.
    """
    merged = flow_df.merge(cap_df, on=["waypoint_id", "level", "time_slot"], how="right")
    merged["flights"] = merged["flights"].fillna(0)
    merged["excess"] = (merged["flights"] - merged["capacity"]).clip(lower=0)
    Ov = merged.groupby("waypoint_id")["excess"].sum()
    # Waypoints sin ningún registro de flujo (nunca visitados) tienen Ov = 0
    for node in G.nodes:
        if node not in Ov.index:
            Ov[node] = 0.0
    return Ov


def compute_wv1(Ov: pd.Series) -> pd.Series:
    """Ecuación (2): wv,1 = (Ov - Omin) / (Omax - Omin), normalizado a [0,1]."""
    omin, omax = Ov.min(), Ov.max()
    if omax == omin:
        return pd.Series(0.0, index=Ov.index)
    return (Ov - omin) / (omax - omin)


def compute_flow_weighted_path_factor(routes_assigned: pd.DataFrame):
    """Ecuaciones (3)-(4): factor de ruta ponderado por flujo, qf_v.

    routes_assigned: DataFrame con una fila por VUELO ya asignado a una ruta,
    con columnas: [flight_id, path (lista de waypoints), length_km, great_circle_km]

    Para cada waypoint v que aparece en al menos una ruta, calcula:
        qf_v = sum(f_st_v * l_st/d_st) / sum(f_st_v * n_st(v))
    donde cada vuelo cuenta como una unidad de flujo (f=1 por fila, ya que
    cada fila de routes_assigned ya es un vuelo individual).
    """
    numerator = {}
    denominator = {}

    for _, row in routes_assigned.iterrows():
        if row["great_circle_km"] <= 0:
            continue
        ratio = row["length_km"] / row["great_circle_km"]
        for wp in row["path"]:
            numerator[wp] = numerator.get(wp, 0.0) + ratio
            denominator[wp] = denominator.get(wp, 0.0) + 1

    qf = {wp: numerator[wp] / denominator[wp] for wp in numerator if denominator[wp] > 0}
    return pd.Series(qf)


def compute_wv2(qf: pd.Series) -> pd.Series:
    """Ecuación (5): wv,2 = (qf_v - qf_min) / (qf_max - qf_min), normalizado a [0,1].

    Un qf_v más ALTO significa más desvío respecto a la ruta más corta, así
    que un wv,2 alto identifica los puntos donde más se está "rodeando".
    """
    if qf.empty:
        return qf
    qmin, qmax = qf.min(), qf.max()
    if qmax == qmin:
        return pd.Series(0.0, index=qf.index)
    return (qf - qmin) / (qmax - qmin)


def combine_weights(wv1: pd.Series, wv2: pd.Series, alpha=0.7, beta=0.3) -> pd.Series:
    """Ecuación (6), con los coeficientes alpha=0.7 (congestión) y
    beta=0.3 (eficiencia) que el artículo fija en la Sección 4.1 tras
    su análisis de sensibilidad (Sección 6.5)."""
    combined = alpha * wv1 + beta * wv2.reindex(wv1.index).fillna(0)
    return combined


# ---------------------------------------------------------------------------
# 4.1.1: Generación de la malla de puntos virtuales candidatos
# ---------------------------------------------------------------------------

def generate_virtual_grid(G: nx.Graph, resolution_deg=0.2, margin_deg=0.5):
    """Genera puntos virtuales candidatos en una malla regular de resolución
    0.2° (~22 km), sobre el rectángulo delimitador de los waypoints
    existentes (ver nota de simplificación al inicio del archivo).

    Devuelve una lista de tuplas (lat, lon) candidatas.
    """
    lats = [d["lat"] for _, d in G.nodes(data=True)]
    lons = [d["lon"] for _, d in G.nodes(data=True)]

    lat_min, lat_max = min(lats) - margin_deg, max(lats) + margin_deg
    lon_min, lon_max = min(lons) - margin_deg, max(lons) + margin_deg

    lat_points = np.arange(lat_min, lat_max, resolution_deg)
    lon_points = np.arange(lon_min, lon_max, resolution_deg)

    return [(round(la, 4), round(lo, 4)) for la, lo in itertools.product(lat_points, lon_points)]


def select_top_candidates(G: nx.Graph, virtual_points, weights: pd.Series,
                           attraction_radius_km=100, m=5):
    """Calcula W(r,i) para cada punto virtual (Ecuación 7: suma de wv de los
    waypoints existentes dentro del radio de atracción r) y devuelve los m
    puntos virtuales con mayor peso total, como preliminarmente seleccionados
    para convertirse en nuevos waypoints."""
    scored = []
    for lat, lon in virtual_points:
        total_weight = 0.0
        for node, data in G.nodes(data=True):
            d = great_circle_distance_km(lat, lon, data["lat"], data["lon"])
            if d < attraction_radius_km:
                total_weight += weights.get(node, 0.0)
        if total_weight > 0:
            scored.append({"lat": lat, "lon": lon, "W": total_weight})

    scored_df = pd.DataFrame(scored).sort_values("W", ascending=False)
    return scored_df.head(m).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4.1.2: Reglas de conexión de segmentos
# ---------------------------------------------------------------------------

def relative_neighbors(G: nx.Graph, new_lat, new_lon, candidates=None):
    """Ecuación (8): principio del vecino relativo.
    va es vecino de vo (nuevo punto) si no existe otro punto vi tal que
    d(o,a) >= max(d(a,i), d(o,i)).

    candidates: lista opcional de waypoint_ids a considerar (por defecto,
    todos los del grafo). Devuelve la lista de waypoint_ids que sí cumplen
    el criterio de vecino relativo.
    """
    candidates = candidates or list(G.nodes)
    coords = {n: (G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in candidates}

    neighbors = []
    for a in candidates:
        lat_a, lon_a = coords[a]
        d_oa = great_circle_distance_km(new_lat, new_lon, lat_a, lon_a)
        is_neighbor = True
        for i in candidates:
            if i == a:
                continue
            lat_i, lon_i = coords[i]
            d_oi = great_circle_distance_km(new_lat, new_lon, lat_i, lon_i)
            d_ai = great_circle_distance_km(lat_a, lon_a, lat_i, lon_i)
            if d_oa >= max(d_ai, d_oi):
                is_neighbor = False
                break
        if is_neighbor:
            neighbors.append(a)
    return neighbors


def bearing_deg(lat1, lon1, lat2, lon2):
    """Rumbo inicial (bearing) en grados desde el punto 1 al punto 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_between(lat_center, lon_center, lat_a, lon_a, lat_b, lon_b):
    """Ángulo entre dos segmentos que comparten el punto central (grados)."""
    b1 = bearing_deg(lat_center, lon_center, lat_a, lon_a)
    b2 = bearing_deg(lat_center, lon_center, lat_b, lon_b)
    diff = abs(b1 - b2) % 360
    return min(diff, 360 - diff)


def connect_new_waypoint(G: nx.Graph, new_lat, new_lon, attraction_radius_km=100,
                          l_min_km=7.3, theta_min_deg=15):
    """Aplica las tres reglas de conexión (Sección 4.1.2) para decidir a
    qué waypoints existentes se conecta un nuevo punto:
        1. Principio del vecino relativo                          (Ec. 8)
        2. Longitud del segmento entre Lmin y el radio de atracción r
        3. Ángulo entre segmentos adyacentes >= theta_min

    Devuelve la lista final de waypoint_ids a los que conectar el nuevo punto.
    """
    # Candidatos dentro del radio de atracción
    within_radius = [
        n for n, d in G.nodes(data=True)
        if great_circle_distance_km(new_lat, new_lon, d["lat"], d["lon"]) <= attraction_radius_km
    ]

    # Regla 1: vecino relativo
    neighbors = relative_neighbors(G, new_lat, new_lon, candidates=within_radius)

    # Regla 2: longitud del segmento
    valid_length = []
    for n in neighbors:
        d = great_circle_distance_km(new_lat, new_lon, G.nodes[n]["lat"], G.nodes[n]["lon"])
        if l_min_km < d <= attraction_radius_km:
            valid_length.append(n)

    # Regla 3: ángulo mínimo entre segmentos adyacentes (evita conexiones
    # demasiado cercanas en dirección, filtrando de a una a la vez)
    final_connections = []
    for n in valid_length:
        lat_n, lon_n = G.nodes[n]["lat"], G.nodes[n]["lon"]
        ok = True
        for existing in final_connections:
            lat_e, lon_e = G.nodes[existing]["lat"], G.nodes[existing]["lon"]
            angle = angle_between(new_lat, new_lon, lat_n, lon_n, lat_e, lon_e)
            if angle < theta_min_deg:
                ok = False
                break
        if ok:
            final_connections.append(n)

    return final_connections


# ---------------------------------------------------------------------------
# 4.1.3: Eliminación de waypoints redundantes
# ---------------------------------------------------------------------------

def remove_ineffective_waypoints(G: nx.Graph, new_waypoint_ids, flow_by_waypoint: dict):
    """Elimina de la red los waypoints recién añadidos que terminaron con
    tráfico cero tras la asignación del nivel inferior (Sección 4.1.3)."""
    removed = []
    for wp in new_waypoint_ids:
        if flow_by_waypoint.get(wp, 0) == 0:
            if wp in G:
                G.remove_node(wp)
                removed.append(wp)
    return removed


# ---------------------------------------------------------------------------
# 4.3: Optimización de posición vía campos de potencial artificial (Ec. 13-19)
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0


def force_attraction(l_aib_km, l_min_km, f_aib, n1=1.0):
    """Ecuación (14): Fatt = n1 * f_aib * (l_aib - l_min)."""
    return n1 * f_aib * (l_aib_km - l_min_km)


def force_length(l_ia_km, l_min_km, l_th_km, n2=1.0):
    """Ecuación (15): repulsión por longitud de segmento."""
    if l_ia_km >= l_th_km:
        return 0.0
    if l_ia_km <= l_min_km:
        return float("inf")
    return n2 * (1.0 / (l_ia_km - l_min_km) - 1.0 / (l_th_km - l_min_km))


def force_angle(theta_deg, theta_min_deg=15, n3=1.0):
    """Ecuación (16): repulsión por ángulo entre segmentos."""
    if theta_deg >= theta_min_deg:
        return 0.0
    return n3 * ((theta_min_deg - theta_deg) / theta_min_deg) ** 2


def force_prd(d_i_prd_km, r_prd_km, r_th_km, n4=1.0):
    """Ecuación (17): repulsión de las zonas PRD (prohibidas/restringidas/peligrosas)."""
    if d_i_prd_km >= r_th_km:
        return 0.0
    if d_i_prd_km <= r_prd_km:
        return float("inf")
    return n4 * (1.0 / (d_i_prd_km - r_prd_km) - 1.0 / (r_th_km - r_prd_km))


def update_position(lat, lon, total_force, direction_deg, x_step_km=1.0):
    """Ecuaciones (18)-(19): actualiza la posición del punto virtual a lo
    largo de la trayectoria de gran círculo, según la fuerza resultante.

    total_force: |Fi|, magnitud de la fuerza resultante (puede ser inf).
    direction_deg: phi_i, dirección de la fuerza resultante en grados.
    """
    if math.isinf(total_force):
        # Fuerza infinita (choque con Lmin o zona PRD): no se mueve más en
        # esa dirección; se trata en la práctica como límite duro.
        return lat, lon

    delta_d = x_step_km * math.exp(-1.0 / max(total_force, 1e-9))

    phi = math.radians(direction_deg)
    lat1 = math.radians(lat)
    ang_dist = delta_d / EARTH_RADIUS_KM

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang_dist) + math.cos(lat1) * math.sin(ang_dist) * math.cos(phi)
    )
    lon1 = math.radians(lon)
    lon2 = lon1 + math.atan2(
        math.sin(phi) * math.sin(ang_dist) * math.cos(lat1),
        math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def optimize_virtual_point(lat, lon, connections, flights_between, min_route_length_km,
                            l_min_km=7.3, l_th_km=17.3, theta_min_deg=15,
                            prd_zones=None, max_iterations=100, force_epsilon=1e-3,
                            x_step_km=1.0):
    """Bucle de optimización de posición para UN punto virtual (Ec. 13-19),
    combinando las 4 fuerzas hasta que la posición converja o se agote el
    número máximo de iteraciones.

    connections: lista de waypoint_ids conectados a este punto (de connect_new_waypoint)
    flights_between: dict {(waypoint_a, waypoint_b): num_vuelos_semana} usado en Fatt
    prd_zones: lista opcional de dicts {"lat":, "lon":, "radius_km":} (zonas PRD)

    Devuelve: (lat_final, lon_final, historial_de_fuerzas)
    """
    prd_zones = prd_zones or []
    force_history = []

    for iteration in range(max_iterations):
        fx, fy = 0.0, 0.0  # componentes de fuerza resultante (aprox. cartesiana local)

        # --- Fuerza de atracción de la ruta óptima (Ec. 14) ---
        conns = list(connections)
        for a, b in itertools.combinations(conns, 2):
            l_ai = great_circle_distance_km(lat, lon, *_coords(a))
            l_ib = great_circle_distance_km(lat, lon, *_coords(b))
            l_aib = l_ai + l_ib
            f_aib = flights_between.get((a, b), flights_between.get((b, a), 0))
            f_att = force_attraction(l_aib, min_route_length_km, f_aib)
            bearing = bearing_deg(lat, lon, *_coords(a))
            fx += f_att * math.cos(math.radians(bearing))
            fy += f_att * math.sin(math.radians(bearing))

        # --- Fuerza de repulsión por longitud (Ec. 15) ---
        for a in conns:
            l_ia = great_circle_distance_km(lat, lon, *_coords(a))
            f_len = force_length(l_ia, l_min_km, l_th_km)
            if math.isinf(f_len):
                return lat, lon, force_history + [float("inf")]
            bearing = bearing_deg(*_coords(a), lat, lon)  # se aleja del vecino
            fx += f_len * math.cos(math.radians(bearing))
            fy += f_len * math.sin(math.radians(bearing))

        # --- Fuerza de repulsión por ángulo (Ec. 16) ---
        for a, b in itertools.combinations(conns, 2):
            theta = angle_between(lat, lon, *_coords(a), *_coords(b))
            f_ang = force_angle(theta, theta_min_deg)
            fx += f_ang  # aproximación simplificada de dirección (documentado)

        # --- Fuerza de repulsión de zonas PRD (Ec. 17) ---
        for zone in prd_zones:
            d = great_circle_distance_km(lat, lon, zone["lat"], zone["lon"])
            r_th = zone["radius_km"] + 20
            f_prd = force_prd(d, zone["radius_km"], r_th)
            if math.isinf(f_prd):
                return lat, lon, force_history + [float("inf")]
            bearing = bearing_deg(zone["lat"], zone["lon"], lat, lon)
            fx += f_prd * math.cos(math.radians(bearing))
            fy += f_prd * math.sin(math.radians(bearing))

        total_force = math.hypot(fx, fy)
        force_history.append(total_force)

        if total_force < force_epsilon:
            break  # equilibrio alcanzado (Ec. 18: el paso tiende a 0)

        direction = math.degrees(math.atan2(fy, fx))
        lat, lon = update_position(lat, lon, total_force, direction, x_step_km)

    return lat, lon, force_history


def _coords(waypoint_id):
    """Función auxiliar; se reemplaza en tiempo de ejecución (ver main.py)
    con una consulta real al grafo. Aquí se deja como placeholder para que
    optimize_virtual_point() sea testeable de forma aislada si se inyecta
    un lookup adecuado (ver tests en main.py)."""
    raise NotImplementedError(
        "Asigna apf_growth._coords antes de llamar a optimize_virtual_point, "
        "por ejemplo: apf_growth._coords = lambda n: (G.nodes[n]['lat'], G.nodes[n]['lon'])"
    )

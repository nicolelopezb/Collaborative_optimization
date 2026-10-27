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


def compute_max_excess_ratio(flow_df: pd.DataFrame, cap_df: pd.DataFrame, tol=0.05):
    """Criterio II de convergencia (Sec. 3.2 del artículo): el tráfico
    excedente horario en cada waypoint no debe superar el 5 % de su capacidad,
    para que el resto pueda resolverse con medidas tácticas de ATC.

    Para cada (waypoint, nivel, franja) calcula max(f - C, 0) / C.
    Devuelve (max_ratio, n_violations): la razón máxima observada y cuántas
    combinaciones (waypoint, nivel, franja) superan `tol`.
    """
    merged = flow_df.merge(cap_df, on=["waypoint_id", "level", "time_slot"], how="right")
    merged["flights"] = merged["flights"].fillna(0)
    merged = merged[merged["capacity"] > 0]
    if merged.empty:
        return 0.0, 0
    ratio = (merged["flights"] - merged["capacity"]).clip(lower=0) / merged["capacity"]
    return float(ratio.max()), int((ratio > tol).sum())


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

    if not scored:
        return pd.DataFrame(columns=["lat", "lon", "W"])
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

    # Regla 3: ángulo mínimo entre segmentos adyacentes (Fig. 5 del artículo).
    # Se comprueba en los dos lugares donde se forman ángulos con el segmento
    # nuevo: (a) en el propio punto nuevo, entre dos de sus conexiones; y
    # (b) en el waypoint existente n, entre el segmento nuevo n-vo y cada
    # segmento que ya salía de n.
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
            for j in G.neighbors(n):
                lat_j, lon_j = G.nodes[j]["lat"], G.nodes[j]["lon"]
                if angle_between(lat_n, lon_n, new_lat, new_lon, lat_j, lon_j) < theta_min_deg:
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


def _destination(lat, lon, bearing_deg_, dist_km):
    """Punto de llegada tras avanzar dist_km por gran círculo desde (lat, lon)
    con rumbo bearing_deg_ (0° = norte, 90° = este). Ec. (19)."""
    phi = math.radians(bearing_deg_)
    lat1 = math.radians(lat)
    ang_dist = dist_km / EARTH_RADIUS_KM

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang_dist) + math.cos(lat1) * math.sin(ang_dist) * math.cos(phi)
    )
    lon2 = math.radians(lon) + math.atan2(
        math.sin(phi) * math.sin(ang_dist) * math.cos(lat1),
        math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def update_position(lat, lon, total_force, direction_deg, x_step_km=1.0):
    """Ecuaciones (18)-(19): actualiza la posición del punto virtual a lo
    largo de la trayectoria de gran círculo, según la fuerza resultante.

    total_force: |Fi|, magnitud de la fuerza resultante.
    direction_deg: phi_i, rumbo de la fuerza resultante (0° = norte).
    """
    if math.isinf(total_force):
        return lat, lon
    delta_d = x_step_km * math.exp(-1.0 / max(total_force, 1e-9))   # Ec. (18)
    return _destination(lat, lon, direction_deg, delta_d)


# ---------------------------------------------------------------------------
# Geometría local auxiliar: plano tangente en km (este, norte)
# ---------------------------------------------------------------------------

# Las Ec. (15) y (17) devuelven +inf cuando el punto cruza un límite duro.
# Un +inf no se puede sumar vectorialmente (el punto quedaría congelado), así
# que dentro del bucle se usa un tope finito muy grande y, además, los pasos
# que violarían el límite se rechazan (ver _feasible y optimize_virtual_point).
BIG_FORCE = 1e6


def _to_enu(lat0, lon0, lat, lon):
    """Desplazamiento (este_km, norte_km) de (lat, lon) respecto a (lat0, lon0)."""
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * EARTH_RADIUS_KM
    y = math.radians(lat - lat0) * EARTH_RADIUS_KM
    return x, y


def _unit(x, y):
    n = math.hypot(x, y)
    return (x / n, y / n) if n > 1e-12 else (0.0, 0.0)


def _closest_on_segment(p, a, b):
    """Punto del segmento a-b más cercano a p (todos en coordenadas planas)."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    l2 = abx * abx + aby * aby
    if l2 < 1e-12:
        return a
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / l2
    t = max(0.0, min(1.0, t))
    return (a[0] + t * abx, a[1] + t * aby)


def _angle_deg(u, v):
    """Ángulo (0-180°) entre dos vectores planos."""
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-12 or nv < 1e-12:
        return 180.0
    c = (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def count_pair_flows(paths, connections, max_hops=3):
    """f_aib de la Ec. (14): vuelos por semana que cruzan a y b con a lo sumo
    max_hops saltos de separación en su ruta, es decir, los vuelos que
    podrían desviarse por un waypoint nuevo conectado a ambos.

    paths: iterable de rutas (listas de waypoint_id), una por vuelo.
    Devuelve dict {(a, b): n_vuelos} para cada par de `connections`.
    """
    conns = set(connections)
    flows = {}
    for path in paths:
        idx = {}
        for pos, n in enumerate(path):
            if n in conns and n not in idx:
                idx[n] = pos
        if len(idx) < 2:
            continue
        for a, b in itertools.combinations(sorted(idx), 2):
            if abs(idx[a] - idx[b]) <= max_hops:
                flows[(a, b)] = flows.get((a, b), 0) + 1
    return flows


def _prd_distance(rel_conns, zone_center_rel):
    """Distancia mínima del centro de una zona PRD al punto virtual (origen) y
    a los segmentos que lo unen con sus conexiones."""
    d = math.hypot(*zone_center_rel)
    for a_rel in rel_conns:
        c = _closest_on_segment(zone_center_rel, (0.0, 0.0), a_rel)
        d = min(d, math.hypot(zone_center_rel[0] - c[0], zone_center_rel[1] - c[1]))
    return d


def _node_xy(G, n):
    return G.nodes[n]["lat"], G.nodes[n]["lon"]


def _angles_ok(G, lat, lon, connections, theta_min_deg):
    """True si todos los ángulos entre segmentos adyacentes son >= theta_min:
    en cada waypoint existente `a` (segmento nuevo vs. segmentos previos) y en
    el propio punto nuevo (entre pares de sus segmentos)."""
    pos = {a: _node_xy(G, a) for a in connections}
    for a in connections:
        vo = _to_enu(*pos[a], lat, lon)
        for j in G.neighbors(a):
            if _angle_deg(vo, _to_enu(*pos[a], *_node_xy(G, j))) < theta_min_deg:
                return False
    rel = {a: _to_enu(lat, lon, *pos[a]) for a in connections}
    for a, b in itertools.combinations(connections, 2):
        if _angle_deg(rel[a], rel[b]) < theta_min_deg:
            return False
    return True


def _feasible(G, lat, lon, connections, start, l_min_km, r_km, prd_zones, max_move_km,
              theta_min_deg=15.0):
    """Restricciones duras: Ec. (11) rango de movimiento, longitud de segmento
    en (Lmin, r], ángulo mínimo entre segmentos, y Ec. (12) sin intersección
    con zonas PRD. El artículo exige que se cumplan estrictamente; las fuerzas
    de las Ec. (15)-(17) las empujan, pero con n1 = n2 = n3 = 1 la atracción
    (f * km, de miles) las supera con facilidad, por eso además se imponen aquí."""
    if great_circle_distance_km(start[0], start[1], lat, lon) > max_move_km:
        return False
    rel = []
    for a in connections:
        la, loa = _node_xy(G, a)
        l = great_circle_distance_km(lat, lon, la, loa)
        if l <= l_min_km or l > r_km:
            return False
        rel.append(_to_enu(lat, lon, la, loa))
    for z in prd_zones:
        c = _to_enu(lat, lon, z["lat"], z["lon"])
        if _prd_distance(rel, c) <= z["radius_km"]:
            return False
    return _angles_ok(G, lat, lon, connections, theta_min_deg)


def pair_route_cost(G, lat, lon, connections, pair_flows):
    """Ec. (10) restringida a los vuelos que pasarían por el nuevo punto:
    suma de f_ab * (l_ai + l_ib) en km-vuelo por semana."""
    cost = 0.0
    for a, b in itertools.combinations(connections, 2):
        f = pair_flows.get((a, b), pair_flows.get((b, a), 0))
        if f <= 0:
            continue
        l_ai = great_circle_distance_km(lat, lon, *_node_xy(G, a))
        l_ib = great_circle_distance_km(lat, lon, *_node_xy(G, b))
        cost += f * (l_ai + l_ib)
    return cost


def total_force(G, lat, lon, connections, pair_flows, l_min_km=7.3, l_th_km=None,
                theta_min_deg=15.0, prd_zones=None, n1=1.0, n2=1.0, n3=1.0, n4=1.0):
    """Fuerza resultante F_i sobre el punto virtual (Ec. 13), como vector
    (fx_este, fy_norte) en km-equivalentes. Suma las cuatro componentes:

      Fatt    (Ec. 14): atrae el punto hacia la recta a-b de la ruta óptima,
                        con magnitud n1 * f_aib * (l_aib - d_ab).
      Flength (Ec. 15): lo aleja de cada vecino a < Lth.
      Fangle  (Ec. 16): ensancha cualquier ángulo < theta_min, tanto en el
                        propio punto (entre dos de sus segmentos) como en cada
                        waypoint existente a donde llega un segmento nuevo.
      FPRD    (Ec. 17): lo aleja de zonas prohibidas/restringidas/peligrosas.
    """
    if l_th_km is None:
        l_th_km = l_min_km + 10.0           # Lth = Lmin + 10 km (Sec. 4.3)
    prd_zones = prd_zones or []
    fx = fy = 0.0

    pos = {a: _node_xy(G, a) for a in connections}
    rel = {a: _to_enu(lat, lon, *pos[a]) for a in connections}

    # --- Atracción de la ruta óptima (Ec. 14) ---
    for a, b in itertools.combinations(connections, 2):
        f_ab = pair_flows.get((a, b), pair_flows.get((b, a), 0))
        if f_ab <= 0:
            continue
        l_aib = (great_circle_distance_km(lat, lon, *pos[a])
                 + great_circle_distance_km(lat, lon, *pos[b]))
        d_ab = great_circle_distance_km(*pos[a], *pos[b])
        mag = force_attraction(l_aib, d_ab, f_ab, n1)
        if mag <= 0:
            continue
        cx, cy = _closest_on_segment((0.0, 0.0), rel[a], rel[b])
        ux, uy = _unit(cx, cy)
        fx += mag * ux
        fy += mag * uy

    # --- Repulsión por longitud de segmento (Ec. 15) ---
    for a in connections:
        l = great_circle_distance_km(lat, lon, *pos[a])
        mag = min(force_length(l, l_min_km, l_th_km, n2), BIG_FORCE)
        if mag > 0:
            ux, uy = _unit(-rel[a][0], -rel[a][1])      # se aleja del vecino
            fx += mag * ux
            fy += mag * uy

    # --- Repulsión por ángulo (Ec. 16) ---
    # (a) en cada waypoint existente `a`: ángulo entre el segmento nuevo a-vo y
    #     cada segmento ya existente a-j (Fig. 5 del artículo).
    for a in connections:
        vo = _to_enu(*pos[a], lat, lon)                 # a -> vo
        for j in G.neighbors(a):
            j_vec = _to_enu(*pos[a], *_node_xy(G, j))   # a -> j
            theta = _angle_deg(vo, j_vec)
            mag = force_angle(theta, theta_min_deg, n3)
            if mag > 0:
                cross = j_vec[0] * vo[1] - j_vec[1] * vo[0]
                # vo está a la izquierda (cross > 0) de j: lo empujamos más a la
                # izquierda; si no, a la derecha. Así el ángulo crece.
                px, py = (-vo[1], vo[0]) if cross >= 0 else (vo[1], -vo[0])
                ux, uy = _unit(px, py)
                fx += mag * ux
                fy += mag * uy
    # (b) en el propio punto: ángulo entre dos de sus segmentos nuevos. Acercarse
    #     al segmento a-b es lo que abre el ángulo que subtiende en vo.
    for a, b in itertools.combinations(connections, 2):
        theta = _angle_deg(rel[a], rel[b])
        mag = force_angle(theta, theta_min_deg, n3)
        if mag > 0:
            cx, cy = _closest_on_segment((0.0, 0.0), rel[a], rel[b])
            ux, uy = _unit(cx, cy)
            fx += mag * ux
            fy += mag * uy

    # --- Repulsión de zonas PRD (Ec. 17) ---
    rel_list = list(rel.values())
    for z in prd_zones:
        c = _to_enu(lat, lon, z["lat"], z["lon"])
        d = _prd_distance(rel_list, c)
        mag = min(force_prd(d, z["radius_km"], z["radius_km"] + 20.0, n4), BIG_FORCE)
        if mag > 0:
            ux, uy = _unit(-c[0], -c[1])                # desde el centro hacia vo
            fx += mag * ux
            fy += mag * uy

    return fx, fy


def optimize_virtual_point(G, lat, lon, connections, pair_flows, l_min_km=7.3,
                           attraction_radius_km=150.0, theta_min_deg=15.0,
                           prd_zones=None, max_move_km=50.0, x_step_km=1.0,
                           max_iterations=300, force_epsilon=1e-3, pos_tol_km=1e-3,
                           max_perturbations=3, cost_tol=1e-3, perturb_km=5.0, seed=0):
    """Optimiza la posición de UN waypoint nuevo con la topología fija
    (Sec. 4.2-4.3): mueve el punto bajo la fuerza resultante (Ec. 13-19) hasta
    el equilibrio, y cuando queda en equilibrio aplica la perturbación
    estocástica descrita en el artículo para escapar de mínimos locales.

    Los límites duros (Lmin, radio r, PRD, rango de movimiento) se imponen
    rechazando pasos que los violen (el +inf de las Ec. 15 y 17).

    Devuelve dict: lat, lon, cost0, cost, iterations, perturbations, feasible.
    `feasible` es False si la posición de partida ya violaba un límite duro.
    """
    prd_zones = prd_zones or []
    rng = np.random.default_rng(seed)
    start = (lat, lon)
    kw = dict(l_min_km=l_min_km, theta_min_deg=theta_min_deg, prd_zones=prd_zones)

    def ok(la, lo):
        return _feasible(G, la, lo, connections, start, l_min_km, attraction_radius_km,
                         prd_zones, max_move_km, theta_min_deg)

    cost0 = pair_route_cost(G, lat, lon, connections, pair_flows)
    start_ok = ok(lat, lon)
    best = (cost0, lat, lon) if start_ok else (float("inf"), lat, lon)
    eq_costs, n_pert, it = [], 0, 0

    while it < max_iterations:
        it += 1
        fx, fy = total_force(G, lat, lon, connections, pair_flows, **kw)
        F = math.hypot(fx, fy)

        moved = False
        if F >= force_epsilon:
            direction = math.degrees(math.atan2(fx, fy))    # rumbo: 0° = norte
            step = x_step_km
            for _ in range(8):                              # rechaza pasos inviables
                new_lat, new_lon = update_position(lat, lon, F, direction, step)
                if ok(new_lat, new_lon) or not ok(lat, lon):
                    moved = great_circle_distance_km(lat, lon, new_lat, new_lon) > pos_tol_km
                    break
                step /= 2.0
            else:
                new_lat, new_lon = lat, lon
            if moved:
                lat, lon = new_lat, new_lon
                if ok(lat, lon):
                    c = pair_route_cost(G, lat, lon, connections, pair_flows)
                    if c < best[0]:
                        best = (c, lat, lon)
                continue

        # Equilibrio (|F| < eps, paso < tolerancia o bloqueado por un límite duro)
        c = pair_route_cost(G, lat, lon, connections, pair_flows)
        if ok(lat, lon) and c < best[0]:
            best = (c, lat, lon)
        if eq_costs and abs(c - eq_costs[-1]) <= cost_tol * max(abs(eq_costs[-1]), 1e-9):
            break                                           # estable: dos equilibrios iguales
        eq_costs.append(c)
        if n_pert >= max_perturbations:
            break
        n_pert += 1                                         # perturbación estocástica
        for _ in range(20):
            ang = rng.uniform(0, 360)
            mag = rng.uniform(0.2, 1.0) * perturb_km
            cand = _destination(best[1], best[2], ang, mag)
            if ok(*cand):
                lat, lon = cand
                break

    final_ok = best[0] != float("inf")
    return {"lat": best[1], "lon": best[2], "cost0": cost0, "cost": best[0] if final_ok else cost0,
            "iterations": it, "perturbations": n_pert, "feasible": final_ok}


def validate_new_waypoint(G, lat, lon, connections, l_min_km=7.3, attraction_radius_km=150.0,
                          theta_min_deg=15.0, prd_zones=None):
    """Comprueba las restricciones del artículo sobre un waypoint nuevo ya
    colocado (Sec. 4.1.2 y Ec. 12). Devuelve la lista de violaciones (vacía
    si cumple todo)."""
    prd_zones = prd_zones or []
    problems = []
    pos = {a: _node_xy(G, a) for a in connections}
    rel = {a: _to_enu(lat, lon, *pos[a]) for a in connections}

    for a in connections:
        l = great_circle_distance_km(lat, lon, *pos[a])
        if not (l_min_km < l <= attraction_radius_km):
            problems.append(f"longitud {a}: {l:.1f} km fuera de ({l_min_km}, {attraction_radius_km}]")
        vo = _to_enu(*pos[a], lat, lon)
        for j in G.neighbors(a):
            th = _angle_deg(vo, _to_enu(*pos[a], *_node_xy(G, j)))
            if th < theta_min_deg:
                problems.append(f"ángulo en {a} entre segmento nuevo y {a}-{j}: {th:.1f}°")
    for a, b in itertools.combinations(connections, 2):
        th = _angle_deg(rel[a], rel[b])
        if th < theta_min_deg:
            problems.append(f"ángulo en el nuevo punto entre {a} y {b}: {th:.1f}°")
    for z in prd_zones:
        c = _to_enu(lat, lon, z["lat"], z["lon"])
        if _prd_distance(list(rel.values()), c) <= z["radius_km"]:
            problems.append("intersecta una zona PRD")
    return problems

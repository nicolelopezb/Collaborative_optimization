"""
pathing.py
-----------
Implementa el paso "K-shortest paths" descrito en la Sección 5 del
artículo: para cada par origen-destino, genera hasta 3 rutas candidatas
ordenadas de menor a mayor longitud (Sección 5, paso 1 del proceso de
extracción de parámetros de simulación).

Estas rutas candidatas son las que luego usa el modelo MILP (milp_model.py)
para decidir cuál escoge cada vuelo (Ec. 32, 37).

Uso:
    import networkx as nx
    from network import build_network
    from pathing import k_shortest_paths, build_candidate_routes

    G = build_network(...)
    routes = build_candidate_routes(G, demand_df, k=3)
"""

import itertools
import networkx as nx
import pandas as pd


def k_shortest_paths(G: nx.Graph, source, target, k=3, weight="length_km"):
    """Devuelve hasta k rutas más cortas (como listas de nodos) entre
    source y target, ordenadas de menor a mayor longitud."""
    if source not in G or target not in G:
        return []
    if source == target:
        return []
    try:
        generator = nx.shortest_simple_paths(G, source, target, weight=weight)
        paths = list(itertools.islice(generator, k))
        return paths
    except nx.NetworkXNoPath:
        return []


def path_length(G: nx.Graph, path, weight="length_km"):
    """Longitud total de una ruta (lista de nodos) sumando los pesos de sus aristas."""
    return sum(G[u][v][weight] for u, v in zip(path[:-1], path[1:]))


def great_circle_length(G: nx.Graph, source, target):
    """Distancia directa (gran círculo) entre origen y destino, usada
    como D(a) en la Ec. 26 y 36 (distancia adicional = L(r) - D(a))."""
    from network import great_circle_distance_km
    lat1, lon1 = G.nodes[source]["lat"], G.nodes[source]["lon"]
    lat2, lon2 = G.nodes[target]["lat"], G.nodes[target]["lon"]
    return great_circle_distance_km(lat1, lon1, lat2, lon2)


def build_candidate_routes(G: nx.Graph, demand_df: pd.DataFrame, k=3):
    """Para cada par (origin, destination) único en la demanda, genera hasta
    k rutas candidatas. Devuelve un DataFrame con una fila por ruta candidata:

        origin, destination, route_idx, path (lista de nodos),
        length_km, great_circle_km, extra_km

    route_idx = 0 es la ruta más corta, 1 la segunda, etc. (Ra en el artículo).
    """
    od_pairs = demand_df[["origin", "destination"]].drop_duplicates()

    rows = []
    for _, r in od_pairs.iterrows():
        o, d = r["origin"], r["destination"]
        paths = k_shortest_paths(G, o, d, k=k)
        if not paths:
            continue
        gc_km = great_circle_length(G, o, d)
        for idx, path in enumerate(paths):
            length_km = path_length(G, path)
            rows.append({
                "origin": o,
                "destination": d,
                "route_idx": idx,
                "path": path,
                "length_km": length_km,
                "great_circle_km": gc_km,
                "extra_km": length_km - gc_km,
            })

    routes_df = pd.DataFrame(rows)
    if routes_df.empty:
        print("Aviso: no se generó ninguna ruta candidata. Verifica que los "
              "aeropuertos de origen/destino existan como nodos en el grafo "
              "(revisa la conectividad de la red).")
    else:
        n_od = routes_df[["origin", "destination"]].drop_duplicates().shape[0]
        print(f"Rutas candidatas generadas para {n_od} pares origen-destino "
              f"({len(routes_df)} rutas en total, hasta {k} por par).")
    return routes_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from network import build_network

    G = build_network("data/processed/waypoints.csv", "data/processed/segments.csv")

    # Prueba rápida: buscar rutas entre dos waypoints existentes en la red
    nodes = list(G.nodes)
    if len(nodes) >= 2:
        demo_demand = pd.DataFrame({"origin": [nodes[0]], "destination": [nodes[-1]]})
        routes = build_candidate_routes(G, demo_demand, k=3)
        print(routes[["origin", "destination", "route_idx", "length_km",
                       "great_circle_km", "extra_km"]])

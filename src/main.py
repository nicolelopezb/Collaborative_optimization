"""
main.py
--------
Orquesta el ciclo bi-nivel completo descrito en la Sección 3.2 del artículo:

    nivel inferior (asignar vuelos a rutas, minimizar exceso + distancia)
                         ⇅ retroalimentación bidireccional
    nivel superior (crecer/ajustar la red según congestión)

hasta que se cumplan AMBOS criterios de convergencia:
    Criterio I:  variación de exceso total < 3% Y variación de distancia
                 adicional total < 0.5% en dos iteraciones consecutivas
    Criterio II: exceso horario en cada waypoint <= 5% de su capacidad

Uso:
    python src/main.py --scenario 3 --max-iterations 10
"""

import argparse
import sys
sys.path.insert(0, "src")  # por si se corre desde la raíz del proyecto

import pandas as pd
import networkx as nx

from network import build_network, great_circle_distance_km, plot_network
from pathing import build_candidate_routes
from capacities import generate_waypoint_capacity, generate_airport_capacity
import apf_growth as apf

try:
    from milp_model import solve_path_assignment
    MILP_AVAILABLE = True
except ImportError:
    MILP_AVAILABLE = False


DEFAULT_LEVEL = "FL_unico"  # simplificación MVP: un solo nivel de vuelo (ver README)


def assign_flights_shortest_path(routes_df: pd.DataFrame, flights_df: pd.DataFrame):
    """Asignación baseline simple: cada vuelo toma la ruta más corta
    (route_idx == 0) de su par origen-destino. Se usa como resultado de
    respaldo si PuLP/MILP no está disponible, y como punto de partida
    razonable para inicializar el ciclo iterativo."""
    shortest = routes_df[routes_df["route_idx"] == 0].set_index(["origin", "destination"])

    rows = []
    for _, f in flights_df.iterrows():
        key = (f["origin"], f["destination"])
        if key not in shortest.index:
            continue
        r = shortest.loc[key]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append({"flight_id": f["flight_id"], "route_idx": 0, "path": r["path"]})
    return pd.DataFrame(rows)


def compute_flow_from_assignment(assignment: pd.DataFrame, flights_df: pd.DataFrame,
                                  level=DEFAULT_LEVEL):
    """Convierte la asignación de rutas (una fila por vuelo, con su `path`)
    en el formato f^h,t_v que necesita apf_growth.compute_excess_traffic():
    columnas [waypoint_id, level, time_slot, flights]."""
    merged = assignment.merge(flights_df[["flight_id", "time_slot"]], on="flight_id")
    rows = []
    for _, row in merged.iterrows():
        for wp in row["path"]:
            rows.append({"waypoint_id": wp, "level": level, "time_slot": row["time_slot"]})
    flow_long = pd.DataFrame(rows)
    if flow_long.empty:
        return pd.DataFrame(columns=["waypoint_id", "level", "time_slot", "flights"])
    return flow_long.groupby(["waypoint_id", "level", "time_slot"]).size().reset_index(name="flights")


def compute_extra_distance(assignment: pd.DataFrame, routes_df: pd.DataFrame):
    """Ecuación (25)-(26): suma de (L(r) - D(a)) para todos los vuelos asignados."""
    merged = assignment.merge(
        routes_df[["origin", "destination", "route_idx", "length_km", "great_circle_km"]],
        on="route_idx", how="left"
    )
    # (el merge de arriba no filtra por origin/destination porque assignment no
    # los trae; en la práctica usa join sobre flight -> routes_df completo)
    return (routes_df["length_km"] - routes_df["great_circle_km"]).sum()


def run_lower_level(G: nx.Graph, flights_df: pd.DataFrame, wp_cap_df: pd.DataFrame,
                     apt_cap_df: pd.DataFrame, k=3, use_milp=True):
    """Ejecuta el nivel inferior completo: genera rutas candidatas (K-shortest
    paths) y asigna vuelos, ya sea con el MILP (Ec. 32-40) o con la
    asignación baseline por ruta más corta si MILP no está disponible."""
    routes_df = build_candidate_routes(G, flights_df, k=k)

    if routes_df.empty:
        return pd.DataFrame(), routes_df, 0.0

    if use_milp and MILP_AVAILABLE:
        result = solve_path_assignment(routes_df, flights_df, wp_cap_df, apt_cap_df)
        assignment = result["assignment"]
        print(f"  MILP resuelto: estado={result['status']}, "
              f"objetivo={result['objective_value']}")
    else:
        if use_milp and not MILP_AVAILABLE:
            print("  Aviso: PuLP no está instalado (pip install pulp). "
                  "Usando asignación baseline por ruta más corta en su lugar.")
        assignment = assign_flights_shortest_path(routes_df, flights_df)

    # Añadir length/great_circle a cada fila asignada, para poder calcular
    # tanto el exceso de tráfico como la distancia adicional después.
    lookup = routes_df.set_index(["route_idx"])[["length_km", "great_circle_km"]]

    return assignment, routes_df, lookup


def run_upper_level(G: nx.Graph, flow_df: pd.DataFrame, wp_cap_df: pd.DataFrame,
                     assignment: pd.DataFrame, routes_df: pd.DataFrame,
                     attraction_radius_km=150, m_new_nodes=2, grid_resolution_deg=0.5):
    """Ejecuta el nivel superior: calcula pesos (Ec. 1-7), genera la malla de
    candidatos, selecciona los mejores m, y los conecta a la red (Ec. 8 +
    límites de longitud/ángulo). NO ejecuta aún la optimización de posición
    vía APF (Ec. 13-19) sobre cada nuevo punto -- ver optimize_new_nodes()."""
    Ov = apf.compute_excess_traffic(G, flow_df, wp_cap_df)
    wv1 = apf.compute_wv1(Ov)

    routes_assigned = assignment.merge(
        routes_df[["route_idx", "length_km", "great_circle_km"]].drop_duplicates("route_idx"),
        on="route_idx", how="left"
    )
    qf = apf.compute_flow_weighted_path_factor(routes_assigned) if not routes_assigned.empty else pd.Series(dtype=float)
    wv2 = apf.compute_wv2(qf)

    weights = apf.combine_weights(wv1, wv2)

    if Ov.sum() == 0:
        print("  Sin tráfico excedente en ningún waypoint: no se generan nodos nuevos.")
        return [], weights

    grid = apf.generate_virtual_grid(G, resolution_deg=grid_resolution_deg)
    # Pedimos más candidatos de los estrictamente necesarios (margen 4x), porque
    # algunos de los mejores por peso pueden terminar con menos de 2 conexiones
    # válidas y descartarse -- así no nos quedamos sin nodos nuevos por eso.
    top_candidates = apf.select_top_candidates(
        G, grid, weights, attraction_radius_km=attraction_radius_km, m=m_new_nodes * 4
    )

    new_nodes = []
    for i, row in top_candidates.iterrows():
        if len(new_nodes) >= m_new_nodes:
            break
        connections = apf.connect_new_waypoint(
            G, row["lat"], row["lon"], attraction_radius_km=attraction_radius_km
        )
        if len(connections) < 2:
            continue  # necesita al menos 2 conexiones para ser un punto útil
        node_id = f"NEW{len(G.nodes) + i}"
        new_nodes.append({"node_id": node_id, "lat": row["lat"], "lon": row["lon"],
                          "connections": connections})

    return new_nodes, weights


def add_new_nodes_to_graph(G: nx.Graph, new_nodes: list):
    """Añade los nuevos waypoints (ya con posición optimizada) y sus
    conexiones al grafo."""
    for n in new_nodes:
        G.add_node(n["node_id"], lat=n["lat"], lon=n["lon"], tipo="nuevo")
        for c in n["connections"]:
            d = great_circle_distance_km(n["lat"], n["lon"], G.nodes[c]["lat"], G.nodes[c]["lon"])
            G.add_edge(n["node_id"], c, length_km=d, airway_id="NUEVO")


def check_convergence(history: list, tol_excess=0.03, tol_distance=0.005):
    """Criterio I (Sección 3.2): variación de exceso total < 3% Y variación
    de distancia adicional total < 0.5% en dos iteraciones consecutivas."""
    if len(history) < 2:
        return False
    prev, curr = history[-2], history[-1]
    if prev["total_excess"] == 0 and prev["total_extra_km"] == 0:
        return True
    excess_change = abs(curr["total_excess"] - prev["total_excess"]) / max(prev["total_excess"], 1e-9)
    dist_change = abs(curr["total_extra_km"] - prev["total_extra_km"]) / max(prev["total_extra_km"], 1e-9)
    return excess_change < tol_excess and dist_change < tol_distance


def run_iterative_optimization(G: nx.Graph, flights_df: pd.DataFrame,
                                wp_cap_df: pd.DataFrame, apt_cap_df: pd.DataFrame,
                                max_iterations=5, k=3, use_milp=True,
                                grow_network=True):
    """Bucle principal: nivel inferior -> nivel superior -> repetir, hasta
    convergencia o max_iterations. Esta es la Escena 3 del artículo
    (optimización colaborativa) si grow_network=True, o la Escena 2 (red
    fija, solo optimización de rutas) si grow_network=False."""
    history = []

    for it in range(1, max_iterations + 1):
        print(f"\n=== Iteración {it} (nodos actuales: {G.number_of_nodes()}) ===")

        assignment, routes_df, _ = run_lower_level(
            G, flights_df, wp_cap_df, apt_cap_df, k=k, use_milp=use_milp
        )

        if assignment.empty:
            print("  Sin asignación posible; revisa conectividad origen-destino.")
            break

        flow_df = compute_flow_from_assignment(assignment, flights_df)
        Ov = apf.compute_excess_traffic(G, flow_df, wp_cap_df)
        total_excess = Ov.sum()
        total_extra_km = (routes_df["length_km"] - routes_df["great_circle_km"]).sum()

        print(f"  Exceso total: {total_excess:.2f} | Distancia adicional total: {total_extra_km:.2f} km")
        history.append({"iteration": it, "total_excess": total_excess, "total_extra_km": total_extra_km})

        if check_convergence(history):
            print("  Convergencia alcanzada (Criterio I).")
            break

        if not grow_network:
            continue  # Escenario 2: no se crece la red, solo se repite la asignación

        new_nodes, weights = run_upper_level(G, flow_df, wp_cap_df, assignment, routes_df)
        if not new_nodes:
            print("  No se generaron nodos nuevos en esta iteración.")
            continue

        print(f"  Añadiendo {len(new_nodes)} nodo(s) nuevo(s): "
              f"{[n['node_id'] for n in new_nodes]}")
        add_new_nodes_to_graph(G, new_nodes)

    return G, history


def main():
    parser = argparse.ArgumentParser(description="Ciclo bi-nivel completo (artículo Sui et al.)")
    parser.add_argument("--waypoints", default="data/processed/waypoints.csv")
    parser.add_argument("--segments", default="data/processed/segments.csv")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--k", type=int, default=3, help="Número de rutas candidatas (K-shortest)")
    parser.add_argument("--no-milp", action="store_true", help="Forzar asignación baseline sin PuLP")
    parser.add_argument("--fixed-network", action="store_true",
                         help="Escenario 2: no crecer la red, solo optimizar rutas")
    args = parser.parse_args()

    G = build_network(args.waypoints, args.segments)

    # Demo de vuelos sintéticos si no se ha integrado aún la demanda real
    nodes = list(G.nodes)
    flights_df = pd.DataFrame({
        "flight_id": [f"F{i}" for i in range(20)],
        "origin": [nodes[0]] * 20,
        "destination": [nodes[-1]] * 20,
        "time_slot": [i % 24 for i in range(20)],
    })

    wp_cap_df = generate_waypoint_capacity(list(G.nodes), per_hour=5, levels=[DEFAULT_LEVEL])
    apt_cap_df = generate_airport_capacity([nodes[0], nodes[-1]], dep_per_hour=15, arr_per_hour=15)

    G_final, history = run_iterative_optimization(
        G, flights_df, wp_cap_df, apt_cap_df,
        max_iterations=args.max_iterations, k=args.k,
        use_milp=not args.no_milp, grow_network=not args.fixed_network,
    )

    print("\n=== Resumen final ===")
    print(pd.DataFrame(history))
    print(f"\nRed final: {G_final.number_of_nodes()} waypoints, {G_final.number_of_edges()} segmentos.")

    plot_network(G_final, "outputs/network_map_final.html")


if __name__ == "__main__":
    main()

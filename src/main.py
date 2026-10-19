"""
main.py
--------
Orquesta el ciclo bi-nivel completo (Sección 3.2 del artículo).

NOVEDAD DE ESTA VERSIÓN: usa la demanda REAL de Colombia
(data/processed/demand_weekly.csv, generada por run_demand_extraction.py
a partir del archivo de Aerocivil) en vez de los 20 vuelos sintéticos de
demostración que usaban las versiones anteriores.

Uso:
    python src/main.py --scenario 3 --max-iterations 5
"""

import argparse
import sys
sys.path.insert(0, "src")

import pandas as pd
import networkx as nx

from network import build_network, great_circle_distance_km, plot_network
from pathing import build_candidate_routes
from capacities import generate_waypoint_capacity, generate_airport_capacity
from demand import synthesize_weekly_flights, build_hourly_demand
from airports import connect_airports_to_network
import apf_growth as apf

try:
    from milp_model import solve_path_assignment
    MILP_AVAILABLE = True
except ImportError:
    MILP_AVAILABLE = False


DEFAULT_LEVEL = "FL_unico"


def load_real_demand(demand_weekly_csv: str, G: nx.Graph, seed: int = 42):
    """Carga la demanda semanal real (origin, destination, weekly_frequency)
    y la convierte en vuelos individuales sintéticos distribuidos en la
    semana, tal como espera el resto del pipeline (pathing, MILP, APF).

    IMPORTANTE: se filtran las rutas cuyo origen o destino no exista como
    nodo en la red actual (data/processed/waypoints.csv), ya que de lo
    contrario esos vuelos no podrían asignarse a ninguna ruta. Esto es
    consecuencia directa de la limitación de conectividad ya documentada
    en la bitácora (los aeropuertos aún no están conectados a la red de
    waypoints extraída de ENR3).
    """
    routes_weekly = pd.read_csv(demand_weekly_csv)

    nodes = set(G.nodes)
    before = len(routes_weekly)
    routes_weekly = routes_weekly[
        routes_weekly["origin"].isin(nodes) & routes_weekly["destination"].isin(nodes)
    ]
    after = len(routes_weekly)

    print(f"\nDemanda real cargada desde {demand_weekly_csv}: {before} rutas totales.")
    print(f"De esas, {after} tienen tanto origen como destino presentes en la red "
          f"actual de {G.number_of_nodes()} waypoints.")
    if after == 0:
        print("AVISO CRÍTICO: ninguna ruta de demanda real coincide con nodos de la "
              "red. Esto es esperado si la red (extraída de ENR3, solo waypoints "
              "de ruta) no incluye códigos de aeropuerto (SKBO, SKRG, etc.) como "
              "nodos -- ver pendiente #2 de la bitácora (conectar aeropuertos a la "
              "red). Sin esto, el modelo no puede asignar vuelos reales todavía.")
        return pd.DataFrame(columns=["flight_id", "origin", "destination", "time_slot"])

    if before > 0 and after < before:
        dropped_examples = pd.read_csv(demand_weekly_csv)
        dropped_examples = dropped_examples[
            ~(dropped_examples["origin"].isin(nodes) & dropped_examples["destination"].isin(nodes))
        ].head(5)
        print(f"Ejemplo de rutas descartadas por no estar en la red:\n{dropped_examples}")

    flights_raw = synthesize_weekly_flights(routes_weekly, seed=seed)
    flights_df = flights_raw[["flight_id", "origin", "destination"]].copy()
    flights_df["time_slot"] = flights_raw["week_day"] * 24 + flights_raw["dep_hour"]

    print(f"Vuelos individuales sintéticos generados a partir de la demanda real: "
          f"{len(flights_df)}")
    return flights_df


def assign_flights_shortest_path(routes_df, flights_df):
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


def compute_flow_from_assignment(assignment, flights_df, level=DEFAULT_LEVEL):
    merged = assignment.merge(flights_df[["flight_id", "time_slot"]], on="flight_id")
    rows = []
    for _, row in merged.iterrows():
        for wp in row["path"]:
            rows.append({"waypoint_id": wp, "level": level, "time_slot": row["time_slot"]})
    flow_long = pd.DataFrame(rows)
    if flow_long.empty:
        return pd.DataFrame(columns=["waypoint_id", "level", "time_slot", "flights"])
    return flow_long.groupby(["waypoint_id", "level", "time_slot"]).size().reset_index(name="flights")


def run_lower_level(G, flights_df, wp_cap_df, apt_cap_df, k=3, use_milp=True):
    routes_df = build_candidate_routes(G, flights_df, k=k)
    if routes_df.empty:
        return pd.DataFrame(), routes_df, None

    if use_milp and MILP_AVAILABLE:
        result = solve_path_assignment(routes_df, flights_df, wp_cap_df, apt_cap_df)
        assignment = result["assignment"]
        print(f"  MILP resuelto: estado={result['status']}, objetivo={result['objective_value']}")
    else:
        if use_milp and not MILP_AVAILABLE:
            print("  Aviso: PuLP no está instalado. Usando asignación baseline.")
        assignment = assign_flights_shortest_path(routes_df, flights_df)

    return assignment, routes_df, None


def run_upper_level(G, flow_df, wp_cap_df, assignment, routes_df,
                     attraction_radius_km=150, m_new_nodes=2, grid_resolution_deg=0.5):
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
    top_candidates = apf.select_top_candidates(G, grid, weights, attraction_radius_km=attraction_radius_km,
                                                m=m_new_nodes * 4)
    new_nodes = []
    for i, row in top_candidates.iterrows():
        if len(new_nodes) >= m_new_nodes:
            break
        connections = apf.connect_new_waypoint(G, row["lat"], row["lon"], attraction_radius_km=attraction_radius_km)
        if len(connections) < 2:
            continue
        node_id = f"NEW{len(G.nodes) + i}"
        new_nodes.append({"node_id": node_id, "lat": row["lat"], "lon": row["lon"], "connections": connections})
    return new_nodes, weights


def add_new_nodes_to_graph(G, new_nodes):
    for n in new_nodes:
        G.add_node(n["node_id"], lat=n["lat"], lon=n["lon"], tipo="nuevo")
        for c in n["connections"]:
            d = great_circle_distance_km(n["lat"], n["lon"], G.nodes[c]["lat"], G.nodes[c]["lon"])
            G.add_edge(n["node_id"], c, length_km=d, airway_id="NUEVO")


def check_convergence(history, tol_excess=0.03, tol_distance=0.005):
    if len(history) < 2:
        return False
    prev, curr = history[-2], history[-1]
    if prev["total_excess"] == 0 and prev["total_extra_km"] == 0:
        return True
    excess_change = abs(curr["total_excess"] - prev["total_excess"]) / max(prev["total_excess"], 1e-9)
    dist_change = abs(curr["total_extra_km"] - prev["total_extra_km"]) / max(prev["total_extra_km"], 1e-9)
    return excess_change < tol_excess and dist_change < tol_distance


def run_iterative_optimization(G, flights_df, wp_cap_df, apt_cap_df, max_iterations=5,
                                k=3, use_milp=True, grow_network=True):
    history = []
    for it in range(1, max_iterations + 1):
        print(f"\n=== Iteración {it} (nodos actuales: {G.number_of_nodes()}) ===")
        assignment, routes_df, _ = run_lower_level(G, flights_df, wp_cap_df, apt_cap_df, k=k, use_milp=use_milp)

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
            continue

        new_nodes, weights = run_upper_level(G, flow_df, wp_cap_df, assignment, routes_df)
        if not new_nodes:
            print("  No se generaron nodos nuevos en esta iteración.")
            continue

        print(f"  Añadiendo {len(new_nodes)} nodo(s) nuevo(s): {[n['node_id'] for n in new_nodes]}")
        add_new_nodes_to_graph(G, new_nodes)

    return G, history


def main():
    parser = argparse.ArgumentParser(description="Ciclo bi-nivel completo con demanda real")
    parser.add_argument("--waypoints", default="data/processed/waypoints.csv")
    parser.add_argument("--segments", default="data/processed/segments.csv")
    parser.add_argument("--demand", default="data/processed/demand_weekly.csv",
                         help="CSV de demanda semanal real generado por run_demand_extraction.py")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--no-milp", action="store_true")
    parser.add_argument("--fixed-network", action="store_true")
    parser.add_argument("--use-demo-demand", action="store_true",
                         help="Usar la demanda sintética de demostración en vez de la real")
    parser.add_argument("--no-airport-connection", action="store_true",
                         help="No conectar aeropuertos a la red (para comparar con/sin esa mejora)")
    args = parser.parse_args()

    G = build_network(args.waypoints, args.segments)

    if not args.no_airport_connection:
        G, _ = connect_airports_to_network(G, k_nearest=2, max_distance_km=200)

    if args.use_demo_demand:
        nodes = list(G.nodes)
        flights_df = pd.DataFrame({
            "flight_id": [f"F{i}" for i in range(20)],
            "origin": [nodes[0]] * 20,
            "destination": [nodes[-1]] * 20,
            "time_slot": [i % 24 for i in range(20)],
        })
    else:
        flights_df = load_real_demand(args.demand, G)
        if flights_df.empty:
            print("\nNo hay vuelos reales asignables con la red actual. "
                  "Usa --use-demo-demand para probar con datos sintéticos mientras "
                  "se resuelve la conexión aeropuerto-waypoint (pendiente #2 de la bitácora).")
            return

    wp_cap_df = generate_waypoint_capacity(list(G.nodes), per_hour=5, levels=[DEFAULT_LEVEL])
    airports = list(set(flights_df["origin"]) | set(flights_df["destination"]))
    apt_cap_df = generate_airport_capacity(airports, dep_per_hour=50, arr_per_hour=50)

    G_final, history = run_iterative_optimization(
        G, flights_df, wp_cap_df, apt_cap_df,
        max_iterations=args.max_iterations, k=args.k,
        use_milp=not args.no_milp, grow_network=not args.fixed_network,
    )

    print("\n=== Resumen final ===")
    print(pd.DataFrame(history))
    print(f"\nRed final: {G_final.number_of_nodes()} waypoints, {G_final.number_of_edges()} segmentos.")

    try:
        plot_network(G_final, "outputs/network_map_final.html")
    except ImportError:
        print("(folium no disponible en este entorno; omite el mapa)")


if __name__ == "__main__":
    main()

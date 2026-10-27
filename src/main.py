"""
main.py
--------
Orquesta el ciclo bi-nivel completo (Sección 3.2 del artículo).

Nivel inferior : K-caminos + MILP de asignación de rutas (Sec. 5).
Nivel superior : crecimiento de la red (Sec. 4.1) + posición óptima de cada
                 waypoint nuevo con campos de potencial artificial, APF
                 (Sec. 4.3, Ec. 13-19) + borrado de waypoints sin tráfico (4.1.3).
Convergencia   : Criterio I (variación de exceso y distancia) Y Criterio II
                 (exceso horario <= 5 % de la capacidad en cada waypoint);
                 el ciclo termina solo cuando se cumplen los dos a la vez.

Uso:
    python src/main.py --max-iterations 5
    python src/main.py --prd-zones data/processed/prd_zones.csv   # lat,lon,radius_km
"""

import argparse
import os
import sys
sys.path.insert(0, "src")

import pandas as pd
import networkx as nx

from network import build_network, great_circle_distance_km, plot_network
from pathing import build_candidate_routes
from capacities import (generate_waypoint_capacity, generate_airport_capacity,
                        extend_waypoint_capacity)
from demand import synthesize_weekly_flights, build_hourly_demand
from airports import connect_airports_to_network
import apf_growth as apf

try:
    from milp_model import solve_path_assignment
    MILP_AVAILABLE = True
except ImportError:
    MILP_AVAILABLE = False


DEFAULT_LEVEL = "FL_unico"

# Parámetros del artículo (Sec. 3.2 y 4.1.2)
TOL_EXCESS = 0.03          # Criterio I: variación del exceso total < 3 %
TOL_DISTANCE = 0.005       # Criterio I: variación de la distancia adicional < 0,5 %
TOL_CAPACITY_RATIO = 0.05  # Criterio II: exceso horario <= 5 % de la capacidad
L_MIN_KM = 7.3             # longitud mínima de segmento
THETA_MIN_DEG = 15.0       # ángulo mínimo entre segmentos adyacentes
GRID_RESOLUTION_DEG = 0.2  # resolución de la malla de waypoints virtuales (Sec. 4.1.1)


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


def load_prd_zones(path):
    """Zonas prohibidas/restringidas/peligrosas (PRD) como círculos
    circunscritos: CSV con columnas lat, lon, radius_km (p. ej. extraídas de
    ENR 5.1 del AIP). Si no se da archivo, no hay zonas PRD."""
    if not path:
        return []
    df = pd.read_csv(path)
    zones = df[["lat", "lon", "radius_km"]].to_dict("records")
    print(f"Zonas PRD cargadas: {len(zones)}")
    return zones


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


def assigned_routes_table(assignment, flights_df, routes_df):
    """Una fila por vuelo asignado, con la longitud y la distancia de gran
    círculo de SU ruta elegida (se une por origen, destino y route_idx; unir
    solo por route_idx daría a todos los pares OD la longitud de un único par)."""
    cols = ["flight_id", "route_idx", "path", "origin", "destination",
            "length_km", "great_circle_km"]
    if assignment.empty:
        return pd.DataFrame(columns=cols)
    t = assignment[["flight_id", "route_idx", "path"]].merge(
        flights_df[["flight_id", "origin", "destination"]], on="flight_id")
    t = t.merge(
        routes_df[["origin", "destination", "route_idx", "length_km", "great_circle_km"]],
        on=["origin", "destination", "route_idx"], how="left")
    return t[cols]


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


def add_new_nodes_to_graph(G, new_nodes):
    registry = G.graph.setdefault("new_nodes", set())
    for n in new_nodes:
        G.add_node(n["node_id"], lat=n["lat"], lon=n["lon"], tipo="nuevo")
        registry.add(n["node_id"])
        for c in n["connections"]:
            d = great_circle_distance_km(n["lat"], n["lon"], G.nodes[c]["lat"], G.nodes[c]["lon"])
            G.add_edge(n["node_id"], c, length_km=d, airway_id="NUEVO")


def run_upper_level(G, flow_df, wp_cap_df, assignment, routes_df, flights_df,
                    attraction_radius_km=150, m_new_nodes=2, grid_resolution_deg=GRID_RESOLUTION_DEG,
                    prd_zones=None, seed=0, use_alpha_shape=True, alpha_radius_km=None):
    """Nivel superior: elige dónde crecer (pesos Ec. 1-7), conecta cada nodo
    nuevo (Ec. 8 + longitud + ángulo), y OPTIMIZA su posición con la APF
    (Ec. 13-19) manteniendo la topología fija. Los nodos nuevos solo se
    conectan a waypoints existentes (Sec. 4.1.2) y se añaden a G todos juntos
    al terminar. Devuelve (nuevos_nodos, pesos)."""
    Ov = apf.compute_excess_traffic(G, flow_df, wp_cap_df)
    wv1 = apf.compute_wv1(Ov)

    routes_assigned = assigned_routes_table(assignment, flights_df, routes_df)
    qf = (apf.compute_flow_weighted_path_factor(routes_assigned)
          if not routes_assigned.empty else pd.Series(dtype=float))
    wv2 = apf.compute_wv2(qf)
    weights = apf.combine_weights(wv1, wv2)

    if Ov.sum() == 0:
        print("  Sin tráfico excedente en ningún waypoint: no se generan nodos nuevos.")
        return [], weights

    # Sec. 4.1.1: contorno del espacio aéreo por alpha-shape y malla virtual
    # dentro de su caja delimitadora (solo los centros de celda dentro del contorno).
    boundary = None
    if use_alpha_shape:
        try:
            from airspace_boundary import compute_airspace_boundary
            boundary = compute_airspace_boundary(G, alpha_radius_km)
            G.graph["boundary"] = boundary.polygon
            print(f"  Contorno {boundary.method}: R = {boundary.radius_km:.0f} km "
                  f"({'automático' if boundary.auto else 'manual'}), "
                  f"{boundary.n_parts} polígono(s), {boundary.n_points} nodos")
        except ImportError as e:
            print(f"  Aviso: no se pudo usar alpha-shape ({e}); se usa el rectángulo "
                  f"delimitador. Instala scipy y shapely (pip install -r requirements.txt).")
    grid = apf.generate_virtual_grid(G, resolution_deg=grid_resolution_deg,
                                     boundary=boundary.polygon if boundary else None)
    print(f"  Malla virtual de {grid_resolution_deg}°: {len(grid)} puntos candidatos")
    top_candidates = apf.select_top_candidates(G, grid, weights, attraction_radius_km=attraction_radius_km,
                                                m=m_new_nodes * 4)
    paths = assignment["path"].tolist()
    new_nodes = []
    for _, row in top_candidates.iterrows():
        if len(new_nodes) >= m_new_nodes:
            break
        lat, lon = row["lat"], row["lon"]
        conns = apf.connect_new_waypoint(G, lat, lon, attraction_radius_km=attraction_radius_km,
                                         l_min_km=L_MIN_KM, theta_min_deg=THETA_MIN_DEG)
        if len(conns) < 2:
            continue

        # Topología ya decidida (conexiones). Ahora solo se mueve el punto.
        pair_flows = apf.count_pair_flows(paths, conns)
        opt = apf.optimize_virtual_point(
            G, lat, lon, conns, pair_flows, l_min_km=L_MIN_KM,
            attraction_radius_km=attraction_radius_km, theta_min_deg=THETA_MIN_DEG,
            prd_zones=prd_zones, seed=seed + len(new_nodes))
        problems = apf.validate_new_waypoint(
            G, opt["lat"], opt["lon"], conns, l_min_km=L_MIN_KM,
            attraction_radius_km=attraction_radius_km, theta_min_deg=THETA_MIN_DEG,
            prd_zones=prd_zones)
        if not opt["feasible"] or problems:
            print(f"  Candidato ({lat:.2f}, {lon:.2f}) descartado por la APF: "
                  f"{problems[:2] if problems else 'posición inicial inviable'}")
            continue

        node_id = f"NEW{G.graph.get('next_new_id', 0)}"
        G.graph["next_new_id"] = G.graph.get("next_new_id", 0) + 1
        node = {"node_id": node_id, "lat": opt["lat"], "lon": opt["lon"], "connections": conns,
                "apf": opt}
        new_nodes.append(node)
        moved = great_circle_distance_km(lat, lon, opt["lat"], opt["lon"])
        print(f"  {node_id} ({opt['lat']:.2f}, {opt['lon']:.2f}): {len(conns)} conexiones, APF movió {moved:.1f} km "
              f"({opt['iterations']} it., {opt['perturbations']} perturb.), "
              f"costo de ruta {opt['cost0']:,.0f} -> {opt['cost']:,.0f} km-vuelo")

    # Los nodos nuevos de esta iteración se agregan juntos al final: el artículo
    # exige (Sec. 4.1.2, regla de longitud) que los waypoints a los que se
    # conecta un nodo nuevo sean waypoints EXISTENTES, así que ningún nodo
    # nuevo puede conectarse a otro nodo nuevo de la misma iteración.
    add_new_nodes_to_graph(G, new_nodes)
    return new_nodes, weights


def check_convergence(history, tol_excess=TOL_EXCESS, tol_distance=TOL_DISTANCE,
                      tol_capacity_ratio=TOL_CAPACITY_RATIO):
    """Criterios de convergencia de la Sec. 3.2 del artículo (item 3).
    Devuelve (converged, crit1, crit2). El artículo detiene la iteración
    "only when both criteria are simultaneously satisfied":

    Criterio I : la variación del exceso total es < 3 % y la de la distancia
                 adicional total es < 0,5 %, entre iteraciones consecutivas.
    Criterio II: el tráfico excedente horario en cada waypoint no supera el
                 5 % de su capacidad, para que el resto pueda eliminarse con
                 medidas tácticas de ATC (control de velocidad y separación).
    """
    curr = history[-1]
    crit2 = curr["max_excess_ratio"] <= tol_capacity_ratio

    crit1 = False
    if len(history) >= 2:
        prev = history[-2]

        def rel_change(a, b):
            if a == 0:
                return 0.0 if b == 0 else float("inf")
            return abs(b - a) / abs(a)

        crit1 = (rel_change(prev["total_excess"], curr["total_excess"]) < tol_excess
                 and rel_change(prev["total_extra_km"], curr["total_extra_km"]) < tol_distance)
    return crit1 and crit2, crit1, crit2


def run_iterative_optimization(G, flights_df, wp_cap_df, apt_cap_df, max_iterations=5,
                                k=3, use_milp=True, grow_network=True,
                                wp_capacity_per_hour=5, prd_zones=None, seed=0,
                                m_new_nodes=2, grid_resolution_deg=GRID_RESOLUTION_DEG,
                                use_alpha_shape=True, alpha_radius_km=None):
    history = []
    for it in range(1, max_iterations + 1):
        print(f"\n=== Iteración {it} (nodos actuales: {G.number_of_nodes()}) ===")
        assignment, routes_df, _ = run_lower_level(G, flights_df, wp_cap_df, apt_cap_df, k=k, use_milp=use_milp)

        if assignment.empty:
            print("  Sin asignación posible; revisa conectividad origen-destino.")
            break

        flow_df = compute_flow_from_assignment(assignment, flights_df)

        # 4.1.3: un waypoint añadido que quedó sin ningún vuelo tras la
        # asignación del nivel inferior es una expansión inútil: se borra
        # antes de la siguiente iteración.
        flow_by_wp = flow_df.groupby("waypoint_id")["flights"].sum().to_dict()
        candidates_to_prune = [n for n in G.graph.get("new_nodes", set()) if n in G]
        removed = apf.remove_ineffective_waypoints(G, candidates_to_prune, flow_by_wp)
        if removed:
            G.graph["new_nodes"] -= set(removed)
            print(f"  Eliminados {len(removed)} waypoint(s) nuevo(s) sin tráfico: {removed}")

        Ov = apf.compute_excess_traffic(G, flow_df, wp_cap_df)
        total_excess = Ov.sum()
        routes_assigned = assigned_routes_table(assignment, flights_df, routes_df)
        total_extra_km = (routes_assigned["length_km"] - routes_assigned["great_circle_km"]).sum()
        max_ratio, n_viol = apf.compute_max_excess_ratio(flow_df, wp_cap_df, tol=TOL_CAPACITY_RATIO)

        print(f"  Exceso total: {total_excess:.0f} | Distancia adicional total: {total_extra_km:,.0f} km | "
              f"máx. exceso/capacidad: {max_ratio:.0%} ({n_viol} casos > {TOL_CAPACITY_RATIO:.0%})")
        history.append({"iteration": it, "total_excess": total_excess, "total_extra_km": total_extra_km,
                        "max_excess_ratio": max_ratio, "n_violations_crit2": n_viol,
                        "nodes": G.number_of_nodes(), "removed": len(removed)})

        converged, crit1, crit2 = check_convergence(history)
        history[-1].update({"crit1": crit1, "crit2": crit2})
        print(f"  Criterio I: {'SÍ' if crit1 else 'no'} | Criterio II: {'SÍ' if crit2 else 'no'}")
        if converged:
            print("  Convergencia alcanzada (Criterios I y II a la vez).")
            break

        if not grow_network:
            continue

        new_nodes, weights = run_upper_level(G, flow_df, wp_cap_df, assignment, routes_df, flights_df,
                                             m_new_nodes=m_new_nodes, prd_zones=prd_zones, seed=seed + it,
                                             grid_resolution_deg=grid_resolution_deg,
                                             use_alpha_shape=use_alpha_shape,
                                             alpha_radius_km=alpha_radius_km)
        history[-1]["added"] = len(new_nodes)
        if not new_nodes:
            print("  No se generaron nodos nuevos en esta iteración.")
            continue

        # Los nodos nuevos necesitan capacidad: sin ella el MILP la toma como 0.
        wp_cap_df = extend_waypoint_capacity(wp_cap_df, [n["node_id"] for n in new_nodes],
                                             per_hour=wp_capacity_per_hour, levels=[DEFAULT_LEVEL])
        print(f"  Añadidos {len(new_nodes)} nodo(s) nuevo(s): {[n['node_id'] for n in new_nodes]}")

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
    parser.add_argument("--wp-capacity", type=int, default=5,
                         help="Capacidad horaria por waypoint (aeronaves/hora). Supuesto, no es dato real.")
    parser.add_argument("--prd-zones", default=None,
                         help="CSV opcional lat,lon,radius_km con zonas PRD (prohibidas/restringidas/peligrosas)")
    parser.add_argument("--m-new-nodes", type=int, default=2,
                         help="Nodos nuevos por iteración")
    parser.add_argument("--grid-resolution", type=float, default=GRID_RESOLUTION_DEG,
                         help="Resolución (grados) de la malla de waypoints virtuales; el artículo usa 0.2")
    parser.add_argument("--alpha-radius-km", type=float, default=None,
                         help="Radio circunscrito del alpha-shape (R=1/alpha). Por defecto se usa el "
                              "radio crítico: el menor que deja dentro a todos los nodos de la red")
    parser.add_argument("--no-alpha-shape", action="store_true",
                         help="No usar el contorno alpha-shape: malla sobre el rectángulo delimitador")
    parser.add_argument("--seed", type=int, default=0)
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

    # Capacidad de waypoint solo para waypoints de ruta. Los aeropuertos se
    # limitan con las capacidades Q de aeropuerto (Ec. 38-40), no como waypoints.
    wp_nodes = [n for n, d in G.nodes(data=True) if d.get("tipo") != "aeropuerto"]
    wp_cap_df = generate_waypoint_capacity(wp_nodes, per_hour=args.wp_capacity, levels=[DEFAULT_LEVEL])
    airports = list(set(flights_df["origin"]) | set(flights_df["destination"]))
    apt_cap_df = generate_airport_capacity(airports, dep_per_hour=50, arr_per_hour=50)
    prd_zones = load_prd_zones(args.prd_zones)

    G_final, history = run_iterative_optimization(
        G, flights_df, wp_cap_df, apt_cap_df,
        max_iterations=args.max_iterations, k=args.k,
        use_milp=not args.no_milp, grow_network=not args.fixed_network,
        wp_capacity_per_hour=args.wp_capacity, prd_zones=prd_zones, seed=args.seed,
        m_new_nodes=args.m_new_nodes, grid_resolution_deg=args.grid_resolution,
        use_alpha_shape=not args.no_alpha_shape, alpha_radius_km=args.alpha_radius_km,
    )

    print("\n=== Resumen final ===")
    hist_df = pd.DataFrame(history)
    print(hist_df.to_string(index=False))
    print(f"\nRed final: {G_final.number_of_nodes()} waypoints, {G_final.number_of_edges()} segmentos.")

    os.makedirs("outputs", exist_ok=True)
    hist_df.to_csv("outputs/convergence_history.csv", index=False)

    try:
        plot_network(G_final, "outputs/network_map_final.html", boundary=G_final.graph.get("boundary"))
    except ImportError:
        print("(folium no disponible en este entorno; omite el mapa)")


if __name__ == "__main__":
    main()

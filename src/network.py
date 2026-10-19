"""network.py -- construcción del grafo G=(V,E) y visualización."""

import math
import os
import pandas as pd
import networkx as nx

EARTH_RADIUS_KM = 6371.0


def great_circle_distance_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def build_network(waypoints_csv: str, segments_csv: str) -> nx.Graph:
    waypoints = pd.read_csv(waypoints_csv)
    segments = pd.read_csv(segments_csv)

    G = nx.Graph()
    for _, row in waypoints.iterrows():
        G.add_node(row["waypoint_id"], lat=row["lat"], lon=row["lon"],
                   tipo=row.get("tipo", "ordinario"))

    skipped = 0
    for _, row in segments.iterrows():
        u, v = row["from_id"], row["to_id"]
        if u not in G.nodes or v not in G.nodes:
            skipped += 1
            continue
        lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
        lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
        dist = great_circle_distance_km(lat1, lon1, lat2, lon2)
        G.add_edge(u, v, length_km=dist, airway_id=row.get("airway_id", None))

    print(f"Grafo construido: {G.number_of_nodes()} waypoints, "
          f"{G.number_of_edges()} segmentos.")
    if skipped:
        print(f"Aviso: se omitieron {skipped} segmentos por waypoints faltantes.")
    return G


def plot_network(G: nx.Graph, output_html: str = "outputs/network_map.html"):
    import folium

    # Crear automáticamente la carpeta de salida si no existe, para no
    # depender de que el usuario la haya creado antes a mano.
    output_dir = os.path.dirname(output_html)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    lats = [d["lat"] for _, d in G.nodes(data=True)]
    lons = [d["lon"] for _, d in G.nodes(data=True)]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    m = folium.Map(location=center, zoom_start=5, tiles="cartodbpositron")
    for u, v, data in G.edges(data=True):
        coords = [(G.nodes[u]["lat"], G.nodes[u]["lon"]), (G.nodes[v]["lat"], G.nodes[v]["lon"])]
        folium.PolyLine(coords, color="steelblue", weight=1.5, opacity=0.7).add_to(m)
    for node, data in G.nodes(data=True):
        folium.CircleMarker(location=(data["lat"], data["lon"]), radius=3, popup=str(node),
                            color="darkred" if data.get("tipo") in ("root", "terminal", "aeropuerto") else "gray",
                            fill=True).add_to(m)
    m.save(output_html)
    print(f"Mapa guardado en {output_html}")

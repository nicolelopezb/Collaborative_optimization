"""
network.py
-----------
Construye el grafo G = (V, E) de la red de rutas aéreas, tal como se define
en la Sección 4.1 del artículo (waypoints = nodos, segmentos = aristas).

Este módulo NO depende de cómo obtuviste los datos (extraídos del PDF a mano,
con explore_pdf.py, o escritos directamente en CSV). Solo espera dos archivos
limpios:

    data/processed/waypoints.csv
        columnas: waypoint_id, lat, lon, tipo
        tipo ∈ {root, terminal, ordinario}  (ver Sección 4.1 del artículo)

    data/processed/segments.csv
        columnas: from_id, to_id, airway_id (opcional)

Si todavía no tienes estos CSV, puedes crearlos a mano en Excel/LibreOffice
a partir de lo que veas en explore_pdf.py -- no es necesario automatizar
el parseo del PDF para poder avanzar con el resto del modelo.

Uso:
    from network import build_network, plot_network
    G = build_network("data/processed/waypoints.csv", "data/processed/segments.csv")
    plot_network(G, "outputs/network_map.html")
"""

import math
import pandas as pd
import networkx as nx


EARTH_RADIUS_KM = 6371.0


def great_circle_distance_km(lat1, lon1, lat2, lon2):
    """Distancia del gran círculo entre dos puntos (fórmula de Haversine).
    Se usa para la longitud de cada segmento (d_ij en el artículo, Ec. 10)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def build_network(waypoints_csv: str, segments_csv: str) -> nx.Graph:
    """Construye el grafo no dirigido G=(V,E) descrito en la Sección 4.1."""
    waypoints = pd.read_csv(waypoints_csv)
    segments = pd.read_csv(segments_csv)

    G = nx.Graph()

    # Añadir nodos (waypoints) con sus coordenadas
    for _, row in waypoints.iterrows():
        G.add_node(
            row["waypoint_id"],
            lat=row["lat"],
            lon=row["lon"],
            tipo=row.get("tipo", "ordinario"),
        )

    # Añadir aristas (segmentos) con su longitud calculada
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
        print(f"Aviso: se omitieron {skipped} segmentos por waypoints faltantes "
              f"(revisa que los IDs coincidan entre los dos CSV).")

    return G


def plot_network(G: nx.Graph, output_html: str = "outputs/network_map.html"):
    """Genera un mapa interactivo HTML para verificar visualmente la red."""
    import folium

    lats = [d["lat"] for _, d in G.nodes(data=True)]
    lons = [d["lon"] for _, d in G.nodes(data=True)]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]

    m = folium.Map(location=center, zoom_start=5, tiles="cartodbpositron")

    for u, v, data in G.edges(data=True):
        coords = [
            (G.nodes[u]["lat"], G.nodes[u]["lon"]),
            (G.nodes[v]["lat"], G.nodes[v]["lon"]),
        ]
        folium.PolyLine(coords, color="steelblue", weight=1.5, opacity=0.7).add_to(m)

    for node, data in G.nodes(data=True):
        folium.CircleMarker(
            location=(data["lat"], data["lon"]),
            radius=3,
            popup=str(node),
            color="darkred" if data.get("tipo") in ("root", "terminal") else "gray",
            fill=True,
        ).add_to(m)

    m.save(output_html)
    print(f"Mapa guardado en {output_html}")


if __name__ == "__main__":
    G = build_network("data/processed/waypoints.csv", "data/processed/segments.csv")
    plot_network(G)

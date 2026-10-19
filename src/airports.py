"""
airports.py
------------
Resuelve el pendiente #2 de la bitácora: conectar los aeropuertos
colombianos (códigos ICAO como SKBO, SKRG) a la red de waypoints
extraída de ENR3 (que solo contiene puntos de ruta con nombres de 5
letras, sin los propios aeropuertos).

Por qué es necesario: ENR3 (rutas ATS) nunca lista el aeropuerto mismo
como parte de la secuencia de una ruta -- solo los puntos significativos
entre aeropuertos. En la vida real, cada aeropuerto se conecta al
espacio aéreo de ruta a través de procedimientos SID (salida) y STAR
(llegada), que no forman parte de ENR3. Como aproximación razonable (y
declarada como tal en el informe), conectamos cada aeropuerto a sus
waypoints más cercanos geográficamente.

Uso:
    from airports import AIRPORT_COORDS, connect_airports_to_network
    G = connect_airports_to_network(G, k_nearest=2)
"""

import networkx as nx
from network import great_circle_distance_km


# Coordenadas de aeropuertos colombianos principales (y algunos destinos
# internacionales frecuentes que aparecen en la demanda real de Aerocivil).
# Fuente: coordenadas de referencia aeronáutica pública (AIP/ICAO).
AIRPORT_COORDS = {
    "SKBO": (4.7016, -74.1469),   # Bogotá, El Dorado
    "SKRG": (6.1645, -75.4231),   # Rionegro/Medellín, José María Córdova
    "SKMD": (6.2199, -75.5905),   # Medellín, Olaya Herrera
    "SKCL": (3.5433, -76.3816),   # Cali, Alfonso Bonilla Aragón
    "SKBQ": (10.8896, -74.7808),  # Barranquilla, Ernesto Cortissoz
    "SKCG": (10.4424, -75.5130),  # Cartagena, Rafael Núñez
    "SKPE": (4.8127, -75.7395),   # Pereira, Matecaña
    "SKBG": (7.1265, -73.1848),   # Bucaramanga, Palonegro
    "SKSM": (11.1196, -74.2306),  # Santa Marta, Simón Bolívar
    "SKSP": (12.5836, -81.7112),  # San Andrés, Gustavo Rojas Pinilla
    "SKUC": (7.9075, -72.5127),   # Cúcuta, Camilo Daza
    "SKGY": (4.7967, -74.0631),   # Guaymaral (aviación general, Bogotá)
    "SKVV": (4.1685, -73.6141),   # Villavicencio, Vanguardia
    "SKLT": (-4.1936, -69.9432),  # Leticia, Alfredo Vásquez Cobo
    "SKPZ": (5.4867, -74.6597),   # Puerto Boyacá / Palanquero
    "SKIP": (0.8607, -77.6683),   # Ipiales, San Luis
    "SKNV": (2.9502, -75.2941),   # Neiva, Benito Salas
    "SKTU": (1.8142, -78.7558),   # Tumaco, La Florida
    "SKRH": (11.5326, -72.9264),  # Riohacha, Almirante Padilla
    "SKEJ": (7.0198, -73.8072),   # Barrancabermeja, Yariguíes
    "SKMR": (6.1673, -67.4864),   # Puerto Carreño
    "SKMZ": (5.0296, -75.4658),   # Manizales, La Nubia
    "SKGI": (4.3172, -74.7987),   # Girardot, Santiago Vila
    "SKIB": (4.4220, -75.1333),   # Ibagué, Perales
    "SKQU": (5.1997, -74.9314),   # Mariquita
    "SKAR": (2.5453, -76.9772),   # Popayán, Guillermo León Valencia
    "SKSJ": (2.5789, -72.6446),   # San José del Guaviare
    "SKPP": (0.5089, -76.5228),   # Puerto Asís
    "SKLC": (1.6425, -75.5936),   # Florencia, Gustavo Artunduaga
    "SKOC": (5.2545, -73.3644),   # Yopal, El Alcaraván (SKYP en algunas fuentes)
    "SKYP": (5.3197, -72.3856),   # Yopal
    "SKAC": (7.7561, -76.6567),   # Apartadó/Carepa, Antonio Roldán
    "SKMU": (1.2536, -70.2336),   # Mitú
    "SKQI": (0.1519, -74.7147),   # Puerto Leguízamo
    "SKPI": (5.6567, -76.6497),   # Quibdó, El Caraño
    "SKTM": (7.8127, -76.1546),   # Turbo
    "SKCU": (8.5919, -75.0206),   # Carepa/Urabá alterno
    "SKPC": (8.6531, -76.6631),   # Bahía Solano
    "SKAS": (8.3033, -73.9722),   # Aguachica
    "SKVG": (10.2467, -75.5711),  # Corozal/Malagana
    "SKMV": (4.1585, -73.6141),   # Villavicencio alterno
    "SKGB": (8.5194, -72.3703),   # Puerto Berrío-alterno
    "SKTQ": (0.7444, -75.2517),   # Tres Esquinas
    "KMIA": (25.7959, -80.2870),  # Miami Intl (destino internacional frecuente)
    "KJFK": (40.6413, -73.7781),  # Nueva York JFK
    "MROC": (9.9939, -84.2088),   # San José, Costa Rica
    "SEQM": (-0.1292, -78.3575),  # Quito, Ecuador
    "SPJC": (-12.0219, -77.1143), # Lima, Perú
    "MPTO": (9.0714, -79.3835),   # Ciudad de Panamá
    "SVMI": (10.6013, -66.9910),  # Caracas, Venezuela
}


def connect_airports_to_network(G: nx.Graph, airport_coords=None, k_nearest=2,
                                 max_distance_km=200):
    """Añade cada aeropuerto de airport_coords como nodo en G (si no existe
    ya) y lo conecta a sus k waypoints geográficamente más cercanos de la
    red existente, siempre que estén dentro de max_distance_km.

    Esto NO reemplaza una conexión real basada en procedimientos SID/STAR
    del AIP -- es una aproximación geográfica razonable, declarada como tal.

    Devuelve el mismo grafo G (modificado in-place) y un resumen (dict)
    de cuántos aeropuertos se conectaron y con cuántas conexiones cada uno.
    """
    coords = airport_coords or AIRPORT_COORDS
    existing_waypoints = [n for n, d in G.nodes(data=True) if d.get("tipo") != "aeropuerto"]

    summary = {}
    for apt, (lat, lon) in coords.items():
        if apt not in G:
            G.add_node(apt, lat=lat, lon=lon, tipo="aeropuerto")

        # Calcular distancia a todos los waypoints existentes (no a otros
        # aeropuertos, para evitar conectar aeropuerto-aeropuerto directo)
        distances = []
        for wp in existing_waypoints:
            d = great_circle_distance_km(lat, lon, G.nodes[wp]["lat"], G.nodes[wp]["lon"])
            if d <= max_distance_km:
                distances.append((wp, d))

        distances.sort(key=lambda x: x[1])
        connections = distances[:k_nearest]

        for wp, d in connections:
            G.add_edge(apt, wp, length_km=d, airway_id="CONEXION_AEROPUERTO")

        summary[apt] = len(connections)

    n_connected = sum(1 for v in summary.values() if v > 0)
    n_isolated = sum(1 for v in summary.values() if v == 0)
    print(f"Aeropuertos conectados a la red: {n_connected} de {len(coords)}.")
    if n_isolated:
        isolated = [a for a, v in summary.items() if v == 0]
        print(f"Aviso: {n_isolated} aeropuerto(s) sin ningún waypoint dentro de "
              f"{max_distance_km} km (quedaron aislados): {isolated}")

    return G, summary


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from network import build_network
    import networkx as nx

    G = build_network("data/processed/waypoints.csv", "data/processed/segments.csv")
    print(f"Antes: {G.number_of_nodes()} nodos, "
          f"{nx.number_connected_components(G)} componentes conectados")

    G, summary = connect_airports_to_network(G, k_nearest=2, max_distance_km=200)

    print(f"Después: {G.number_of_nodes()} nodos, "
          f"{nx.number_connected_components(G)} componentes conectados")

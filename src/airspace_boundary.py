"""
airspace_boundary.py
---------------------
Contorno del espacio aéreo por alpha-shape y malla de waypoints virtuales
(Sec. 4.1.1 del artículo, primera regla de generación de waypoints):

    "To expand the network, the airspace boundary is first extracted using the
    alpha-shape method (Liao et al., 2021). Within its bounding box, a regular
    grid is partitioned at a latitude and longitude resolution of 0.2 degrees
    (approximately 22 km), where the center of each grid cell represents a
    virtual waypoint position."

Lo que el artículo NO especifica y aquí hubo que decidir (declararlo al
escribir la metodología):
  1. El valor de alpha. Aquí: el radio crítico, es decir, el MENOR radio de
     círculo circunscrito para el que el contorno deja dentro a todos los
     nodos de la red (ver critical_radius_km). Se puede fijar a mano.
  2. Qué puntos definen el espacio aéreo. Aquí: los nodos de la red con al
     menos un segmento (waypoints, aeropuertos conectados y nodos nuevos).
  3. Los huecos interiores del contorno se rellenan: el artículo habla de un
     contorno (boundary), no de regiones vacías internas.
  4. Que solo los centros de celda DENTRO del contorno son waypoints virtuales
     candidatos (el artículo da la caja delimitadora y el contorno, pero no
     dice expresamente que se descarte lo que queda fuera de él).
  5. Proyección plana equirrectangular (cos de la latitud media) para medir
     el radio circunscrito en km; el error es de pocos % en Colombia.

Método (alpha-shape clásico): triangulación de Delaunay de los puntos; se
conservan los triángulos cuyo radio de círculo circunscrito es <= R (R = 1/alpha);
el contorno es la unión de esos triángulos. Liao et al. (2021) proponen un
algoritmo que acelera la extracción con una partición en rejilla y paso
variable; da el mismo tipo de contorno, y con unos cientos de nodos la versión
clásica tarda milisegundos, por eso no se reimplementa esa aceleración.

Requiere: numpy, scipy (Delaunay) y shapely.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union

KM_PER_DEG = 111.195     # 2*pi*6371/360, coherente con network.great_circle_distance_km


@dataclass
class AirspaceBoundary:
    polygon: object                 # shapely Polygon/MultiPolygon, coordenadas (lon, lat)
    radius_km: float                # radio circunscrito usado (R = 1/alpha)
    auto: bool                      # True si R salió de la regla del radio crítico
    n_points: int                   # nodos usados para definir el contorno
    n_parts: int                    # polígonos disjuntos del contorno
    method: str = "alpha-shape"     # o "convex-hull" si no hubo triangulación posible
    uncovered: list = field(default_factory=list)   # índices de puntos que quedaron fuera


def _project(lonlat, lat0):
    kx = KM_PER_DEG * np.cos(np.radians(lat0))
    return np.column_stack([lonlat[:, 0] * kx, lonlat[:, 1] * KM_PER_DEG])


def _circumradius(a, b, c):
    la = np.linalg.norm(b - c, axis=1)
    lb = np.linalg.norm(a - c, axis=1)
    lc = np.linalg.norm(a - b, axis=1)
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return la * lb * lc / (4.0 * np.maximum(area, 1e-12))


def _triangulate(lonlat):
    """Delaunay en el plano (km). Devuelve (simplices, radios circunscritos)."""
    X = _project(lonlat, float(lonlat[:, 1].mean()))
    tri = Delaunay(X)
    S = tri.simplices
    return S, _circumradius(X[S[:, 0]], X[S[:, 1]], X[S[:, 2]])


def critical_radius_km(lonlat):
    """Menor R tal que TODO punto es vértice de algún triángulo con radio
    circunscrito <= R. Es exacto: para cada punto se toma su triángulo
    incidente de menor radio, y R es el máximo de esos mínimos."""
    S, R = _triangulate(lonlat)
    best = np.full(len(lonlat), np.inf)
    for k in range(3):
        np.minimum.at(best, S[:, k], R)
    finite = best[np.isfinite(best)]       # puntos duplicados no entran en la triangulación
    return float(finite.max()) if finite.size else 0.0


def _shape_from_radius(lonlat, radius_km):
    S, R = _triangulate(lonlat)
    keep = S[R <= radius_km * (1.0 + 1e-9)]
    if len(keep) == 0:
        raise ValueError("ningún triángulo cumple el radio pedido")
    union = unary_union([Polygon(lonlat[t]) for t in keep])
    parts = list(union.geoms) if isinstance(union, MultiPolygon) else [union]
    parts = [p for p in parts if not p.is_empty and p.area > 0]
    # solo el contorno exterior de cada parte (se rellenan los huecos interiores)
    outer = unary_union([Polygon(p.exterior) for p in parts])
    n_parts = len(outer.geoms) if isinstance(outer, MultiPolygon) else 1
    used = set(keep.ravel().tolist())
    return outer, n_parts, [i for i in range(len(lonlat)) if i not in used]


def alpha_shape(lonlat, radius_km=None):
    """Contorno alpha-shape de un conjunto de puntos (columnas lon, lat).

    radius_km=None -> radio crítico automático (ver critical_radius_km).
    Si los puntos no permiten triangular (menos de 4, o alineados), se usa la
    envolvente convexa y se deja constancia en `method`."""
    lonlat = np.asarray(lonlat, dtype=float)
    try:
        auto = radius_km is None
        r = critical_radius_km(lonlat) if auto else float(radius_km)
        polygon, n_parts, unc = _shape_from_radius(lonlat, r)
        return AirspaceBoundary(polygon, r, auto, len(lonlat), n_parts, "alpha-shape", unc)
    except Exception:                       # QhullError, ValueError, etc.
        hull = MultiPoint([tuple(p) for p in lonlat]).convex_hull
        return AirspaceBoundary(hull, float("nan"), radius_km is None, len(lonlat), 1, "convex-hull")


def boundary_points(G):
    """Nodos que definen el espacio aéreo colombiano: los de la red con al
    menos un segmento (un aeropuerto sin conexión no forma parte del espacio
    aéreo de rutas), dejando fuera los aeropuertos extranjeros (los que no
    empiezan por SK), que están en la red solo como extremos de rutas
    internacionales y no delimitan el espacio aéreo de Colombia.
    Devuelve (ids, array lon/lat)."""
    ids = [n for n in G.nodes
           if G.degree(n) > 0
           and (G.nodes[n].get("tipo") != "aeropuerto" or str(n).startswith("SK"))]
    arr = np.array([[G.nodes[n]["lon"], G.nodes[n]["lat"]] for n in ids], dtype=float)
    return ids, arr


def compute_airspace_boundary(G, radius_km=None):
    """Contorno del espacio aéreo de la red G (ver alpha_shape)."""
    _, pts = boundary_points(G)
    return alpha_shape(pts, radius_km)


def contains_lonlat(polygon, lons, lats):
    """Máscara booleana: qué puntos (lon, lat) están dentro del polígono."""
    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    try:
        import shapely
        return np.asarray(shapely.contains_xy(polygon, lons, lats))       # shapely >= 2.0
    except AttributeError:
        from shapely import vectorized                                    # shapely 1.8
        return np.asarray(vectorized.contains(polygon, lons, lats))


def grid_inside_boundary(polygon, resolution_deg=0.2):
    """Malla regular lat/lon de `resolution_deg` sobre la caja delimitadora del
    contorno; el centro de cada celda es un waypoint virtual y solo se
    conservan los que caen dentro del contorno. Devuelve lista (lat, lon)."""
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.arange(minx + resolution_deg / 2.0, maxx, resolution_deg)
    ys = np.arange(miny + resolution_deg / 2.0, maxy, resolution_deg)
    if xs.size == 0 or ys.size == 0:
        return []
    GX, GY = np.meshgrid(xs, ys)
    gx, gy = GX.ravel(), GY.ravel()
    mask = contains_lonlat(polygon, gx, gy)
    return [(round(float(y), 4), round(float(x), 4)) for x, y in zip(gx[mask], gy[mask])]


def polygon_rings_latlon(polygon):
    """Anillos exteriores como listas de (lat, lon), para dibujarlos en folium."""
    geoms = polygon.geoms if hasattr(polygon, "geoms") else [polygon]
    return [[(y, x) for x, y in g.exterior.coords] for g in geoms]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from network import build_network
    from airports import connect_airports_to_network

    G = build_network("data/processed/waypoints.csv", "data/processed/segments.csv")
    G, _ = connect_airports_to_network(G, k_nearest=2, max_distance_km=200)
    b = compute_airspace_boundary(G)
    grid = grid_inside_boundary(b.polygon, 0.2)
    minx, miny, maxx, maxy = b.polygon.bounds
    print(f"Método: {b.method} | R = {b.radius_km:.1f} km ({'automático' if b.auto else 'manual'}) | "
          f"{b.n_parts} polígono(s) | {b.n_points} nodos | sin cubrir: {len(b.uncovered)}")
    print(f"Caja: lat {miny:.2f}..{maxy:.2f}, lon {minx:.2f}..{maxx:.2f} | malla 0.2°: {len(grid)} puntos dentro")

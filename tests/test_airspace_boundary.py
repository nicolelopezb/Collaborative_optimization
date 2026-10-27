"""Pruebas del contorno alpha-shape y de la malla virtual (Sec. 4.1.1).

Ejecutar desde la raíz del repo:   python tests/test_airspace_boundary.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point

import airspace_boundary as ab
import apf_growth as apf
from network import great_circle_distance_km


def _forma_en_L(paso=0.25):
    """Puntos densos sobre una L: dos brazos de 1,5° x 0,5°. El hueco entre
    los brazos (esquina superior derecha) es lo que distingue el alpha-shape
    de la envolvente convexa."""
    pts = []
    for x in np.arange(0, 1.5 + 1e-9, paso):
        for y in np.arange(0, 0.5 + 1e-9, paso):
            pts.append((x, y))                        # brazo horizontal
    for x in np.arange(0, 0.5 + 1e-9, paso):
        for y in np.arange(0.5 + paso, 1.5 + 1e-9, paso):
            pts.append((x, y))                        # brazo vertical
    return np.array(pts)


def test_alpha_shape_excluye_la_concavidad_que_la_envolvente_convexa_incluye():
    pts = _forma_en_L()
    centro_del_hueco = (0.8, 0.8)      # dentro de la envolvente convexa (x + y < 2) pero en el hueco de la L
    hull = MultiPoint([tuple(p) for p in pts]).convex_hull
    assert hull.contains(Point(*centro_del_hueco))        # la envolvente convexa lo incluye
    b = ab.alpha_shape(pts, radius_km=30)                 # 0,25 deg ~ 28 km entre puntos
    assert b.method == "alpha-shape"
    assert not b.polygon.contains(Point(*centro_del_hueco))
    assert b.polygon.contains(Point(0.25, 0.25))          # el interior de la L sí


def test_radio_critico_deja_dentro_todos_los_puntos():
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(-77, -72, 150), rng.uniform(2, 8, 150)])
    b = ab.alpha_shape(pts)
    assert b.auto and b.uncovered == []
    poligono = b.polygon.buffer(1e-9)
    assert all(poligono.covers(Point(*p)) for p in pts)


def test_radio_critico_es_el_minimo():
    # con un radio un poco menor al crítico, algún punto queda sin cubrir
    rng = np.random.default_rng(5)
    pts = np.column_stack([rng.uniform(-77, -72, 80), rng.uniform(2, 8, 80)])
    r = ab.critical_radius_km(pts)
    assert ab.alpha_shape(pts, radius_km=r * 1.0001).uncovered == []
    assert len(ab.alpha_shape(pts, radius_km=r * 0.95).uncovered) > 0


def test_malla_de_0_2_grados_centros_de_celda_dentro_del_contorno():
    pts = _forma_en_L()
    b = ab.alpha_shape(pts, radius_km=30)
    grid = ab.grid_inside_boundary(b.polygon, 0.2)
    assert len(grid) > 0
    minx, miny, maxx, maxy = b.polygon.bounds
    lats = np.array([g[0] for g in grid])
    lons = np.array([g[1] for g in grid])
    assert ab.contains_lonlat(b.polygon, lons, lats).all()           # todos dentro
    # centros de celda: el primero está a media celda de la esquina de la caja
    assert abs(lons.min() - (minx + 0.1)) < 0.2 and abs(lats.min() - (miny + 0.1)) < 0.2
    # separación entre puntos vecinos: múltiplo de 0,2
    pasos = np.diff(np.unique(np.round(lons, 4)))
    assert np.allclose(pasos / 0.2, np.round(pasos / 0.2), atol=1e-3)
    # y no hay candidatos en el hueco de la L
    assert not any(0.7 < lon < 0.95 and 0.7 < lat < 0.95 for lat, lon in grid)


def test_degenerado_no_revienta():
    b = ab.alpha_shape(np.array([[-74.0, 4.0], [-73.0, 4.0], [-72.0, 4.0]]))   # alineados
    assert b.method == "convex-hull"
    assert ab.alpha_shape(np.array([[-74.0, 4.0], [-73.0, 5.0]])).method == "convex-hull"


def test_generate_virtual_grid_con_contorno_es_subconjunto_de_la_caja():
    G = nx.Graph()
    for i, (lon, lat) in enumerate(_forma_en_L() * 1.0):
        G.add_node(f"N{i}", lat=float(lat) + 4, lon=float(lon) - 76)
    nodos = list(G.nodes)
    for a, b in zip(nodos[:-1], nodos[1:]):
        G.add_edge(a, b)
    b = ab.compute_airspace_boundary(G)
    con = apf.generate_virtual_grid(G, 0.2, boundary=b.polygon)
    sin = apf.generate_virtual_grid(G, 0.2, margin_deg=0.0)
    assert 0 < len(con) < len(sin)


def test_aeropuertos_extranjeros_y_aislados_no_definen_el_contorno():
    G = nx.Graph()
    for i, (lat, lon) in enumerate([(4, -75), (4, -74), (5, -75), (5, -74), (4.5, -74.5)]):
        G.add_node(f"W{i}", lat=lat, lon=lon, tipo="ordinario")
    G.add_node("SKBO", lat=4.7, lon=-74.1, tipo="aeropuerto")
    G.add_node("KJFK", lat=40.6, lon=-73.7, tipo="aeropuerto")      # extranjero conectado
    G.add_node("SKXX", lat=10.0, lon=-70.0, tipo="aeropuerto")      # aislado
    G.add_edges_from([("W0", "W1"), ("W1", "W3"), ("W3", "W2"), ("W2", "W0"), ("W4", "W0"),
                      ("SKBO", "W4"), ("KJFK", "W3")])
    ids, _ = ab.boundary_points(G)
    assert "SKBO" in ids and "KJFK" not in ids and "SKXX" not in ids


def _top_candidates_original(G, virtual_points, weights, r, m):
    """Copia del bucle original, para comprobar que la versión vectorizada
    da el mismo resultado."""
    scored = []
    for lat, lon in virtual_points:
        total = 0.0
        for node, data in G.nodes(data=True):
            if great_circle_distance_km(lat, lon, data["lat"], data["lon"]) < r:
                total += weights.get(node, 0.0)
        if total > 0:
            scored.append((lat, lon, total))
    return sorted(scored, key=lambda t: -t[2])[:m]


def test_select_top_candidates_vectorizado_igual_al_original():
    rng = np.random.default_rng(11)
    G = nx.Graph()
    for i in range(60):
        G.add_node(f"N{i}", lat=float(rng.uniform(2, 9)), lon=float(rng.uniform(-77, -70)))
    w = pd.Series({f"N{i}": float(rng.choice([0, 0, 0.2, 0.5, 1.0])) for i in range(60)})
    grid = [(round(la, 4), round(lo, 4)) for la in np.arange(2, 9, 0.35) for lo in np.arange(-77, -70, 0.35)]
    nuevo = apf.select_top_candidates(G, grid, w, attraction_radius_km=150, m=10)
    viejo = _top_candidates_original(G, grid, w, 150, 10)
    assert np.allclose(nuevo["W"].values, [t[2] for t in viejo])      # mismos pesos, mismo orden
    assert nuevo.shape[0] == 10


def test_select_top_candidates_sin_pesos_devuelve_vacio():
    G = nx.Graph()
    G.add_node("A", lat=4.0, lon=-74.0)
    out = apf.select_top_candidates(G, [(4.0, -74.0)], pd.Series({"A": 0.0}), m=3)
    assert out.empty and list(out.columns) == ["lat", "lon", "W"]


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for nombre, f in pruebas:
        f()
        print("ok  ", nombre)
    print(f"\n{len(pruebas)} pruebas pasaron")

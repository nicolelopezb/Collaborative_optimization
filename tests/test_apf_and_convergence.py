"""Pruebas de la APF (Ec. 13-19) y de los criterios de convergencia I y II.

Ejecutar desde la raíz del repo:   python tests/test_apf_and_convergence.py
(o con pytest, si está instalado: pytest tests/)
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import networkx as nx
import pandas as pd

import apf_growth as apf
from main import check_convergence
from network import great_circle_distance_km as gc


def _graph():
    G = nx.Graph()
    G.add_node("A", lat=5.0, lon=-75.0)
    G.add_node("B", lat=5.0, lon=-73.92)       # ~120 km al este de A
    G.add_node("C", lat=5.9, lon=-74.5)
    G.add_edge("A", "B")
    G.add_edge("A", "C")
    return G


def test_sin_flujo_ni_vecinos_cercanos_no_hay_fuerza():
    fx, fy = apf.total_force(_graph(), 5.5, -74.5, ["A", "B"], {})
    assert (fx, fy) == (0.0, 0.0)


def test_repulsion_por_longitud_aleja_del_vecino():
    # punto ~10 km al este de A (< Lth = 17.3 km): debe empujar hacia el este
    fx, _ = apf.total_force(_graph(), 5.0, -74.91, ["A"], {})
    assert fx > 0


def test_repulsion_prd_aleja_de_la_zona():
    zona = {"lat": 5.0, "lon": -74.4, "radius_km": 5}   # al este del punto
    fx, _ = apf.total_force(_graph(), 5.0, -74.5, [], {}, prd_zones=[zona])
    assert fx < 0


def test_fuerza_angular_ensancha_el_angulo():
    G = _graph()
    vo = (5.45, -74.80)                                  # casi alineado con A->C

    def ang(p):
        return apf._angle_deg(apf._to_enu(5.0, -75.0, *p), apf._to_enu(5.0, -75.0, 5.9, -74.5))
    fx, fy = apf.total_force(G, *vo, ["A"], {})
    moved = apf._destination(*vo, math.degrees(math.atan2(fx, fy)), 2.0)
    assert ang(moved) > ang(vo)


def test_apf_acerca_a_la_ruta_optima_y_respeta_restricciones_duras():
    G, conns, flows = _graph(), ["A", "B"], {("A", "B"): 200}
    lat0, lon0 = 5.27, -74.46                            # ~30 km al norte de la recta A-B
    res = apf.optimize_virtual_point(G, lat0, lon0, conns, flows,
                                     attraction_radius_km=150, max_move_km=60, seed=1)
    assert res["feasible"]
    assert res["cost"] < apf.pair_route_cost(G, lat0, lon0, conns, flows)
    # sin violaciones de Lmin, radio r, ni ángulo mínimo de 15 grados
    assert apf.validate_new_waypoint(G, res["lat"], res["lon"], conns) == []
    for a in conns:
        assert gc(res["lat"], res["lon"], *[G.nodes[a][k] for k in ("lat", "lon")]) > 7.3


def test_apf_nunca_entra_en_zona_prd():
    G, conns, flows = _graph(), ["A", "B"], {("A", "B"): 500}
    zona = {"lat": 5.04, "lon": -74.46, "radius_km": 4}  # sobre el camino hacia la recta
    res = apf.optimize_virtual_point(G, 5.27, -74.46, conns, flows, attraction_radius_km=150,
                                     max_move_km=60, prd_zones=[zona], seed=2)
    assert res["feasible"]
    assert apf.validate_new_waypoint(G, res["lat"], res["lon"], conns, prd_zones=[zona]) == []


def test_criterio_II_exceso_horario_maximo_5_por_ciento():
    cap = pd.DataFrame({"waypoint_id": ["W"] * 2, "level": ["L"] * 2,
                        "time_slot": [0, 1], "capacity": [100, 100]})
    ok = pd.DataFrame({"waypoint_id": ["W"], "level": ["L"], "time_slot": [0], "flights": [105]})
    mal = pd.DataFrame({"waypoint_id": ["W"], "level": ["L"], "time_slot": [0], "flights": [106]})
    r_ok, n_ok = apf.compute_max_excess_ratio(ok, cap)
    r_mal, n_mal = apf.compute_max_excess_ratio(mal, cap)
    assert math.isclose(r_ok, 0.05) and n_ok == 0        # exactamente 5 %: cumple
    assert math.isclose(r_mal, 0.06) and n_mal == 1


def _h(excess, km, ratio):
    return {"total_excess": excess, "total_extra_km": km, "max_excess_ratio": ratio}


def test_convergencia_exige_criterios_I_y_II_a_la_vez():
    estable = [_h(1000, 5000, 0.5), _h(1000, 5000, 0.5), _h(1000, 5000, 0.5)]
    assert check_convergence(estable) == (False, True, False)          # I sí, II no
    estable[-1]["max_excess_ratio"] = 0.04
    assert check_convergence(estable) == (True, True, True)            # ambos
    cambiando = [_h(1000, 5000, 0.01), _h(900, 5000, 0.01), _h(800, 5000, 0.01)]
    assert check_convergence(cambiando) == (False, False, True)        # II sí, I no


def test_criterio_I_umbrales_3_por_ciento_y_0_5_por_ciento():
    # exceso -2 % y distancia -0,4 %: dentro de los umbrales -> Criterio I cumplido
    assert check_convergence([_h(1000, 5000, 0.01), _h(980, 4980, 0.01)])[1] is True
    # exceso -4 % (> 3 %): no cumple
    assert check_convergence([_h(1000, 5000, 0.01), _h(960, 4990, 0.01)])[1] is False
    # distancia -0,6 % (> 0,5 %): no cumple
    assert check_convergence([_h(1000, 5000, 0.01), _h(990, 4970, 0.01)])[1] is False
    # con una sola iteración no hay variación que medir
    assert check_convergence([_h(1000, 5000, 0.01)])[1] is False


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for nombre, f in pruebas:
        f()
        print("ok  ", nombre)
    print(f"\n{len(pruebas)} pruebas pasaron")

import pytest
import networkx as nx
import pandas as pd
from src.planning.scenario_simulator import simulate_departure_impact


@pytest.fixture
def simple_graph():
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B", {"weight": 10}),
        ("B", "C", {"weight": 8}),
        ("C", "D", {"weight": 5}),
        ("A", "C", {"weight": 3}),
    ])
    return G


def test_removing_central_node_increases_components(simple_graph):
    G      = simple_graph
    df_emp = pd.DataFrame({
        "employee_id":  ["A", "B", "C", "D"],
        "department":   ["Eng"] * 4,
        "tenure_years": [2, 3, 1, 4],
        "full_name":    ["Ana", "Budi", "Citra", "Dani"]
    })
    df_cent = pd.DataFrame({
        "employee_id":    ["A", "B", "C", "D"],
        "influence_score": [0.9, 0.5, 0.3, 0.1]
    })
    result = simulate_departure_impact(G, df_emp, ["B"], df_cent)
    assert result["lost_direct_connections"] >= 2


def test_empty_departure_list_no_impact(simple_graph):
    G      = simple_graph
    df_emp = pd.DataFrame({
        "employee_id":  ["A", "B", "C", "D"],
        "department":   ["Eng"] * 4,
        "tenure_years": [2, 3, 1, 4],
        "full_name":    ["Ana", "Budi", "Citra", "Dani"]
    })
    df_cent = pd.DataFrame({
        "employee_id":    ["A", "B", "C", "D"],
        "influence_score": [0.9, 0.5, 0.3, 0.1]
    })
    result = simulate_departure_impact(G, df_emp, [], df_cent)
    assert result["lost_direct_connections"] == 0
    assert result["total_influence_lost"] == 0.0
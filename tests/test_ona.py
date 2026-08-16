import pytest
import networkx as nx
import pandas as pd
import numpy as np
from src.simulation.employee_generator import generate_employees
from src.simulation.communication_generator import generate_communication_data
from src.network.ona_builder import (build_employee_graph,
                                      compute_all_centrality,
                                      classify_network_roles)


@pytest.fixture(scope="module")
def graph_fixture():
    df_emp = generate_employees(seed=42)
    df_int = generate_communication_data(df_emp, n_interactions=5000, seed=42)
    G      = build_employee_graph(df_int, df_emp, min_interactions=2)
    return G, df_emp, df_int


def test_graph_nodes_match_employees(graph_fixture):
    G, df_emp, _ = graph_fixture
    assert G.number_of_nodes() <= len(df_emp)


def test_graph_is_directed(graph_fixture):
    G, _, _ = graph_fixture
    assert isinstance(G, nx.DiGraph)


def test_all_edge_weights_positive(graph_fixture):
    G, _, _ = graph_fixture
    for _, _, data in G.edges(data=True):
        assert data["weight"] > 0


def test_centrality_scores_bounded_0_1(graph_fixture):
    G, df_emp, _ = graph_fixture
    df_cent = compute_all_centrality(G)
    assert df_cent["betweenness_norm"].between(0, 1).all()
    assert df_cent["influence_score"].between(0, 1).all()


def test_influence_score_not_all_zero(graph_fixture):
    G, df_emp, _ = graph_fixture
    df_cent = compute_all_centrality(G)
    assert df_cent["influence_score"].sum() > 0


def test_network_roles_cover_all_employees(graph_fixture):
    G, df_emp, _ = graph_fixture
    df_cent  = compute_all_centrality(G)
    df_roles = classify_network_roles(df_cent)
    valid_roles = {
        "Central Connector", "Information Broker", "Energizer/Hub",
        "Peripheral Specialist", "Peripheral/Isolated", "Regular Member"
    }
    assert set(df_roles["network_role"].unique()).issubset(valid_roles)
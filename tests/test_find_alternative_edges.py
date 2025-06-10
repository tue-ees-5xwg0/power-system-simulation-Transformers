import unittest

import pytest

from power_system_simulation.assignment1 import EdgeAlreadyDisabledError, GraphProcessor, IDNotFoundError

vertex_ids = [0, 2, 4, 6, 10]
edge_ids = [1, 3, 5, 7, 8, 9]
edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]
edge_enabled = [True, True, True, False, False, True]
source_vertex_id = 10


def test_alternative_edges():
    gp = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    assert gp.find_alternative_edges(3) == [7, 8]
    assert gp.find_alternative_edges(1) == [7]
    assert gp.find_alternative_edges(5) == [8]
    assert gp.find_alternative_edges(9) == []


def test_EdgeAlreadyDisabledError():
    with pytest.raises(EdgeAlreadyDisabledError):
        graph = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
        graph.find_alternative_edges(7)


def test_IDNotFound():
    with pytest.raises(IDNotFoundError):
        graph = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
        graph.find_alternative_edges(11)


def test_emtpy_list():
    graph = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    assert graph.find_alternative_edges(9) == []

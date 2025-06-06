import unittest

import pytest

from power_system_simulation.assignment1 import GraphProcessor, IDNotFoundError, IDNotUniqueError, GraphNotFullyConnectedError
vertex_ids = [0, 2, 4, 6, 10]
edge_ids = [1, 3, 5, 7, 8, 9]
edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]
edge_enabled = [True, True, True, False, False, True]
source_vertex_id = 10

graph = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)

def test_downstream_vertices():
    assert graph.find_downstream_vertices(1) == [0, 4, 6]
    assert graph.find_downstream_vertices(3) == [4]
    assert graph.find_downstream_vertices(5) == [6]
    assert graph.find_downstream_vertices(7) == []
    assert graph.find_downstream_vertices(8) == []
    assert graph.find_downstream_vertices(9) == [0, 2, 4, 6]

def test_downstream_vertices_1():
    vertex_ids = [0,2,4]
    edge_ids = [1 , 3]
    edge_vertex_id_pairs = [(0,2), (2,4)]
    edge_enabled = [True, True]
    source_vertex_id = 0
    graph = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    assert graph.find_downstream_vertices(1) == [2, 4]
    assert graph.find_downstream_vertices(3) == [4]

def test_IDNotFound():
    with pytest.raises(IDNotFoundError):
        graph = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
        graph.find_downstream_vertices(11)
        # graph.find_downstream_vertices(12)

def test_duplicate_vertex_and_edge_ids():
    vertex_ids = [0, 1, 2]
    edge_ids = [1]  # 1 is duplicated
    edge_vertex_id_pairs = [(0, 2)]
    edge_enabled = [True]
    source_vertex_id = 0

    with pytest.raises(IDNotUniqueError):
        GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)

def test_graph_not_fully_connected():
    vertex_ids = [0, 2, 4]
    edge_ids = [1]
    edge_vertex_id_pairs = [(0, 2)]
    edge_enabled = [True]
    source_vertex_id = 0

    # vertex 2 is isolated
    with pytest.raises(GraphNotFullyConnectedError):
        GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
#     # Create a sample graph
#     vertex_ids = [0,2,4]
#     edge_ids = [1 , 3]
#     #edge_vertex_id_pairs = [(0, 1), (2, 1), (2, 3), (4, 3)]
#     edge_vertex_id_pairs = [(0,2), (2,4)]
#     edge_enabled = [True, True]
#     source_vertex_id = 0

#     # Initialize the GraphProcessor
#     graph_processor = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
#     assert graph_processor.find_downstream_vertices(1) == [2, 4]
#     assert graph_processor.find_downstream_vertices(3) == [4]
    
# def test_find_downstream_vertices_with_disabled_edge():
#     vertex_ids = [0, 2, 4]
#     edge_ids = [1, 3]
#     edge_vertex_id_pairs = [(0, 2), (2, 4)]
#     edge_enabled = [False, True]  # Edge 1 is disabled
#     source_vertex_id = 0
#     graph_processor = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)

#     # Edge 1 is disabled, so it should return an empty list
#     assert graph_processor.find_downstream_vertices(1) == []

# def test_find_downstream_vertices_invalid_edge_id():
#     vertex_ids = [0, 2, 4]
#     edge_ids = [1, 3]
#     edge_vertex_id_pairs = [(0, 2), (2, 4)]
#     edge_enabled = [True, True]
#     source_vertex_id = 0
#     graph_processor = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)

#     # Edge ID 99 does not exist
#     with pytest.raises(IDNotFoundError):
#         graph_processor.find_downstream_vertices(99)
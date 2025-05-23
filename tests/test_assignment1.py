from power_system_simulation.assignment1 import GraphProcessor, IDNotFoundError, EdgeAlreadyDisabledError

def test_find_alternative_edges():
    vertex_ids = [0,2,4,6,10]
    edge_ids = [1,3,5,7,8,9]
    edge_vertex_id_pairs = [(0,2), (0,4), (0,6), (2,4),(4,6),(2,10)]
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 0
    gp = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)

    assert gp.find_alternative_edges(1) == [7]
    assert gp.find_alternative_edges(3) == [7,8]
    assert gp.find_alternative_edges(5) == [8]
    assert gp.find_alternative_edges(9) == []

def test_find_downstream_vertices():
    # Create a sample graph
    vertex_ids = [0,2,4]
    edge_ids = [1 , 3]
    #edge_vertex_id_pairs = [(0, 1), (2, 1), (2, 3), (4, 3)]
    edge_vertex_id_pairs = [(0,2), (2,4)]
    edge_enabled = [True, True]
    source_vertex_id = 0

    # Initialize the GraphProcessor
    graph_processor = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    assert graph_processor.find_downstream_vertices(1) == [2, 4]
    assert graph_processor.find_downstream_vertices(3) == [4]
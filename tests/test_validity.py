import pytest

from power_system_simulation.assignment1 import (
    GraphCycleError,
    GraphNotFullyConnectedError,
    GraphProcessor,
    IDNotFoundError,
    IDNotUniqueError,
    InputLengthDoesNotMatchError,
)

# Common minimal valid input
VALID_VERTEX_IDS = [1, 2, 3]
VALID_EDGE_IDS = [10, 11]
VALID_EDGE_PAIRS = [(1, 2), (2, 3)]
VALID_EDGE_ENABLED = [True, True]
VALID_SOURCE_ID = 1


def test_successful_initialization():
    gp = GraphProcessor(
        vertex_ids=VALID_VERTEX_IDS,
        edge_ids=VALID_EDGE_IDS,
        edge_vertex_id_pairs=VALID_EDGE_PAIRS,
        edge_enabled=VALID_EDGE_ENABLED,
        source_vertex_id=VALID_SOURCE_ID,
    )
    assert isinstance(gp, GraphProcessor)


def test_duplicate_vertex_and_edge_ids_raises():
    vertex_ids = [1, 2, 10]  # 10 overlaps with edge_ids
    with pytest.raises(IDNotUniqueError):
        GraphProcessor(vertex_ids, VALID_EDGE_IDS, VALID_EDGE_PAIRS, VALID_EDGE_ENABLED, VALID_SOURCE_ID)


def test_mismatched_edge_list_length_raises():
    invalid_edge_pairs = [(1, 2)]  # Length 1 instead of 2
    with pytest.raises(InputLengthDoesNotMatchError):
        GraphProcessor(VALID_VERTEX_IDS, VALID_EDGE_IDS, invalid_edge_pairs, VALID_EDGE_ENABLED, VALID_SOURCE_ID)


def test_invalid_vertex_in_edge_pair_raises():
    invalid_edge_pairs = [(1, 99), (2, 3)]  # 99 is not a valid vertex
    with pytest.raises(IDNotFoundError):
        GraphProcessor(VALID_VERTEX_IDS, VALID_EDGE_IDS, invalid_edge_pairs, VALID_EDGE_ENABLED, VALID_SOURCE_ID)


def test_mismatched_enabled_flags_length_raises():
    invalid_edge_enabled = [True]  # Only one flag for two edges
    with pytest.raises(InputLengthDoesNotMatchError):
        GraphProcessor(VALID_VERTEX_IDS, VALID_EDGE_IDS, VALID_EDGE_PAIRS, invalid_edge_enabled, VALID_SOURCE_ID)


def test_invalid_source_vertex_raises():
    with pytest.raises(IDNotFoundError):
        GraphProcessor(VALID_VERTEX_IDS, VALID_EDGE_IDS, VALID_EDGE_PAIRS, VALID_EDGE_ENABLED, source_vertex_id=99)


def test_graph_not_connected_raises():
    vertex_ids = [1, 2, 3]
    edge_ids = [10]  # Only one edge connecting 1 and 2
    edge_pairs = [(1, 2)]
    edge_enabled = [True]
    with pytest.raises(GraphNotFullyConnectedError):
        GraphProcessor(vertex_ids, edge_ids, edge_pairs, edge_enabled, source_vertex_id=1)


def test_graph_with_cycle_raises():
    # Triangle creates a cycle
    vertex_ids = [1, 2, 3]
    edge_ids = [10, 11, 12]
    edge_pairs = [(1, 2), (2, 3), (3, 1)]
    edge_enabled = [True, True, True]
    with pytest.raises(GraphCycleError):
        GraphProcessor(vertex_ids, edge_ids, edge_pairs, edge_enabled, source_vertex_id=1)

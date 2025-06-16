# pylint: disable=redefined-outer-name, import-error, no-name-in-module,invalid-name, function-redefined

"""Test for graph_processor"""
# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────


import matplotlib
from matplotlib.figure import Figure
import pytest

from power_system_simulation.graph_processor import (
    EdgeAlreadyDisabledError,
    GraphCycleError,
    GraphNotFullyConnectedError,
    GraphProcessor,
    IDNotFoundError,
    IDNotUniqueError,
    InputLengthDoesNotMatchError,
)
# ─────────────────────────────────────────────────────────────
# Create the object that is used for all tests
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def graph():
    """
    Pytest fixture for creating a GraphProcessor instance.

    Constructs a graph with specified vertex and edge configurations for use in tests.

    Returns:
        GraphProcessor: Initialized graph object for testing.
    """
    # Define graph components
    vertex_ids = [0, 2, 4, 6, 10]
    edge_ids = [1, 3, 5, 7, 8, 9]
    edge_vertex_id_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]
    edge_enabled = [True, True, True, False, False, True]
    source_vertex_id = 10

    # Create and return GraphProcessor instance
    return GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)


# ─────────────────────────────────────────────────────────────
# TEST: Graph initialization
# ─────────────────────────────────────────────────────────────


def test_successful_initialization(graph):
    """
    Verify successful graph initialization from the graph fixture.
    """
    # Ensure that the fixture creates a valid GraphProcessor
    assert isinstance(graph, GraphProcessor)


def test_duplicate_vertex_and_edge_ids_raises(graph):
    """
    Ensure that overlapping IDs between vertex and edge lists raise IDNotUniqueError.
    """
    # Modify vertex_ids to conflict with edge IDs
    vertex_ids = [0, 2, 5, 6, 10]
    graph.vertex_ids = vertex_ids

    # Expect error due to ID conflict
    with pytest.raises(IDNotUniqueError):
        GraphProcessor(
            vertex_ids, graph.edge_ids, graph.edge_vertex_id_pairs, graph.edge_enabled, graph.source_vertex_id
        )


def test_mismatched_edge_list_length_raises(graph):
    """
    Validate that a malformed edge list raises InputLengthDoesNotMatchError.
    """
    # Provide fewer edge pairs than edge IDs
    invalid_edge_pairs = [(1, 2)]

    # Expect error due to length mismatch
    with pytest.raises(InputLengthDoesNotMatchError):
        GraphProcessor(graph.vertex_ids, graph.edge_ids, invalid_edge_pairs, graph.edge_enabled, graph.source_vertex_id)


def test_invalid_vertex_in_edge_pair_raises(graph):
    """
    Ensure that unknown vertex IDs in edges raise IDNotFoundError.
    """
    # Include invalid vertex 1000 in edge pairs
    invalid_edge_pairs = [(0, 1000), (0, 4), (0, 6), (2, 4), (4, 6), (2, 10)]

    # Expect error due to invalid vertex
    with pytest.raises(IDNotFoundError):
        GraphProcessor(graph.vertex_ids, graph.edge_ids, invalid_edge_pairs, graph.edge_enabled, graph.source_vertex_id)


def test_mismatched_enabled_flags_length_raises(graph):
    """
    Ensure that the length mismatch in enabled edge flags raises InputLengthDoesNotMatchError.
    """
    # Only one flag for several edges
    invalid_edge_enabled = [True]

    # Expect error due to mismatch
    with pytest.raises(InputLengthDoesNotMatchError):
        GraphProcessor(
            graph.vertex_ids, graph.edge_ids, graph.edge_vertex_id_pairs, invalid_edge_enabled, graph.source_vertex_id
        )


def test_invalid_source_vertex_raises(graph):
    """
    Ensure that an unknown source vertex raises IDNotFoundError.
    """
    # Use invalid source vertex ID
    with pytest.raises(IDNotFoundError):
        GraphProcessor(graph.vertex_ids, graph.edge_ids, graph.edge_vertex_id_pairs, graph.edge_enabled, 99)


def test_duplicate_vertex_and_edge_ids():
    """
    Ensure that GraphProcessor raises IDNotUniqueError when vertex and edge IDs overlap.
    """
    # Overlapping ID '1' in vertex_ids and edge_ids
    vertex_ids = [0, 1, 2]
    edge_ids = [1]
    edge_vertex_id_pairs = [(0, 2)]
    edge_enabled = [True]
    source_vertex_id = 0

    # Expect error due to ID conflict
    with pytest.raises(IDNotUniqueError):
        GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)


def test_graph_not_fully_connected():
    """
    Ensure that GraphProcessor raises GraphNotFullyConnectedError if not all vertices are reachable.
    """
    # Vertex 4 is disconnected
    vertex_ids = [0, 2, 4]
    edge_ids = [1]
    edge_vertex_id_pairs = [(0, 2)]
    edge_enabled = [True]
    source_vertex_id = 0

    # Expect error due to disconnected graph
    with pytest.raises(GraphNotFullyConnectedError):
        GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)


def test_graph_with_cycle_raises():
    """
    Ensure that cycles in the graph raise GraphCycleError.
    """
    # Create a triangle (cycle)
    vertex_ids = [1, 2, 3]
    edge_ids = [10, 11, 12]
    edge_pairs = [(1, 2), (2, 3), (3, 1)]
    edge_enabled = [True, True, True]

    # Expect cycle error
    with pytest.raises(GraphCycleError):
        GraphProcessor(vertex_ids, edge_ids, edge_pairs, edge_enabled, source_vertex_id=1)


# ─────────────────────────────────────────────────────────────
# TEST: find_downstream_vertices
# ─────────────────────────────────────────────────────────────


def test_downstream_vertices(graph):
    """
    Validate downstream vertex resolution for various enabled edges.
    """
    # Test multiple edge IDs for correct downstream vertex sets
    assert graph.find_downstream_vertices(1) == [0, 4, 6]
    assert graph.find_downstream_vertices(3) == [4]
    assert graph.find_downstream_vertices(5) == [6]
    assert graph.find_downstream_vertices(7) == []
    assert graph.find_downstream_vertices(8) == []
    assert graph.find_downstream_vertices(9) == [0, 2, 4, 6]


def test_downstream_vertices_1():
    """
    Custom downstream vertex test for a simple linear graph.
    """
    # Create a simple linear graph
    vertex_ids = [0, 2, 4]
    edge_ids = [1, 3]
    edge_vertex_id_pairs = [(0, 2), (2, 4)]
    edge_enabled = [True, True]
    source_vertex_id = 0

    # Create graph and test downstream discovery
    graph = GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_vertex_id)
    assert graph.find_downstream_vertices(1) == [2, 4]
    assert graph.find_downstream_vertices(3) == [4]


def test_IDNotFound(graph):
    """
    Ensure that querying an unknown edge raises IDNotFoundError.
    """
    # Edge ID 11 is not in the graph
    with pytest.raises(IDNotFoundError):
        graph.find_downstream_vertices(11)


# ─────────────────────────────────────────────────────────────
# TEST: find_alternative_edges
# ─────────────────────────────────────────────────────────────


def test_alternative_edges(graph):
    """
    Validate correct resolution of alternative edges.
    """
    # Check correct alternatives for disabled edges
    assert graph.find_alternative_edges(3) == [7, 8]
    assert graph.find_alternative_edges(1) == [7]
    assert graph.find_alternative_edges(5) == [8]
    assert graph.find_alternative_edges(9) == []


def test_EdgeAlreadyDisabledError(graph):
    """
    Ensure that attempting to find alternatives for an already-disabled edge raises an error.
    """
    # Edge 7 is already disabled
    with pytest.raises(EdgeAlreadyDisabledError):
        graph.find_alternative_edges(7)


def test_IDNotFound(graph):
    """
    Ensure that an unknown edge ID raises IDNotFoundError in alternative edge search.
    """
    # Edge ID 11 does not exist
    with pytest.raises(IDNotFoundError):
        graph.find_alternative_edges(11)


def test_emtpy_list(graph):
    """
    Check that find_alternative_edges returns an empty list when no alternatives exist.
    """
    # No valid alternatives for edge 9
    assert graph.find_alternative_edges(9) == []


# ─────────────────────────────────────────────────────────────
# TEST: get_figure
# ─────────────────────────────────────────────────────────────

matplotlib.use("Agg")


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown")
def test_get_figure(graph):
    """
    Smoke test for GraphProcessor.get_figure ensuring it returns a valid matplotlib Figure object.
    """
    # Generate and check figure output
    fig = graph.get_figure(seed=0, figsize=(4, 3))
    assert isinstance(fig, Figure)

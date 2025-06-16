"""
This is the graph processing assignment.

"""

from typing import List, Tuple

import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import networkx as nx


class IDNotFoundError(Exception):
    """Raised when the vertex ID is not valid"""


class InputLengthDoesNotMatchError(Exception):
    """Raised when the edge list does not match the input list"""


class IDNotUniqueError(Exception):
    """Raised if duplicate vertex or edge ID"""


class GraphNotFullyConnectedError(Exception):
    """Raised when there are unconnected nodes in the graph"""


class GraphCycleError(Exception):
    """Raised if the graph contains a cycle"""


class EdgeAlreadyDisabledError(Exception):
    """Raised when an edge is already disabled"""


class GraphProcessor:
    """A class for processing undirected graphs"""

    def __init__(
        self,
        vertex_ids: List[int],
        edge_ids: List[int],
        edge_vertex_id_pairs: List[Tuple[int, int]],
        edge_enabled: List[bool],
        source_vertex_id: int,
    ) -> None:
        self.vertex_ids = vertex_ids
        self.edge_ids = edge_ids
        self.edge_vertex_id_pairs = edge_vertex_id_pairs
        self.edge_enabled = edge_enabled
        self.source_vertex_id = source_vertex_id

        # Check uniqueness of vertex and edge IDs
        if not set(vertex_ids).isdisjoint(set(edge_ids)):
            raise IDNotUniqueError("Duplicate vertex or edge ID")

        # Compare the length of the edge vertex ID pairs with the edge IDs
        if len(edge_vertex_id_pairs) != len(edge_ids):
            raise InputLengthDoesNotMatchError("Edge list does not match the input list")

        # Check if the vertex pairs IDs are valid IDs
        for node1, node2 in edge_vertex_id_pairs:
            if node2 not in vertex_ids or node1 not in vertex_ids:
                raise IDNotFoundError("Vertex ID is not valid")

        # Check if the number of enabled edges is the same as the number of edge IDs
        if len(edge_enabled) != len(edge_ids):
            raise InputLengthDoesNotMatchError("Edge list does not match the input list")

        # Check if the source vertex ID is a valid ID
        if source_vertex_id not in vertex_ids:
            raise IDNotFoundError("Duplicate vertex or edge ID")

        # Initialize graph
        self._graph = nx.Graph()
        self._graph.add_nodes_from(vertex_ids)

        # Add edged to the graph
        for edge_vertex_id_pair, enabled, edge_id in zip(edge_vertex_id_pairs, edge_enabled, edge_ids):
            if enabled:
                self._graph.add_edge(*edge_vertex_id_pair, id=edge_id)

        # Check if graph is connected
        if not nx.is_connected(self._graph):
            raise GraphNotFullyConnectedError()

        # Check if the graph contains cycles
        try:
            nx.find_cycle(self._graph)
            raise GraphCycleError()
        except nx.NetworkXNoCycle:
            pass

    def find_downstream_vertices(self, first_edge_id: int) -> List[int]:
        """Find downstream vertices"""
        # We check if the given edge ID is valid
        if first_edge_id not in self.edge_ids:
            raise IDNotFoundError()

        # We check if the given edge is enabled
        index = self.edge_ids.index(first_edge_id)
        if not self.edge_enabled[index]:
            return []

        # We get the vertices connected to this edge
        vertex1, vertex2 = self.edge_vertex_id_pairs[index]

        # create a copy of the graph in order to remove the edge
        graph_copy = self._graph.copy()
        graph_copy.remove_edge(vertex1, vertex2)

        # Find all the connections after the removal
        connected = list(nx.connected_components(graph_copy))

        # Find the component that contains the source vertex
        for component in connected:
            if self.source_vertex_id in component:
                source_component = component
                break

        # Return the nodes from the component that does not contain the source
        for component in connected:
            if component != source_component:
                return list(component)

        # If both vertices still belong to the same component, return empty list
        return []

    def find_alternative_edges(self, disabled_edge_id: int) -> List[int]:
        """Find alternative edges"""
        # We check if the given edge ID is valid
        if disabled_edge_id not in self.edge_ids:
            raise IDNotFoundError()

        # We check if the given edge ID is already disabled
        index = self.edge_ids.index(disabled_edge_id)
        if not self.edge_enabled[index]:
            raise EdgeAlreadyDisabledError()

        # We find all the disabled edge IDs in the graph
        disabled_edges = [self.edge_ids[i] for i, enabled in enumerate(self.edge_enabled) if not enabled]

        # Empty list in case there are no alternatives to make the graph fully connected again
        alt_list = []

        # Create a dictionary with the states of all edges in the graph and disable the given edge
        edge_status_map = dict(zip(self.edge_ids, self.edge_enabled))
        edge_status_map[disabled_edge_id] = False

        # We take all disabled edges one by one and enable them again to test for their feasbility
        # by recreating the graph for each scenario
        for test_edge_id in disabled_edges:
            temp_status_map = edge_status_map.copy()
            temp_status_map[test_edge_id] = True

            temp_graph = nx.Graph()
            temp_graph.add_nodes_from(self.vertex_ids)
            for eid, (vertex1, vertex2) in zip(self.edge_ids, self.edge_vertex_id_pairs):
                if temp_status_map[eid]:
                    temp_graph.add_edge(vertex1, vertex2, id=eid)
            # Check if the new graph is connected and check for cycles; If no cycle is found, add
            # the edge ID to the alternative list
            if nx.is_connected(temp_graph):
                try:
                    nx.find_cycle(temp_graph)
                    has_cycle = True
                except nx.NetworkXNoCycle:
                    has_cycle = False

                if not has_cycle:
                    alt_list.append(test_edge_id)

        return alt_list

        # ───────────────────────────

    # Visualization helper
    # ───────────────────────────
    def get_figure(self, *, seed: int = 42, figsize: tuple = (6, 4)):
        """
        Draw the graph.

        • Enabled lines  → solid grey
        • Disabled lines → dashed red
        • Edge-ID labels for all lines
        """
        pos = nx.spring_layout(self._graph, seed=seed)
        fig, ax = plt.subplots(figsize=figsize)

        # ── nodes ──────────────────────────────────────────────
        nx.draw_networkx_nodes(self._graph, pos, node_color="lightsteelblue", node_size=600, ax=ax)
        nx.draw_networkx_labels(self._graph, pos, font_size=12, font_weight="bold", ax=ax)

        # ── enabled edges + labels ─────────────────────────────
        nx.draw_networkx_edges(self._graph, pos, width=2, edge_color="gray", ax=ax)
        nx.draw_networkx_edge_labels(
            self._graph, pos, edge_labels=nx.get_edge_attributes(self._graph, "id"), font_size=9, ax=ax
        )

        # ── disabled edges + labels ────────────────────────────
        disabled_triplets = [
            (u, v, k)  #  u-node, v-node, edge-id
            for (u, v), en, k in zip(self.edge_vertex_id_pairs, self.edge_enabled, self.edge_ids)
            if not en
        ]
        if disabled_triplets:
            g_off = nx.Graph()
            for u, v, k in disabled_triplets:
                g_off.add_edge(u, v, id=k)

            # dashed red edges
            nx.draw_networkx_edges(g_off, pos, style="dashed", edge_color="red", width=1.5, ax=ax)

            # red ID labels on those edges
            off_labels = {(u, v): k for u, v, k in disabled_triplets}
            nx.draw_networkx_edge_labels(g_off, pos, edge_labels=off_labels, font_size=9, font_color="red", ax=ax)

        ax.set_axis_off()
        fig.tight_layout()
        plt.show()
        return fig

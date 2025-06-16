"""
Validation Module

This script defines a validation class with the following exceptions
- *TimestampsDoNotMatchError* (Timestamps of active and reactive power profiles do not match.)
- *LoadIdsDoNotMatchError* (Load IDs of active and reactive power profiles do not match.)
- *IDNotFoundError* (Vertex ID present in edge_vertex_id_pairs does not exist.)
- *InputLengthDoesNotMatchError* (The amount of vertex pairs is not equal to the amount of edges.)
- *IDNotUniqueError* (vertex and edge ids are not unique.)
- *GraphNotFullyConnectedError* (Graph not fully connected)
- *GraphCycleError* (The graph contains cycles.)
- *TooManyTransformers* (This Input data contains more than one transformer)
- *TooManySources* (This Input data contains more than one source)
- *NotAllFeederIDsareValid* (not all feeders are valid lines)
- *TransformerAndFeedersNotConnected* (Feeders and transformers are not connected to the same graph)
- *TooFewEVs* (there are less EVs profiles than symloads)

"""
# Ce lipseste:
# The IDs in active load profile and reactive load profile are matching.
# The IDs in active load profile and reactive load profile are valid IDs of sym_load.




# Load dependencies and functions from graph_processing
import json
from typing import Dict

import numpy as np

# Load dependencies and functions from calculation_module
import pandas as pd
from power_system_simulation.graph_processor import GraphProcessor as graph
from power_grid_model import CalculationType
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import assert_valid_input_data


class TooManyTransformers(Exception):
    """There are more than one Transformers in the system"""


class TooManySources(Exception):
    """There are more than one Sources in the system"""


class NotAllFeederIDsareValid(Exception):
    """The feeder IDs do not match the only possible line IDs"""


class TransformerAndFeedersNotConnected(Exception):
    """The transformer is not connected to the same node from which the feeders are connected"""


class TooFewEVs(Exception):
    """There are less EVs than Sym_loads"""


class TimestampsDoNotMatchError(Exception):
   """Exception raised when Timestamps of active and reactive power profiles do not match."""


class LoadIdsDoNotMatchError(Exception):
   """Exception raised when Load IDs of active and reactive power profiles do not match.""" 

# Make a class for:
# # The IDs in active load profile and reactive load profile are valid IDs of sym_load.


class validate_power_system_simulation:
    """summary
    (input_network_data: str), (meta_data: str), (ev_active_power_profile: str) Checks and validates all of the data for this package.
    """

    def __init__(
        self,
        input_network_data: str,
        meta_data_str: str,
        ev_active_power_profile: str,
        active_power_profile: str,
        reactive_power_profile: str
    ) -> Dict:
        """
        Check the following validity criteria for the input data. Raise or passthrough relevant errors.
            * The LV grid should be a valid PGM input data.
            * The LV grid has exactly one transformer, and one source.
            * All IDs in the LV Feeder IDs are valid line IDs.
            * All the lines in the LV Feeder IDs have the from_node the same as the to_node of the transformer.
            * The grid is fully connected in the initial state.
            * The grid has no cycles in the initial state.
            * The timestamps are matching between the active load profile, reactive load profile,
            and EV charging profile.
            * The IDs in active load profile and reactive load profile are matching.
            * The IDs in active load profile and reactive load profile are valid IDs of sym_load.
            * The number of EV charging profile is at least the same as the number of sym_load.
        """
        # Do power flow calculations with validity checks
        # Read and load input data

        ev_power_profile = pd.read_parquet(ev_active_power_profile)
        active_profile = pd.read_parquet(active_power_profile)
        reactive_profile = pd.read_parquet(reactive_power_profile)

        # Specify the encoding explicitly when opening the files
        with open(meta_data_str, "r", encoding="utf-8") as fp:
            meta_data = json.load(fp)

        with open(input_network_data, "r", encoding="utf-8") as fp:
            input_data = json_deserialize(fp.read())

        # The LV grid should be a valid PGM input data -> Validate data for PGM
        assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow)

        # Ensure there is only one transformer in the LV grid -> Check if "transformer" in meta_data is not an int
        if not isinstance(meta_data["transformer"], int):
            raise TooManyTransformers("This Input data contains more than one transformer")

        # Ensure there is only one source in the LV grid -> Check if "source" in meta_data is not an int
        if not isinstance(meta_data["source"], int):
            raise TooManySources("This Input data contains more than one source")

        # Select the line IDs and the feeder IDs from the input data and the meta data
        line_ids = [l["id"] for l in input_data["line"]]
        feeder_ids = meta_data["lv_feeders"]

        # Ensure all the IDs in the LV Feeder IDs are valid line IDs
        if not np.all(np.isin(feeder_ids, line_ids)):
            raise NotAllFeederIDsareValid("not all feeders are valid lines")

        # Filter the matrix in order to find the of transformers and the feeders and then compare their to_ and from_ nodes
        line_from_node = [l["from_node"] for l in input_data["line"]]
        line_ids_from_node = np.column_stack((line_ids, line_from_node))
        filter = np.isin(line_ids_from_node[:, 0], feeder_ids)
        filtered = line_ids_from_node[filter]
        transformer = [t["to_node"] for t in input_data["transformer"]]

        # Ensure all the lines in the LV Feeder IDs have the from_node the same as the to_node of the transformer.
        for i in filtered:
            if i[1] != transformer:
                raise TransformerAndFeedersNotConnected("not all feeders are connected to the transformer")
            
        timestamps_EV = ev_power_profile.index
        timestamps_active = active_profile.index
        timestamps_reactive = reactive_profile.index

        if not (timestamps_EV.equals(timestamps_active) and timestamps_EV.equals(timestamps_reactive)):
            raise TimestampsDoNotMatchError("Timestamps of EV, active and reactive power profiles do not match.")

        # if not (reactive_profile.columns == active_profile.columns).all():
        #    raise LoadIdsDoNotMatchError("Load IDs of active and reactive power profiles do not match.")

        # Check if:
        #   # # The IDs in active load profile and reactive load profile are valid IDs of sym_load.

        # Select the sym_loads and count them, select the number of EV-profiles and count them
        nr_house = len(input_data["sym_load"])
        EV_profiles = ev_power_profile.to_numpy()

        # Ensure the number of EV charging profile is at least the same as the number of sym_load.
        if not EV_profiles.shape[1] >= nr_house:
            raise TooFewEVs("not enough EV_profiles")

        # Calling GraphProcessor to ensure that:
        #       The grid is fully connected in the initial state.
        #       The grid has no cycles in the initial state.
        # vertex_ids = input_data["node"]["id"]
        vertex_ids = [n["id"] for n in input_data["node"]]
        transformer_id = input_data["transformer"][0]["id"]

        # The edge_ids consist of the line IDs and the transformer ID
        # edge_ids_init = pd.DataFrame(input_data["line"]["id"]).to_numpy()
        edge_ids_init = pd.DataFrame(line_ids).to_numpy()
        edge_ids = np.append(edge_ids_init, transformer_id).tolist()

        line_from_status = np.array([n["from_status"] for n in input_data["line"]])
        line_to_status = np.array([n["to_status"] for n in input_data["line"]])
        edge_enabled_init = (line_from_status == 1) & (line_to_status== 1)
        edge_enabled = np.append(edge_enabled_init, [True])

        # Tupling the vertex IDs in pairs
        # edge_vertex_id_pairs_init = list(zip(input_data["line"]["from_node"], input_data["line"]["to_node"]))
        line_to_node = [l["to_node"] for l in input_data["line"]]
        source_node = input_data["source"][0]["node"]
        edge_vertex_id_pairs_init = list(zip(line_from_node, line_to_node))
        edge_vertex_id_pairs = (edge_vertex_id_pairs_init) + [(source_node, meta_data["lv_busbar"])]
    
        graph(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_node)
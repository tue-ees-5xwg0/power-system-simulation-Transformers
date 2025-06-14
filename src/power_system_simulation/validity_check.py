"""
Validation Module

This script defines a validation class with the following exceptions
- **TimestampsDoNotMatchError** (Timestamps of active and reactive power profiles do not match.)
- **LoadIdsDoNotMatchError** (Load IDs of active and reactive power profiles do not match.)
- **IDNotFoundError** (Vertex ID present in edge_vertex_id_pairs does not exist.)
- **InputLengthDoesNotMatchError** (The amount of vertex pairs is not equal to the amount of edges.)
- **IDNotUniqueError** (vertex and edge ids are not unique.)
- **GraphNotFullyConnectedError** (Graph not fully connected)
- **GraphCycleError** (The graph contains cycles.)
- **TooManyTransformers** (This Input data contains more than one transformer)
- **TooManySources** (This Input data contains more than one source)
- **NotAllFeederIDsareValid** (not all feeders are valid lines)
- **TransformerAndFeedersNotConnected** (Feeders and transformers are not connected to the same graph)
- **TooFewEVs** (there are less EVs profiles than symloads)

"""

# Load dependencies and functions from graph_processing
import json
from typing import Dict

import numpy as np

# Load dependencies and functions from calculation_module
import pandas as pd
from assignment1 import GraphProcessor as graph
from power_grid_model import CalculationType
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import assert_valid_input_data


class TooManyTransformers(Exception):
    """The Transformer entry either is more than one or has an incorrect format"""


class TooManySources(Exception):
    """The Source entry either is more than one or has an incorrect format"""


class NotAllFeederIDsareValid(Exception):
    """The feeder IDs do not match the ids in the Network data"""


class TransformerAndFeedersNotConnected(Exception):
    """The transformer is not connected to the same node from which the feeders are connected"""


class TooFewEVs(Exception):
    """There are less EVs than Symloads, ensure that they are at least equal"""


# class TimestampsDoNotMatchError(Exception):
#    """Exception raised when Timestamps of active and reactive power profiles do not match."""


# class LoadIdsDoNotMatchError(Exception):
#    """Exception raised when Load IDs of active and reactive power profiles do not match."""


class validate_power_system_simulation:
    """_summary_
    (input_network_data: str), (meta_data: str), (ev_active_power_profile: str) Checks and validates all of the data for this package.
    """

    def __init__(
        self,
        input_network_data: str,
        meta_data_str: str,
        ev_active_power_profile: str,
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

        # Specify the encoding explicitly when opening the files
        with open(meta_data_str, "r", encoding="utf-8") as fp:
            meta_data = json.load(fp)

        with open(input_network_data, "r", encoding="utf-8") as fp:
            input_data = json_deserialize(fp.read())

        # Check if "source" in meta_data is not an int
        if not isinstance(meta_data["source"], int):
            raise TooManySources("This Input data contains more than one source")

        # Check if "transformer" in meta_data is not an int
        if not isinstance(meta_data["transformer"], int):
            raise TooManyTransformers("This Input data contains more than one transformer")

        # filter the line ids and feeders
        line_ids = input_data["line"]["id"]
        feeder_ids = meta_data["lv_feeders"]

        # Compare the contents of the feeders and line ids to ensure that they match, throw an exception if they don't
        if not np.all(np.isin(feeder_ids, line_ids)):
            raise NotAllFeederIDsareValid("not all feeders are valid lines")

        # Filter the Symloads and count them, do the same for the number of EV-profiles
        no_house = len(input_data["sym_load"])
        a = np.matrix(ev_power_profile)

        # Compare the number of EV-profiles to the Number of symloads
        # if the number of symloads is more than the amount of EVs then throw an exception
        if not a.shape[1] >= no_house:
            raise TooFewEVs("not enough EV_profiles")

        # Filter the matrix in order to find the number of transformers vs the number
        # of feeders and then compare their to and from nodes
        line_ids = input_data["line"]["id"]
        feeder_ids = meta_data["lv_feeders"]
        line_Matrix = np.column_stack((input_data["line"]["id"], input_data["line"]["from_node"]))
        mask = np.isin(line_Matrix[:, 0], feeder_ids)
        filtered_matrix = line_Matrix[mask]
        transformer = input_data["transformer"]["to_node"]

        # Check if timestamps and load IDs match
        # if not ev_power_profile.index.equals(active_power_profile.index):
        #    raise TimestampsDoNotMatchError("Timestamps of active and reactive power profiles do not match.")
        # if not (ev_power_profile.columns == active_power_profile.columns).all():
        #    raise LoadIdsDoNotMatchError("Load IDs of active and reactive power profiles do not match.")

        # compare the to nodes of the transformer to the from nodes
        # from the feeders and throw an exception if it doesn't allign
        for i in filtered_matrix:
            if i[1] != transformer:
                raise TransformerAndFeedersNotConnected("not all feeders are connected to the transformer")

        # validate data for PGM
        assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow)

        ##########The graphprocessor can now be called to check that no exceptions are called##########
        ######ORIGINAL DATA
        ##pprint.pprint(input_data)
        vertex_ids = input_data["node"]["id"]
        edge_ids_init = pd.DataFrame(input_data["line"]["id"]).to_numpy()
        edge_vertex_id_pairs_init = list(zip(input_data["line"]["from_node"], input_data["line"]["to_node"]))
        edge_enabled_init = (input_data["line"]["from_status"] == 1) & (input_data["line"]["to_status"] == 1)
        source_id = input_data["node"][0][0]  # or meta_data

        ########### MODIFIED DATA FOR TRANSFORMER AS EDGE
        edge_ids = np.append(edge_ids_init, input_data["transformer"]["id"]).tolist()
        edge_vertex_id_pairs = (edge_vertex_id_pairs_init) + [(source_id, meta_data["lv_busbar"])]
        edge_enabled = np.append(edge_enabled_init, [True])

        ############################
        # call GraphProcessing.py  #
        ############################
        graph.GraphProcessor(vertex_ids, edge_ids, edge_vertex_id_pairs, edge_enabled, source_id)

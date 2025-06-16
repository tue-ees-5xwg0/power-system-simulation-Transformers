"""N-1 Security Analysis Module.

This module implements N-1 security analysis for power distribution networks.
It provides functionality for:
- Contingency analysis of network components
- Alternative path identification
- Network reconfiguration assessment
- Security constraint validation

The module ensures power system reliability by analyzing the impact of
single component failures and identifying necessary network adjustments.

Authors:
    Andrei Dobre
    Stefan Porfir
    Diana Ionica
"""

import copy
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from power_grid_model import ComponentType

from power_system_simulation import model_processor as calc
from power_system_simulation.graph_processor import GraphProcessor as gp


class IDNotFoundError(Exception):
    """The inserted line ID is not valid."""


class IDNotInt(Exception):
    """The ID is not an int"""


class LineIDNotConnectedOnBothSides(Exception):
    """The inserted line ID is not connected at both sides."""


def nm_function(
    given_lineid: int,
    input_data_path: str,
    metadata_path: str,
    active_power_profile_path: str,
    reactive_power_profile_path: str,
) -> list[int]:
    """
    Analyze network modification impact by finding alternative line configurations.

    This function:
    - Loads grid topology and time-series power profiles.
    - Constructs a GraphProcessor instance based on the network data.
    - Validates the input line ID.
    - Identifies feasible alternative lines to replace the given line.
    - Runs power flow simulations for each alternative.
    - Returns a DataFrame with the max loading and timestamp per alternative.

    Args:
        given_lineid (int): The line ID to be tested for alternatives.
        input_data_path (str): Path to the grid model JSON file.
        metadata_path (str): Path to metadata containing transformer connections.
        active_power_profile_path (str): CSV file for active power time series.
        reactive_power_profile_path (str): CSV file for reactive power time series.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - "Alternative ID": The ID of the enabled alternative line.
            - "Max Loading": Peak loading observed during simulation.
            - "ID_max": Line ID with the peak loading.
            - "Timestamp_max": Time at which the max loading occurred.
    """

    # read input data
    active_power, reactive_power, input_data = calc.load_input_data(
        active_data_path=active_power_profile_path,
        reactive_data_path=reactive_power_profile_path,
        model_data_path=input_data_path,
    )

    with open(metadata_path, encoding="utf-8") as fp:
        meta_data = json.load(fp)

    vertex_ids = input_data[ComponentType.node]["id"].tolist()

    edge_ids_init = input_data[ComponentType.line]["id"].tolist()
    edge_vertex_id_pairs_init = list(
        zip(input_data[ComponentType.line]["from_node"], input_data[ComponentType.line]["to_node"])
    )

    edge_enabled_init = (np.array(input_data[ComponentType.line]["from_status"]) == 1) & (
        np.array(input_data[ComponentType.line]["to_status"]) == 1
    ).tolist()
    source_id = input_data[ComponentType.node][0][0].item()

    edge_ids = np.concatenate([edge_ids_init, np.array(input_data[ComponentType.transformer]["id"])]).tolist()
    edge_vertex_id_pairs = edge_vertex_id_pairs_init + [(source_id, meta_data["lv_busbar"])]

    edge_enabled = (np.append(edge_enabled_init, True)).tolist()

    gra = gp(
        vertex_ids=vertex_ids,
        edge_ids=edge_ids,
        edge_vertex_id_pairs=edge_vertex_id_pairs,
        edge_enabled=edge_enabled,
        source_vertex_id=source_id,
    )

    if not isinstance(given_lineid, int):
        raise IDNotInt("The inserted line ID is not valid")

    if given_lineid not in input_data["line"]["id"]:
        raise IDNotFoundError("The ID is not in the data!")

    from_status = input_data["line"]["from_status"][input_data["line"]["id"] == given_lineid]
    to_status = input_data["line"]["to_status"][input_data["line"]["id"] == given_lineid]

    if from_status == 0 or to_status == 0:
        raise LineIDNotConnectedOnBothSides("The inserted line ID is not connected at both sides")

    alt_list = gra.find_alternative_edges(given_lineid)

    rows = []

    date_list = [datetime(2024, 1, 1) + timedelta(minutes=i * 15) for i in range(active_power.shape[0])]

    for alt_id in alt_list:
        new_data = copy.deepcopy(input_data)

        new_data["line"]["to_status"][new_data["line"]["id"] == alt_id] = 1
        new_data["line"]["to_status"][new_data["line"]["id"] == given_lineid] = 0
        new_data["line"]["from_status"][new_data["line"]["id"] == given_lineid] = 0

        output_data = calc.run_updated_power_flow_analysis(active_power, reactive_power, new_data)

        line_loading = output_data["line"]["loading"]
        # Get position in the array
        max_idx = np.argmax(line_loading)
        row, col = np.unravel_index(max_idx, line_loading.shape)

        # Get max value and its id and the timestamp

        max_line_load = line_loading[row, col]

        max_line_load_id = output_data["line"]["id"][row, col]

        # The given line  repeats each timestamp so i have to
        # find whihc row(which timestamp) corresponds to this one

        timestamp_max = date_list[row]

        rows.append([alt_id, max_line_load, max_line_load_id, timestamp_max])

    return_df = pd.DataFrame(rows, columns=["Alternative ID", "Max Loading", "ID_max", "Timestamp_max"])
    return return_df

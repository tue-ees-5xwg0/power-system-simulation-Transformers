"""Electric Vehicle (EV) Penetration Analysis Module.

This module provides functionality to analyze the impact of electric vehicle charging on power distribution networks.
It simulates the integration of EV chargers into existing power grids and calculates the resulting effects on
network performance, including voltage profiles and line statistics.

Authors:
    Andrei Dobre
    Stefan Porfir
    Diana Ionica
"""

import json
import math
import random

import numpy as np
import pandas as pd
from power_grid_model import CalculationMethod, CalculationType, PowerGridModel, initialize_array
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import assert_valid_batch_data

from power_system_simulation.graph_processor import GraphProcessor as gp
from power_system_simulation.model_processor import line_statistics_summary, node_voltage_summary


def ev_penetration(
    input_network_data: str,
    meta_data_str: str,
    active_power_profile_path: str,
    ev_active_power_profile: str,
    percentage: float,
    seed: int,
) -> tuple:
    """Simulate the impact of electric vehicle (EV) penetration on a power distribution network.

    This function analyzes how adding EV chargers to a percentage of households affects the power grid.
    It randomly distributes EV chargers across different feeders while maintaining a balanced distribution,
    and calculates the resulting voltage profiles and line statistics.

    Args:
        input_network_data (str): Path to the JSON file containing the power grid network data.
        meta_data_str (str): Path to the JSON file containing metadata about the network structure.
        active_power_profile_path (str): Path to the parquet file containing the base active power profiles.
        ev_active_power_profile (str): Path to the parquet file containing EV charging power profiles.
        percentage (float): Percentage of households to be equipped with EV chargers (0-100).
        seed (int): Random seed for reproducible EV charger distribution.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - voltage_df: Summary of node voltages across the network
            - line_df: Summary of line statistics (power flows, losses, etc.)
    """

    with open(meta_data_str, "r", encoding="utf-8") as fp_open:
        input_metadata = json.load(fp_open)

    with open(input_network_data, "r", encoding="utf-8") as fp_open:
        input_data = json_deserialize(fp_open.read())

    active_power_profile = pd.read_parquet(active_power_profile_path)
    ev_power_profile = pd.read_parquet(ev_active_power_profile)

    model = PowerGridModel(input_data=input_data)

    vertex_ids = input_data["node"]["id"]
    edge_ids_init = np.array(input_data["line"]["id"])
    edge_vertex_id_pairs_init = list(zip(input_data["line"]["from_node"], input_data["line"]["to_node"]))
    edge_enabled_init = (np.array(input_data["line"]["from_status"]) == 1) & (
        np.array(input_data["line"]["to_status"]) == 1
    )
    source_id = input_data["node"][0][0]  # or meta_data

    edge_ids = np.concatenate([edge_ids_init, np.array(input_data["transformer"]["id"])]).tolist()
    edge_vertex_id_pairs = edge_vertex_id_pairs_init + [(source_id, input_metadata["lv_busbar"])]
    edge_enabled = np.append(edge_enabled_init, True)

    grid = gp(
        vertex_ids=vertex_ids,
        edge_ids=edge_ids,
        edge_vertex_id_pairs=edge_vertex_id_pairs,
        edge_enabled=edge_enabled,
        source_vertex_id=source_id,
    )

    random.seed(seed)

    number_feeders = len(input_metadata.get("lv_feeders", []))
    no_house = len(input_data["sym_load"])

    ev_feeder = math.floor((percentage / 100) * no_house / number_feeders)

    feeder_to_loads = {feeder: [] for feeder in input_metadata["lv_feeders"]}

    selected_ids = []

    # Iterate thrugh the network to find downstream vertices for each feeder, chekc what houses =(sym_load)
    # are  connected through which feeder

    for feeder in input_metadata["lv_feeders"]:
        downstream_vertices = grid.find_downstream_vertices(feeder)

        matched_loads = [load["id"] for load in input_data["sym_load"] if load["node"] in downstream_vertices]

        if matched_loads:
            # Randomly select a household that has EV charger, and making sure that
            # we do not select more than ev_feeder households for each feeder
            selected_ids_for_feeder = random.sample(matched_loads, min(ev_feeder, len(matched_loads)))

            feeder_to_loads[feeder].extend(selected_ids_for_feeder)

            selected_ids.extend(selected_ids_for_feeder)

    # Select the houses with EV chargers (the sym_loads) from main data
    filtered_profile = active_power_profile.loc[:, selected_ids]

    # Number of selected houses with EV chargers
    num_selected = len(selected_ids)

    # Randomly select the profil of the EV charger
    selected_columns = random.sample(ev_power_profile.columns.tolist(), num_selected)

    # Get the column of the slecetd EV charger profiles
    selected_ev_profile = ev_power_profile[selected_columns]

    # Fix the ids of the slected profiles to be equal to the ids of the slected houses(sym_loads)
    selected_ev_profile.index = filtered_profile.index
    # Same but with column names
    selected_ev_profile.columns = filtered_profile.columns

    # add the change thta the EV charger will cause on the sym_loads
    summed_profile = filtered_profile.add(selected_ev_profile, fill_value=0)

    # Create the update array for the sym_loads
    update_sym_load = initialize_array("update", "sym_load", summed_profile.shape)
    update_sym_load["id"] = summed_profile.columns.to_numpy()
    update_sym_load["p_specified"] = summed_profile.to_numpy()

    update_data = {"sym_load": update_sym_load}
    assert_valid_batch_data(input_data=input_data, update_data=update_data, calculation_type=CalculationType.power_flow)
    # calculate the updated power flow
    model_2 = model.copy()
    model_2.update(update_data=update_data)

    output_data = model_2.calculate_power_flow(
        update_data=update_data, calculation_method=CalculationMethod.newton_raphson
    )
    # Use the developed functions to summarize the results
    voltage_df = node_voltage_summary(output_data, active_power_profile.index)
    line_df = line_statistics_summary(output_data, active_power_profile.index)
    return voltage_df, line_df

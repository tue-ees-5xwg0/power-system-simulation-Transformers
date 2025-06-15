import json
import math
import random

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from power_grid_model import CalculationMethod, CalculationType, PowerGridModel, initialize_array
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import assert_valid_batch_data

from power_system_simulation.assignment1 import GraphProcessor as gp
from power_system_simulation.assignment2 import line_statistics_summary, node_voltage_summary


def ev_penetration(
    input_network_data: str,
    meta_data_str: str,
    active_power_profile_path: str,
    ev_active_power_profile: str,
    percentage: float,
    seed: int,
) -> tuple:

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


# PATH_INPUT_NETWORK_DATA = "data/test_data/input_EV_penetration/input_network_data.json"
# PATH_META_DATA = "data/test_data/input_EV_penetration/meta_data.json"
# PATH_ACTIVE_POWER_PROFILE = "data/test_data/input_EV_penetration/active_power_profile.parquet"
# PATH_EV_ACTIVE_POWER_PROFILE = "data/test_data/input_EV_penetration/ev_active_power_profile.parquet"


# PATH_EXPECTED_LINE_DF = "data/test_data/output_EV_penetration/EV_penetration_line_df.json"
# PATH_EXPECTED_VOLTAGE_DF = "data/test_data/output_EV_penetration/EV_penetration_voltage_df.json"

# Run the ev_penetration function with the provided paths and a sample percentage and seed
# voltage_df, line_df = ev_penetration(
#     PATH_INPUT_NETWORK_DATA,
#     PATH_META_DATA,
#     PATH_ACTIVE_POWER_PROFILE,
#     PATH_EV_ACTIVE_POWER_PROFILE,
#     percentage=60,  # example percentage
#     seed=42         # example seed
# )

# # ---------- write ----------
# line_df.to_json(
#     PATH_EXPECTED_LINE_DF,
#     orient="split",
#     date_format="iso",      # keep them as "YYYY-MM-DDTHH:MM:SS"
#     indent=2
# )

# # ---------- read ----------
# # line_df2 = pd.read_json(
#     PATH_EXPECTED_LINE_DF,
#     orient="split",
#     convert_dates=["Max_Loading_Timestamp", "Min_Loading_Timestamp"]   # <── parse this column
# )
# print(line_df2)
# # Read the expected dataframes from the saved JSON files
# expected_voltage_df = pd.read_json(PATH_EXPECTED_VOLTAGE_DF, orient="split")
# expected_line_df = pd.read_json(PATH_EXPECTED_LINE_DF, orient="split")

# # Explicitly set the index name after loading (if not preserved)
# expected_voltage_df.index.name = "Timestamp"
# expected_line_df.index.name = "Line_ID"

# # Convert index back to datetime for voltage_df (if applicable)
# try:
#     voltage_df.index = pd.to_datetime(voltage_df.index)
#     expected_voltage_df.index = pd.to_datetime(expected_voltage_df.index)
# except Exception:
#     pass


# # Round both dataframes to 10 decimal points before comparison
# voltage_df_rounded = voltage_df.round(10)
# expected_voltage_df_rounded = expected_voltage_df.round(10)
# line_df_rounded = line_df.round(10)
# expected_line_df_rounded = expected_line_df.round(10)


# print(line_df_rounded)
# print( "---------------------------------------")
# print(expected_line_df_rounded)

# # Compare the computed and expected dataframes using the .compare function
# voltage_comparison = voltage_df_rounded.compare(expected_voltage_df_rounded)
# line_comparison = line_df_rounded.compare(expected_line_df_rounded)

# print("Voltage DataFrame comparison:")
# print(voltage_comparison)
# print("\nLine DataFrame comparison:")
# print(line_comparison)

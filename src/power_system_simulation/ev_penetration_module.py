import json
import math
import random

import numpy as np
import pandas as pd
from power_grid_model import CalculationMethod, CalculationType, PowerGridModel, initialize_array
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import assert_valid_batch_data
from assignment1 import GraphProcessor as gp

def ev_penetration(
    input_network_data: str,
    meta_data_str: str,
    active_power_profile_path: str,
    ev_active_power_profile: str,
    percentage: float,
    seed: int,
) -> tuple:

    print(input_network_data)

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
    edge_enabled_init = (np.array(input_data["line"]["from_status"]) == 1) & \
                         (np.array(input_data["line"]["to_status"]) == 1)
    source_id = input_data["node"][0][0]  # or meta_data

    edge_ids = np.concatenate([edge_ids_init, np.array(input_data["transformer"]["id"])])
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

    no_feeders = len(input_metadata.get("lv_feeders", []))
    no_house = len(input_data["sym_load"])
    ev_feeder = math.floor((percentage / 100) * no_house / no_feeders)

    feeder_to_loads = {feeder: [] for feeder in input_metadata["lv_feeders"]}
    selected_ids = []

    for feeder in input_metadata["lv_feeders"]:
        downstream_vertices = grid.find_downstream_vertices(feeder)
        matched_loads = [load["id"] for load in input_data["sym_load"] if load["node"] in downstream_vertices]
        if matched_loads:
            selected_ids_for_feeder = random.sample(matched_loads, min(ev_feeder, len(matched_loads)))
            feeder_to_loads[feeder].extend(selected_ids_for_feeder)
            selected_ids.extend(selected_ids_for_feeder)

    filtered_profile = active_power_profile.loc[:, selected_ids]
    num_selected = len(selected_ids)
    selected_columns = random.sample(ev_power_profile.columns.tolist(), num_selected)
    selected_ev_profile = ev_power_profile[selected_columns]

    selected_ev_profile.index = filtered_profile.index
    selected_ev_profile.columns = filtered_profile.columns

    summed_profile = filtered_profile.add(selected_ev_profile, fill_value=0)

    update_sym_load = initialize_array("update", "sym_load", summed_profile.shape)
    update_sym_load["id"] = summed_profile.columns.to_numpy()
    update_sym_load["p_specified"] = summed_profile.to_numpy()

    update_data = {"sym_load": update_sym_load}
    assert_valid_batch_data(input_data=input_data, update_data=update_data, calculation_type=CalculationType.power_flow)

    model_2 = model.copy()
    model_2.update(update_data=update_data)

    output_data = model_2.calculate_power_flow(
        update_data=update_data,
        calculation_method=CalculationMethod.newton_raphson
    )

    node_voltages = output_data["node"]["u_pu"]
    node_ids = output_data["node"]["id"]
    line_loadings = output_data["line"]["loading"]
    line_ids = output_data["line"]["id"]
    p_from = output_data["line"]["p_from"]
    p_to = output_data["line"]["p_to"]
    timestamps = active_power_profile.index

    max_voltage = np.max(node_voltages, axis=1)
    min_voltage = np.min(node_voltages, axis=1)
    max_voltage_node = node_ids[np.arange(len(node_voltages)), np.argmax(node_voltages, axis=1)]
    min_voltage_node = node_ids[np.arange(len(node_voltages)), np.argmin(node_voltages, axis=1)]

    voltage_df = pd.DataFrame({
        "Timestamp": timestamps,
        "Max_Voltage": max_voltage,
        "Max_Voltage_Node": max_voltage_node,
        "Min_Voltage": min_voltage,
        "Min_Voltage_Node": min_voltage_node
    }).set_index("Timestamp")

    line_results = []
    for i, line_id in enumerate(np.unique(line_ids)):
        mask = (line_ids == line_id)
        loadings = line_loadings[mask]
        max_loading = loadings.max()
        min_loading = loadings.min()
        max_loading_time = timestamps[np.argmax(loadings)]
        min_loading_time = timestamps[np.argmin(loadings)]
        energy_losses = abs(p_from[:, i] + p_to[:, i])
        energy_loss_kwh = np.trapz(energy_losses) / 1000

        line_results.append({
            "Line_ID": line_id,
            "Total_Loss": energy_loss_kwh,
            "Max_Loading": max_loading,
            "Max_Loading_Timestamp": max_loading_time,
            "Min_Loading": min_loading,
            "Min_Loading_Timestamp": min_loading_time
        })

    line_df = pd.DataFrame(line_results).set_index("Line_ID")
    return voltage_df, line_df

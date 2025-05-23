"""
This is the power gird modeling assignment 
Editors: Andrei Dobre, Diana Ionica, Stefan Porfir 
"""
import pandas as pd
import numpy as np
from power_grid_model import (
    PowerGridModel,
    CalculationType,
    CalculationMethod,
    ComponentType,
    DatasetType,
    initialize_array)
from power_grid_model.validation import (
    assert_valid_input_data,
    assert_valid_batch_data
)
from power_grid_model.utils import json_deserialize


def analyse_model(
        active_data_path: str,
        reactive_data_path: str,
        model_data: str) -> PowerGridModel:
    """
    Update the model with the active and reactive data.
    """
    # Read the model data and deserialize it
    with open(model_data, encoding="utf-8") as fp:
        data = fp.read()
    dataset = json_deserialize(data)

    # Read the parquet files for active and reactive data
    active_data = pd.read_parquet(active_data_path, engine='pyarrow')
    reactive_data = pd.read_parquet(reactive_data_path, engine='pyarrow')

    update_data = initialize_array(
        DatasetType.update,
        ComponentType.sym_load,
        active_data.shape)

    update_data["id"] = active_data.columns.to_numpy()
    update_data["p_specified"] = active_data.to_numpy()
    update_data["q_specified"] = reactive_data.to_numpy()

    update_model = {ComponentType.sym_load: update_data}

    model = PowerGridModel(dataset)
    assert_valid_batch_data(
        input_data=dataset,
        update_data=update_model,
        calculation_type=CalculationType.power_flow)

    output = model.calculate_power_flow(
        calculation_method=CalculationMethod.newton_raphson,
        update_data=update_model)

    node_voltages = output[ComponentType.node]["u_pu"]
    ids = output[ComponentType.node]["id"][0]

    required_columns = [
        "Max_Voltage",
        "Max_Voltage_Node",
        "Min_Voltage",
        "Min_Voltage_Node"]
    node_voltages_df = pd.DataFrame(
        index=active_data.index,
        columns=required_columns)

    max_voltage = np.max(node_voltages, axis=1)
    max_voltages_id = np.argmax(node_voltages, axis=1)
    selected_ids = ids[max_voltages_id]

    min_voltage = np.min(node_voltages, axis=1)
    min_voltages_id = np.argmin(node_voltages, axis=1)
    selected_ids_min = ids[min_voltages_id]

    node_voltages_df["Max_Voltage"] = max_voltage
    node_voltages_df["Max_Voltage_Node"] = selected_ids
    node_voltages_df["Min_Voltage"] = min_voltage
    node_voltages_df["Min_Voltage_Node"] = selected_ids_min

    lines = output[ComponentType.line]
    loading_data_per_line = np.transpose(output[ComponentType.line]["loading"])
    p_to_per_line = np.transpose(output[ComponentType.line]["p_to"])
    p_from_per_line = np.transpose(output[ComponentType.line]["p_from"])

    timestamps = active_data.index.to_numpy()
    columns_line = [
        "Total_Loss",
        "Max_Loading",
        "Max_Loading_Timestamp",
        "Min_Loading",
        "Min_Loading_Timestamp"]
    line_df = pd.DataFrame(index=list(lines["id"])[0], columns=columns_line)
    line_df.index.name = "Line_ID"

    loss_energy = abs(p_to_per_line + p_from_per_line)
    total_energy_loss = np.trapz(loss_energy) / 1000

    max_loading = np.max(loading_data_per_line, axis=1)
    min_loading = np.min(loading_data_per_line, axis=1)
    position_max_loading = np.argmax(loading_data_per_line, axis=1)
    position_min_loading = np.argmin(loading_data_per_line, axis=1)

    timestamp_max = timestamps[position_max_loading]
    timestamp_min = timestamps[position_min_loading]

    line_df["Total_Loss"] = total_energy_loss
    line_df["Max_Loading"] = max_loading
    line_df["Max_Loading_Timestamp"] = timestamp_max
    line_df["Min_Loading"] = min_loading
    line_df["Min_Loading_Timestamp"] = timestamp_min

    return node_voltages_df, line_df

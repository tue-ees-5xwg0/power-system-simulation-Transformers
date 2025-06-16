"""Optimal Tap Setting Analysis Module.

This module provides functionality for analyzing and optimizing transformer tap settings
in power distribution networks. It includes capabilities for:
- Tap position optimization
- Voltage regulation analysis
- Network performance assessment
- Optimal control strategy determination

The module helps in maintaining optimal voltage levels throughout the network
by determining the best transformer tap positions.

Authors:
    Andrei Dobre
    Stefan Porfir
    Diana Ionica
"""

from pathlib import Path

from power_grid_model.utils import json_serialize_to_file

from power_system_simulation import model_processor as calc


class InvalidOptimizeInput(Exception):
    """Exception raised when user inputs an invalid optimize_by value."""


def optimal_tap_position(
    input_network_data: str, active_power_profile_path: str, reactive_power_profile_path: str, optimize_by: int
) -> int:
    """
    Calculates the optimal transformer tap position based on user-defined optimization metric.

    Args:
        input_network_data (str): Path to the network data file.
        active_power_profile_path (str): Path to the active power profile file.
        reactive_power_profile_path (str): Path to the reactive power profile file.
        optimize_by (int): 0 for minimizing total losses, 1 for minimizing voltage deviation.

    Raises:
        InvalidOptimizeInput: If optimize_by is not 0 or 1.

    Returns:
        int: Optimal tap position based on the selected optimization metric.
    """
    if optimize_by not in (0, 1):
        raise InvalidOptimizeInput("Option to optimize by is invalid, please only input 0 or 1.")

    # load input data
    active_power_df, reactive_power_df, input_network_data_dict = calc.load_input_data(
        active_power_profile_path, reactive_power_profile_path, input_network_data
    )

    path_input_network_data_temp = Path(input_network_data).parent / "input_network_data_temp.json"

    pos_min = input_network_data_dict["transformer"]["tap_min"][0]
    pos_max = input_network_data_dict["transformer"]["tap_max"][0]

    total_losses_min = float("inf")
    average_dev_max_node_min = float("inf")
    total_losses_min_tap_pos = pos_max
    average_dev_min_tap_pos = pos_max

    for tap_pos in range(pos_max, pos_min + 1):
        input_network_data_dict["transformer"]["tap_pos"] = tap_pos
        json_serialize_to_file(path_input_network_data_temp, input_network_data_dict)

        result_power_flow_analysis = calc.run_updated_power_flow_analysis(
            active_power_df, reactive_power_df, input_network_data_dict
        )
        voltage_summary = calc.node_voltage_summary(result_power_flow_analysis, reactive_power_df.index)
        losses_summary = calc.line_statistics_summary(result_power_flow_analysis, reactive_power_df.index)

        average_dev_max_node = (voltage_summary["Max_Voltage_Node"] - 1).abs().mean()
        # total_losses = losses_summary["Total_Loss"].sum()

        if tap_pos == pos_max:
            total_losses = losses_summary["Total_Loss"].sum()
            total_losses_min = total_losses
            total_losses_min_tap_pos = tap_pos
            average_dev_max_node_min = average_dev_max_node
            average_dev_min_tap_pos = tap_pos
        else:

            if total_losses < total_losses_min:
                total_losses_min = total_losses
                total_losses_min_tap_pos = tap_pos

            if average_dev_max_node < average_dev_max_node_min:
                average_dev_max_node_min = average_dev_max_node
                average_dev_min_tap_pos = tap_pos

    if optimize_by == 0:
        return total_losses_min_tap_pos

    if optimize_by == 1:
        return average_dev_min_tap_pos

    return

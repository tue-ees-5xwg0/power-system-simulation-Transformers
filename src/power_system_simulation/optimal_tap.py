from pathlib import Path

import numpy as np
from power_grid_model.utils import json_deserialize, json_serialize_to_file

from power_system_simulation import assignment2 as calc


class InvalidOptimizeInput(Exception):
    """Exception raised when user inputs an invalid optimize_by value."""

    pass


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

    with open(input_network_data, "r", encoding="utf-8") as fp:
        input_data = json_deserialize(fp.read())

    input_network_data_alt = Path(input_network_data).parent / "input_network_data_alt.json"

    pos_min = input_data["transformer"]["tap_min"][0]
    pos_max = input_data["transformer"]["tap_max"][0]

    total_losses_min = float("inf")
    average_dev_max_node_min = float("inf")
    total_losses_min_tap_pos = pos_max
    average_dev_min_tap_pos = pos_max

    for tap_pos in range(pos_min, pos_max - 1, -1):
        input_data["transformer"]["tap_pos"] = tap_pos
        json_serialize_to_file(input_network_data_alt, input_data)

        voltage_results, line_results = calc.run_updated_power_flow_analysis(
            input_network_data_alt, active_power_profile_path, reactive_power_profile_path
        )

        average_dev_max_node = (voltage_results["Max_Voltage_Node"] - 1).abs().mean()
        total_losses = line_results["Total_Loss"].sum()

        if total_losses < total_losses_min:
            total_losses_min = total_losses
            total_losses_min_tap_pos = tap_pos

        if average_dev_max_node < average_dev_max_node_min:
            average_dev_max_node_min = average_dev_max_node
            average_dev_min_tap_pos = tap_pos

    return total_losses_min_tap_pos if optimize_by == 0 else average_dev_min_tap_pos

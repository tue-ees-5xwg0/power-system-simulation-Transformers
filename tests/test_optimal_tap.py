# pylint: disable= missing-module-docstring, import-error, no-name-in-module
""""Test for optimal_tap module"""
from pathlib import Path

import pytest

from power_system_simulation.optimal_tap import InvalidOptimizeInput, optimal_tap_position

DATA_PATH = Path(__file__).parent / "stefan_data"

PATH_INPUT_NETWORK_DATA = DATA_PATH / "input_network_data.json"
PATH_ACTIVE_POWER_PROFILE = DATA_PATH / "active_power_profile.parquet"
PATH_REACTIVE_POWER_PROFILE = DATA_PATH / "reactive_power_profile.parquet"


def test_optimal_tap_0():
    """Test the optimal tap functionality with a custom made input."""
    result = optimal_tap_position(
        input_network_data=str(PATH_INPUT_NETWORK_DATA),
        active_power_profile_path=str(PATH_ACTIVE_POWER_PROFILE),
        reactive_power_profile_path=str(PATH_REACTIVE_POWER_PROFILE),
        optimize_by=0,
    )
    assert result == 1


def test_optimal_tap_1():
    """Test the optimal tap functionality with a custom made input."""
    result = optimal_tap_position(
        input_network_data=str(PATH_INPUT_NETWORK_DATA),
        active_power_profile_path=str(PATH_ACTIVE_POWER_PROFILE),
        reactive_power_profile_path=str(PATH_REACTIVE_POWER_PROFILE),
        optimize_by=1,
    )
    assert result == 3


def test_invalid():
    """"Test for invalid input data"""
    with pytest.raises(InvalidOptimizeInput):
        result = optimal_tap_position(
            input_network_data=str(PATH_INPUT_NETWORK_DATA),
            active_power_profile_path=str(PATH_ACTIVE_POWER_PROFILE),
            reactive_power_profile_path=str(PATH_REACTIVE_POWER_PROFILE),
            optimize_by=5,
        )

from power_system_simulation.ev_penetration_module import (ev_penetration
)
import pandas as pd
from pandas.testing import assert_frame_equal

PATH_INPUT_NETWORK_DATA = "data/test_data/input_EV_penetration/input_network_data.json"
PATH_META_DATA = "data/test_data/input_EV_penetration/meta_data.json"
PATH_ACTIVE_POWER_PROFILE = "data/test_data/input_EV_penetration/active_power_profile.parquet"
PATH_EV_ACTIVE_POWER_PROFILE = "data/test_data/input_EV_penetration/ev_active_power_profile.parquet"


PATH_EXPECTED_LINE_DF = "data/test_data/output_EV_penetration/EV_penetration_line_df.json"
PATH_EXPECTED_VOLTAGE_DF = "data/test_data/output_EV_penetration/EV_penetration_voltage_df.json"

def test_ev_penetration():	
    """Test the ev_penetration function with a sample input."""
    percentage = 60
    seed = 42

    result = ev_penetration(
        input_network_data=PATH_INPUT_NETWORK_DATA,
        meta_data_str=PATH_META_DATA,
        active_power_profile_path=PATH_ACTIVE_POWER_PROFILE,
        ev_active_power_profile=PATH_EV_ACTIVE_POWER_PROFILE,
        percentage=percentage,
        seed=seed
    )
    voltage_df  = result[0]
    line_df = result[1]
    voltage_df_correct = pd.read_json(PATH_EXPECTED_VOLTAGE_DF, orient="split")
    line_df_correct = pd.read_json(
    PATH_EXPECTED_LINE_DF,
    orient="split",
    convert_dates=["Max_Loading_Timestamp", "Min_Loading_Timestamp"])


    assert isinstance(result, tuple), "Result should be a tuple."
    assert len(result) == 2, "Result should contain three elements."
    assert isinstance(result[0], pd.DataFrame), "First element should be a DataFrame."
    assert isinstance(result[1], pd.DataFrame), "Second element should be a DataFrame."

    voltage_df = voltage_df.sort_index().sort_index(axis=1)
    voltage_df_correct = voltage_df_correct.sort_index().sort_index(axis=1)
    voltage_df_correct.index.name = "Timestamp"
    line_df = line_df.sort_index().sort_index(axis=1)
    line_df_correct = line_df_correct.sort_index().sort_index(axis=1)
    line_df_correct.index.name = "Line_ID"


    assert ((voltage_df.round(10).compare(voltage_df_correct.round(10))).empty)
    assert ((line_df.round(10).compare(line_df_correct.round(10))).empty)


import pytest
from pathlib import Path
import pandas as pd
from power_system_simulation import n1_calculation as n1
from pandas.testing import assert_frame_equal

BASE_DIR = Path(__file__).parent / "stefan_data"

INPUT_DATA_PATH   = BASE_DIR / "input_network_data.json"
ACTIVE_DATA_PATH  = BASE_DIR / "active_power_profile.parquet"
REACTIVE_DATA_PATH = BASE_DIR / "reactive_power_profile.parquet"
METADATA_PATH     = BASE_DIR / "meta_data.json"
CORRECT_SOL_PATH  = BASE_DIR / "n1_calculation_correct.json"
EXPECTED_COLS = ["Alternative ID",
                 "Max Loading",
                 "ID_max",
                 "Timestamp_max"]
def test_corect_output():
    corect_output = pd.read_json(CORRECT_SOL_PATH, orient="records", lines=True)
    test = n1.nm_function(20, INPUT_DATA_PATH, METADATA_PATH, ACTIVE_DATA_PATH, REACTIVE_DATA_PATH)
    assert (assert_frame_equal(test , corect_output,check_dtype=False) == None)

def test_with_no_output():
    test = n1.nm_function(19, INPUT_DATA_PATH, METADATA_PATH, ACTIVE_DATA_PATH, REACTIVE_DATA_PATH)
    assert len(test)  == 0
    assert list(test.columns) == EXPECTED_COLS

def test_ID_not_found():
    with pytest.raises(n1.IDNotFoundError):
       n1.nm_function(1000, INPUT_DATA_PATH, METADATA_PATH, ACTIVE_DATA_PATH, REACTIVE_DATA_PATH)

def test_line_not_connected_both_sides():
    with pytest.raises(n1.LineIDNotConnectedOnBothSides):
        n1.nm_function(24, INPUT_DATA_PATH, METADATA_PATH, ACTIVE_DATA_PATH, REACTIVE_DATA_PATH)
def test_given_lineid_is_not_int():
    with pytest.raises(n1.IDNotInt):
        n1.nm_function("value", INPUT_DATA_PATH, METADATA_PATH, ACTIVE_DATA_PATH, REACTIVE_DATA_PATH)


     
    






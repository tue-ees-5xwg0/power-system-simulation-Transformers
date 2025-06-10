"""
This module contains tests for the power system simulation assignment 2.
"""

import pandas as pd
import pytest

from power_system_simulation.assignment2 import (
    IDsDoNotMatchError,
    TimestampMismatchError,
    ValidationException,
    data_processing,
    line_statistics_summary,
    load_input_data,
    node_voltage_summary,
    run_updated_power_flow_analysis,
)

MODEL_DATA = "data/test_data/input/input_network_data.json"
ACTIVE_DATA_PATH = "data/test_data/input/active_power_profile.parquet"
REACTIVE_DATA_PATH = "data/test_data/input/reactive_power_profile.parquet"
WRONG_TMESTAMP_PATH = "data/test_data/input/active_power_profile_wrong_datetime.parquet"
WRONG_IDS_PATH = "data/test_data/input/active_power_profile_wrong_ids.parquet"
DIFFERENT_SHAPE_PATH = "data/test_data/input/active_power_profile_different_shape.parquet"


CORRECT_ROW_PER_LINE_PATH = "data/test_data/expected_output/output_table_row_per_line.parquet"
CORRECT_ROW_PER_TIMESTAMP_PATH = "data/test_data/expected_output/output_table_row_per_timestamp.parquet"


def test_load_input_data():
    """
    Test the load_input_data function for correct loading and validation of input data.
    This test checks that the active and reactive dataframes have the same shape, index, and columns.
    """
    active_df, reactive_df,_ = load_input_data(ACTIVE_DATA_PATH, REACTIVE_DATA_PATH, MODEL_DATA)
    assert active_df.shape == reactive_df.shape, "Active and reactive data must have the same shape."
    assert active_df.index.equals(reactive_df.index), "Active and reactive data must share the same index."
    assert active_df.columns.equals(reactive_df.columns), "Active and reactive data must share the same column IDs."


def test_load_input_data_wrong_timestamp():
    """
    Test that the load_input_data function raises a TimestampMismatchError
    when the active and reactive dataframes have different timestamps.
    """
    with pytest.raises(TimestampMismatchError):
        load_input_data(WRONG_TMESTAMP_PATH, REACTIVE_DATA_PATH, MODEL_DATA)


def test_load_input_data_wrong_ids():
    """
    Test that the load_input_data function raises an IDsDoNotMatchError
    when the active and reactive dataframes have different column IDs.
    """
    with pytest.raises(IDsDoNotMatchError):
        load_input_data(WRONG_IDS_PATH, REACTIVE_DATA_PATH, MODEL_DATA)


def test_load_input_data_different_shape():
    """
    Test that the load_input_data function raises a ValidationException
    when the active and reactive dataframes have different shapes.
    """
    with pytest.raises(ValidationException):
        load_input_data(DIFFERENT_SHAPE_PATH, REACTIVE_DATA_PATH, MODEL_DATA)


def test_load_input_data_invalid_file():
    """
    Test that the load_input_data function raises a FileNotFoundError
    when the provided file paths are invalid.
    """
    with pytest.raises(FileNotFoundError):
        load_input_data("invalid_path.parquet", REACTIVE_DATA_PATH, MODEL_DATA)

    with pytest.raises(FileNotFoundError):
        load_input_data(ACTIVE_DATA_PATH, "invalid_path.parquet", MODEL_DATA)

    with pytest.raises(FileNotFoundError):
        load_input_data(ACTIVE_DATA_PATH, REACTIVE_DATA_PATH, "invalid_model.json")


def test_node_voltage_summary():
    """
    Test the node_voltage_summary function to ensure it returns the correct summary DataFrame.
    This test checks that the output matches the expected DataFrame for each timestamp.
    """
    active_df, reactive_df, dataset = load_input_data(ACTIVE_DATA_PATH, REACTIVE_DATA_PATH, MODEL_DATA)
    output = run_updated_power_flow_analysis(active_df, reactive_df, dataset)
    node_summary_correct_row_per_timestamp = pd.read_parquet(CORRECT_ROW_PER_TIMESTAMP_PATH, engine="pyarrow")

    assert node_voltage_summary(output, active_df.index).equals(node_summary_correct_row_per_timestamp)


def test_line_voltage_summary():
    """
    Test the line_statistics_summary function to ensure it returns the correct summary DataFrame.
    This test checks that the output matches the expected DataFrame for each line.
    """
    active_df, reactive_df, dataset = load_input_data(ACTIVE_DATA_PATH, REACTIVE_DATA_PATH, MODEL_DATA)
    output = run_updated_power_flow_analysis(active_df, reactive_df, dataset)
    line_summary_correct_row_per_line = pd.read_parquet(CORRECT_ROW_PER_LINE_PATH, engine="pyarrow")

    assert (
        line_statistics_summary(output, active_df.index).round(14).equals(line_summary_correct_row_per_line.round(14))
    )


# Add this test to test_assignment2.py


def test_data_processing():
    """
    Test the data_processing function to ensure it processes the
      input data correctly and returns the expected DataFrames.
    """

    line_summary_correct_row_per_line = pd.read_parquet(CORRECT_ROW_PER_LINE_PATH, engine="pyarrow")
    node_summary_correct_row_per_timestamp = pd.read_parquet(CORRECT_ROW_PER_TIMESTAMP_PATH, engine="pyarrow")

    node_df, line_df = data_processing(ACTIVE_DATA_PATH, REACTIVE_DATA_PATH, MODEL_DATA)

    assert node_df.equals(node_summary_correct_row_per_timestamp)

    assert line_df.round(14).equals(line_summary_correct_row_per_line.round(14))

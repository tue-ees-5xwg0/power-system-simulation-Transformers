"""
Unit tests for assignment2 helper functions.

We keep the test completely independent from `power_grid_model`s heavy
numerical engine by feeding in small, deterministic NumPy arrays.
"""
from _future_ import annotations

import numpy as np
import pandas as pd

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(_file_).resolve().parents[1] / "src"))
# import from your package ─────────────────────────────────────────
from power_system_simulation.assignment2 import (  # adjust the import path if needed
    ComponentType,
    data_processing,
    line_statistics_summary,
    node_voltage_summary,
)

# tiny dummy data used by several tests
TIMESTAMPS = pd.date_range("2024-01-01", periods=3, freq="H")

# 3 × 3 matrix: (timestamps, nodes)
NODE_U_PU = np.array(
    [
        [1.00, 0.95, 1.05],  # t0
        [1.02, 0.98, 1.04],  # t1
        [0.99, 0.97, 1.06],  # t2
    ]
)
NODE_IDS = np.array([[10, 20, 30]])  # shape (1, n_nodes)

# 3 × 2 matrix: (timestamps, lines)
LINE_LOADING = np.array([[70, 50], [90, 30], [80, 40]])
P_TO = np.array([[10, -5], [12, -4], [11, -3]])
P_FROM = np.array([[-9, 4], [-11, 3], [-10, 2]])
LINE_IDS = np.array([[100, 200]])

# Compose an output-dict that uses the same ComponentType keys
OUTPUT = {
    ComponentType.node: {"u_pu": NODE_U_PU, "id": NODE_IDS},
    ComponentType.line: {
        "loading": LINE_LOADING,
        "p_to": P_TO,
        "p_from": P_FROM,
        "id": LINE_IDS,
    },
}


# ─────────────────────────────────────────────────────────────────────
#  node_voltage_summary
# ─────────────────────────────────────────────────────────────────────
def test_node_voltage_summary_basic():
    df = node_voltage_summary(OUTPUT, TIMESTAMPS)

    # shape and columns
    assert df.shape == (3, 4)
    assert list(df.columns) == [
        "Max_Voltage",
        "Max_Voltage_Node",
        "Min_Voltage",
        "Min_Voltage_Node",
    ]

    # check a couple of specific values
    assert df.loc[TIMESTAMPS[0], "Max_Voltage"] == 1.05
    assert df.loc[TIMESTAMPS[2], "Min_Voltage_Node"] == 20


# ─────────────────────────────────────────────────────────────────────
#  line_statistics_summary
# ─────────────────────────────────────────────────────────────────────
def test_line_statistics_summary_basic():
    df = line_statistics_summary(OUTPUT, TIMESTAMPS)

    # expected index and columns
    assert list(df.index) == [100, 200]
    assert set(df.columns) == {
        "Total_Loss",
        "Max_Loading",
        "Max_Loading_Timestamp",
        "Min_Loading",
        "Min_Loading_Timestamp",
    }

    # total loss should be positive and identical for both dummy lines here
    assert np.allclose(df["Total_Loss"], df["Total_Loss"][0])
    assert df["Total_Loss"][0] > 0

    # max / min loading sanity
    assert df["Max_Loading"][100] == 90
    assert df["Min_Loading"][200] == 30


# ─────────────────────────────────────────────────────────────────────
#  data_processing wrapper
# ─────────────────────────────────────────────────────────────────────
def test_data_processing_combines_helpers():
    node_df, line_df = data_processing(OUTPUT, TIMESTAMPS)

    # wrapper must delegate to helpers faithfully
    expected_node = node_voltage_summary(OUTPUT, TIMESTAMPS)
    expected_line = line_statistics_summary(OUTPUT, TIMESTAMPS)

    pd.testing.assert_frame_equal(node_df, expected_node)
    pd.testing.assert_frame_equal(line_df, expected_line)
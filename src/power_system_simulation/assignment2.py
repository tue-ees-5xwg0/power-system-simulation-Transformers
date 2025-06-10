"""
Process power-system-simulation graphs.
"""
from _future_ import annotations

import numpy as np
import pandas as pd
from power_grid_model import (
    CalculationMethod,
    CalculationType,
    ComponentType,
    DatasetType,
    PowerGridModel,
    initialize_array,
)
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import assert_valid_batch_data


class ValidationException(Exception):
    """Raised when active/reactive shapes differ."""


class IDsDoNotMatchError(Exception):
    """Raised when column IDs do not match."""


class TimestampMismatchError(Exception):
    """Raised when indices do not match."""


# ──────────────────────────────────────────────────────────
# 1)  I/O  +  BASIC VALIDATION
# ──────────────────────────────────────────────────────────
def load_input_data(
    active_data_path: str,
    reactive_data_path: str,
    model_data_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read parquet / json files and return active-df, reactive-df, dataset-dict."""
    with open(model_data_path, encoding="utf-8") as fp:
        dataset = json_deserialize(fp.read())

    active_df = pd.read_parquet(active_data_path, engine="pyarrow")
    reactive_df = pd.read_parquet(reactive_data_path, engine="pyarrow")

    if active_df.shape != reactive_df.shape:
        raise ValidationException("Active and reactive data must have the same shape.")
    if not active_df.index.equals(reactive_df.index):
        raise TimestampMismatchError("Active and reactive data must share the same index.")
    if not active_df.columns.equals(reactive_df.columns):
        raise IDsDoNotMatchError("Active and reactive data must share the same column IDs.")

    return active_df, reactive_df, dataset


# ──────────────────────────────────────────────────────────
# 2)  SOLVER RUN
# ──────────────────────────────────────────────────────────
def run_updated_power_flow_analysis(
    active_df: pd.DataFrame,
    reactive_df: pd.DataFrame,
    dataset: dict,
) -> dict:
    """Build update-array, validate, run Newton-Raphson, return raw output dict."""
    update_data = initialize_array(
        DatasetType.update, ComponentType.sym_load, active_df.shape
    )
    update_data["id"] = active_df.columns.to_numpy()
    update_data["p_specified"] = active_df.to_numpy()
    update_data["q_specified"] = reactive_df.to_numpy()

    update_model = {ComponentType.sym_load: update_data}

    model = PowerGridModel(dataset)
    assert_valid_batch_data(
        input_data=dataset,
        update_data=update_model,
        calculation_type=CalculationType.power_flow,
    )

    return model.calculate_power_flow(
        calculation_method=CalculationMethod.newton_raphson,
        update_data=update_model,
    )


# ──────────────────────────────────────────────────────────
# 3)  POST-PROCESSING HELPERS
# ──────────────────────────────────────────────────────────
def node_voltage_summary(output: dict, timestamps: pd.Index) -> pd.DataFrame:
    """Return one DataFrame with max / min voltage per timestamp."""
    node_voltages = output[ComponentType.node]["u_pu"]
    ids = output[ComponentType.node]["id"][0]

    max_v = np.max(node_voltages, axis=1)
    min_v = np.min(node_voltages, axis=1)
    max_ids = ids[np.argmax(node_voltages, axis=1)]
    min_ids = ids[np.argmin(node_voltages, axis=1)]

    return pd.DataFrame(
        {
            "Max_Voltage": max_v,
            "Max_Voltage_Node": max_ids,
            "Min_Voltage": min_v,
            "Min_Voltage_Node": min_ids,
        },
        index=timestamps,
    )


def line_statistics_summary(output: dict, timestamps: pd.Index) -> pd.DataFrame:
    """Return one DataFrame with energy-loss & loading stats per line."""
    lines = output[ComponentType.line]
    load = output[ComponentType.line]["loading"].T
    p_to = output[ComponentType.line]["p_to"].T
    p_from = output[ComponentType.line]["p_from"].T

    loss_energy = np.abs(p_to + p_from)
    total_loss = np.trapz(loss_energy, axis=1) / 1000  # kWh if p in kW

    max_load = np.max(load, axis=1)
    min_load = np.min(load, axis=1)
    ts_max = timestamps[np.argmax(load, axis=1)]
    ts_min = timestamps[np.argmin(load, axis=1)]

    line_df = pd.DataFrame(
        {
            "Total_Loss": total_loss,
            "Max_Loading": max_load,
            "Max_Loading_Timestamp": ts_max,
            "Min_Loading": min_load,
            "Min_Loading_Timestamp": ts_min,
        },
        index=lines["id"][0],
    )
    line_df.index.name = "Line_ID"
    return line_df


# ──────────────────────────────────────────────────────────
# 4)  HIGH-LEVEL WRAPPER
# ──────────────────────────────────────────────────────────
def data_processing(
    active_data_path: str,
    reactive_data_path: str,
    model_data_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full pipeline and return node- & line-level DataFrames."""
    active_df, reactive_df, dataset = load_input_data(
        active_data_path, reactive_data_path, model_data_path
    )
    output = run_updated_power_flow_analysis(active_df, reactive_df, dataset)
    node_df = node_voltage_summary(output, active_df.index)
    line_df = line_statistics_summary(output, active_df.index)
    return node_df, line_df
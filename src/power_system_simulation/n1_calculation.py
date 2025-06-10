import copy
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from assignment1 import GraphProcessor as gp
from power_grid_model import CalculationMethod, CalculationType, PowerGridModel, initialize_array
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import assert_valid_input_data
from prettytable import PrettyTable


class IDNotFoundError(Exception):
    """The inserted line ID is not valid."""

    pass


class LineIDNotConnectedOnBothSides(Exception):
    """The inserted line ID is not connected at both sides."""

    pass


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"Execution time for {self.name} is {elapsed:.6f} s")


def nm_function(
    given_lineid: int,
    input_data_path: str,
    metadata_path: str,
    active_power_profile_path: str,
    reactive_power_profile_path: str,
) -> list[int]:

    with open(metadata_path, "r", encoding="utf-8") as fp:
        meta_data = json.load(fp)

    with open(input_data_path, "r", encoding="utf-8") as fp:
        input_data = json_deserialize(fp.read())

    assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow)

    active_power_profile = pd.read_parquet(active_power_profile_path)
    reactive_power_profile = pd.read_parquet(reactive_power_profile_path)

    vertex_ids = input_data["node"]["id"]
    edge_ids_init = np.array(input_data["line"]["id"])
    edge_vertex_id_pairs_init = list(zip(input_data["line"]["from_node"], input_data["line"]["to_node"]))
    edge_enabled_init = (np.array(input_data["line"]["from_status"]) == 1) & (
        np.array(input_data["line"]["to_status"]) == 1
    )
    source_id = input_data["node"][0][0]

    edge_ids = np.concatenate([edge_ids_init, np.array(input_data["transformer"]["id"])])
    edge_vertex_id_pairs = edge_vertex_id_pairs_init + [(source_id, meta_data["lv_busbar"])]
    edge_enabled = np.append(edge_enabled_init, True)

    gra = gp(
        vertex_ids=vertex_ids,
        edge_ids=edge_ids,
        edge_vertex_id_pairs=edge_vertex_id_pairs,
        edge_enabled=edge_enabled,
        source_vertex_id=source_id,
    )

    if not isinstance(given_lineid, int):
        raise IDNotFoundError("The inserted line ID is not valid")

    from_status = input_data["line"]["from_status"][input_data["line"]["id"] == given_lineid]
    to_status = input_data["line"]["to_status"][input_data["line"]["id"] == given_lineid]
    if int(from_status) == 0 or int(to_status) == 0:
        raise LineIDNotConnectedOnBothSides("The inserted line ID is not connected at both sides")

    alt_list = gra.find_alternative_edges(given_lineid)

    table = PrettyTable(["Alternative ID", "Max Loading", "ID_max", "Timestamp_max"])

    date_list = [datetime(2024, 1, 1) + timedelta(minutes=i * 15) for i in range(active_power_profile.shape[0])]

    for alt_id in alt_list:
        new_data = copy.deepcopy(input_data)

        new_data["line"]["to_status"][new_data["line"]["id"] == alt_id] = 1
        new_data["line"]["to_status"][new_data["line"]["id"] == given_lineid] = 0
        new_data["line"]["from_status"][new_data["line"]["id"] == given_lineid] = 0

        model = PowerGridModel(input_data=new_data)

        load_profile = initialize_array("update", "sym_load", active_power_profile.shape)
        load_profile["id"] = active_power_profile.columns.to_numpy()
        load_profile["p_specified"] = active_power_profile.to_numpy()
        load_profile["q_specified"] = reactive_power_profile.to_numpy()

        update_data = {"sym_load": load_profile}

        with Timer("Batch Calculation using the linear method"):
            output_data = model.calculate_power_flow(
                update_data=update_data, calculation_method=CalculationMethod.linear
            )

        line_loading = output_data["line"]["loading"]
        max_idx = np.argmax(line_loading)
        max_line_load = line_loading[max_idx]
        max_line_load_id = output_data["line"]["id"][max_idx]
        timestamp_max = date_list[max_idx % len(date_list)]

        table.add_row([alt_id, max_line_load, max_line_load_id, timestamp_max])

        del new_data, output_data, model

        print(table)

    return alt_list

import pytest
import json
import pandas as pd
import copy
import numpy as np
from power_grid_model.utils import json_deserialize, json_serialize_to_file
from power_grid_model.validation.assertions import ValidationException

from power_system_simulation.graph_processor import (
    GraphCycleError,
    GraphNotFullyConnectedError,
    GraphProcessor,
    IDNotFoundError,
    IDNotUniqueError,
    InputLengthDoesNotMatchError,
)

from power_system_simulation.validity_check import TooManyTransformers, TooManySources, TooFewEVs, NotAllFeederIDsareValid, TransformerAndFeedersNotConnected, TimestampsDoNotMatchError, LoadIdsDoNotMatchError, validate_power_system_simulation
from pathlib import Path


#### TESTS FOR ASSIGNMENT 1 ####

# Common minimal valid input
VALID_VERTEX_IDS = [1, 2, 3]
VALID_EDGE_IDS = [10, 11]
VALID_EDGE_PAIRS = [(1, 2), (2, 3)]
VALID_EDGE_ENABLED = [True, True]
VALID_SOURCE_ID = 1


def test_successful_initialization():
    gp = GraphProcessor(
        vertex_ids=VALID_VERTEX_IDS,
        edge_ids=VALID_EDGE_IDS,
        edge_vertex_id_pairs=VALID_EDGE_PAIRS,
        edge_enabled=VALID_EDGE_ENABLED,
        source_vertex_id=VALID_SOURCE_ID,
    )
    assert isinstance(gp, GraphProcessor)


def test_duplicate_vertex_and_edge_ids_raises():
    vertex_ids = [1, 2, 10]  # 10 overlaps with edge_ids
    with pytest.raises(IDNotUniqueError):
        GraphProcessor(vertex_ids, VALID_EDGE_IDS, VALID_EDGE_PAIRS, VALID_EDGE_ENABLED, VALID_SOURCE_ID)


def test_mismatched_edge_list_length_raises():
    invalid_edge_pairs = [(1, 2)]  # Length 1 instead of 2
    with pytest.raises(InputLengthDoesNotMatchError):
        GraphProcessor(VALID_VERTEX_IDS, VALID_EDGE_IDS, invalid_edge_pairs, VALID_EDGE_ENABLED, VALID_SOURCE_ID)


def test_invalid_vertex_in_edge_pair_raises():
    invalid_edge_pairs = [(1, 99), (2, 3)]  # 99 is not a valid vertex
    with pytest.raises(IDNotFoundError):
        GraphProcessor(VALID_VERTEX_IDS, VALID_EDGE_IDS, invalid_edge_pairs, VALID_EDGE_ENABLED, VALID_SOURCE_ID)


def test_mismatched_enabled_flags_length_raises():
    invalid_edge_enabled = [True]  # Only one flag for two edges
    with pytest.raises(InputLengthDoesNotMatchError):
        GraphProcessor(VALID_VERTEX_IDS, VALID_EDGE_IDS, VALID_EDGE_PAIRS, invalid_edge_enabled, VALID_SOURCE_ID)


def test_invalid_source_vertex_raises():
    with pytest.raises(IDNotFoundError):
        GraphProcessor(VALID_VERTEX_IDS, VALID_EDGE_IDS, VALID_EDGE_PAIRS, VALID_EDGE_ENABLED, source_vertex_id=99)


def test_graph_not_connected_raises():
    vertex_ids = [1, 2, 3]
    edge_ids = [10]  # Only one edge connecting 1 and 2
    edge_pairs = [(1, 2)]
    edge_enabled = [True]
    with pytest.raises(GraphNotFullyConnectedError):
        GraphProcessor(vertex_ids, edge_ids, edge_pairs, edge_enabled, source_vertex_id=1)


def test_graph_with_cycle_raises():
    # Triangle creates a cycle
    vertex_ids = [1, 2, 3]
    edge_ids = [10, 11, 12]
    edge_pairs = [(1, 2), (2, 3), (3, 1)]
    edge_enabled = [True, True, True]
    with pytest.raises(GraphCycleError):
        GraphProcessor(vertex_ids, edge_ids, edge_pairs, edge_enabled, source_vertex_id=1)



#### TEST FOR ASSIGNMENT 3 ####

DATA_PATH = Path(__file__).parent / "stefan_data"

PATH_INPUT_NETWORK_DATA = DATA_PATH / "input_network_data.json"
PATH_META_DATA = DATA_PATH / "meta_data.json"
PATH_EV_ACTIVE_POWER_PROFILE = DATA_PATH / "ev_active_power_profile.parquet"
PATH_ACTIVE_POWER_PROFILE = DATA_PATH / "active_power_profile.parquet"
PATH_REACTIVE_POWER_PROFILE = DATA_PATH / "reactive_power_profile.parquet"

ev_power_profile = pd.read_parquet(str(PATH_EV_ACTIVE_POWER_PROFILE))
active_power_profile = pd.read_parquet(str(PATH_ACTIVE_POWER_PROFILE))
reactive_power_profile = pd.read_parquet(str(PATH_REACTIVE_POWER_PROFILE))

with open(str(PATH_META_DATA), "r", encoding="utf-8") as fp:
            meta_data = json.load(fp)

with open(str(PATH_INPUT_NETWORK_DATA), "r", encoding="utf-8") as fp:
    input_data = json_deserialize(fp.read())

def test_too_many_sources():
    meta_data_copy = copy.deepcopy(meta_data)
    meta_data_copy["source"] = [10, 30]
    with open(DATA_PATH / "meta_data_copy.json", "w", encoding="utf-8") as f:
        json.dump(meta_data_copy, f, indent=2)
    with open(DATA_PATH / "meta_data_copy.json") as f:
        print(json.load(f))
    with pytest.raises(TooManySources):
        ValidatePowerSystemSimulation(str(PATH_INPUT_NETWORK_DATA), str(DATA_PATH / "meta_data_copy.json"), str(PATH_EV_ACTIVE_POWER_PROFILE), str(PATH_ACTIVE_POWER_PROFILE), str(PATH_REACTIVE_POWER_PROFILE))

def test_too_many_transformers():
    meta_data_copy = copy.deepcopy(meta_data)
    meta_data_copy["transformer"] = [11, 30]
    with open(DATA_PATH / "meta_data_copy.json", "w", encoding="utf-8") as f:
        json.dump(meta_data_copy, f, indent=2)
    with pytest.raises(TooManyTransformers):
        ValidatePowerSystemSimulation(str(PATH_INPUT_NETWORK_DATA), str(DATA_PATH / "meta_data_copy.json"), str(PATH_EV_ACTIVE_POWER_PROFILE), str(PATH_ACTIVE_POWER_PROFILE), str(PATH_REACTIVE_POWER_PROFILE))

def test_feeder_ids_not_valid():
    meta_data_copy = copy.deepcopy(meta_data)
    meta_data_copy["lv_feeders"] = [16, 20, 30]
    with open(DATA_PATH / "meta_data_copy.json", "w", encoding="utf-8") as f:
        json.dump(meta_data_copy, f, indent=2)
    with pytest.raises(NotAllFeederIDsareValid):
        ValidatePowerSystemSimulation(str(PATH_INPUT_NETWORK_DATA), str(DATA_PATH / "meta_data_copy.json"), str(PATH_EV_ACTIVE_POWER_PROFILE), str(PATH_ACTIVE_POWER_PROFILE), str(PATH_REACTIVE_POWER_PROFILE))

def test_transformer_feeder_not_connected():
    input_data_copy = copy.deepcopy(input_data)
    input_data_copy["line"][4]["from_node"] = 2
    json_serialize_to_file(DATA_PATH / "input_data_copy.json", input_data_copy)
    with pytest.raises(TransformerAndFeedersNotConnected):
        ValidatePowerSystemSimulation(str(DATA_PATH / "input_data_copy.json"), str(PATH_META_DATA), str(PATH_EV_ACTIVE_POWER_PROFILE), str(PATH_ACTIVE_POWER_PROFILE), str(PATH_REACTIVE_POWER_PROFILE))

def test_too_few_ev():
    ev_power_profile_copy = ev_power_profile.copy(deep=True)
    ev_power_profile_copy = ev_power_profile_copy.drop(3, axis=1)
    ev_power_profile_copy.to_parquet(DATA_PATH / "ev_power_profile_copy.parquet", engine="pyarrow")
    with pytest.raises(TooFewEVs):
        ValidatePowerSystemSimulation(str(PATH_INPUT_NETWORK_DATA), str(PATH_META_DATA), str(DATA_PATH / "ev_power_profile_copy.parquet"), str(PATH_ACTIVE_POWER_PROFILE), str(PATH_REACTIVE_POWER_PROFILE))

def test_validation_error():
    input_data_copy = copy.deepcopy(input_data)
    input_data_copy["line"][0]["from_node"] = 2
    json_serialize_to_file(DATA_PATH / "input_data_copy.json", input_data_copy)
    with pytest.raises(ValidationException):
        ValidatePowerSystemSimulation(str(DATA_PATH / "input_data_copy.json"), str(PATH_META_DATA), str(PATH_EV_ACTIVE_POWER_PROFILE), str(PATH_ACTIVE_POWER_PROFILE), str(PATH_REACTIVE_POWER_PROFILE))

def test_graph_unconnected():
    input_data_copy = copy.deepcopy(input_data)
    input_data_copy["line"][7]["to_status"] = 0
    json_serialize_to_file(DATA_PATH / "input_data_copy.json", input_data_copy)
    with pytest.raises(GraphNotFullyConnectedError):
        ValidatePowerSystemSimulation(str(DATA_PATH / "input_data_copy.json"), str(PATH_META_DATA), str(PATH_EV_ACTIVE_POWER_PROFILE), str(PATH_ACTIVE_POWER_PROFILE), str(PATH_REACTIVE_POWER_PROFILE))

def test_timestamps():
    ev_power_profile_copy = ev_power_profile.copy(deep=True)
    new_timestamp = ev_power_profile_copy.index.tolist()
    new_timestamp[0] = pd.Timestamp("2025-01-01 00:10:00")
    ev_power_profile_copy.index = new_timestamp
    ev_power_profile_copy.to_parquet(DATA_PATH / "ev_power_profile_copy.parquet", engine="pyarrow")
    with pytest.raises(TimestampsDoNotMatchError):
        ValidatePowerSystemSimulation(str(PATH_INPUT_NETWORK_DATA), str(PATH_META_DATA), str(DATA_PATH / "ev_power_profile_copy.parquet"), str(PATH_ACTIVE_POWER_PROFILE), str(PATH_REACTIVE_POWER_PROFILE))

# Nu merge inca
# def test_active_reactive_IDs():
#     active_power_profile_copy = active_power_profile.copy(deep=True)
#     active_power_profile_copy.rename(columns={3: 4})
#     active_power_profile_copy.to_parquet(DATA_PATH / "active_power_profile_copy.parquet", engine="pyarrow")
#     with pytest.raises(LoadIdsDoNotMatchError):
#         ValidatePowerSystemSimulation(str(PATH_INPUT_NETWORK_DATA), str(PATH_META_DATA), str(PATH_EV_ACTIVE_POWER_PROFILE), str(DATA_PATH / "active_power_profile.parquet"), str(PATH_REACTIVE_POWER_PROFILE))

# def test_graph_cycle():
#     input_data_copy = copy.deepcopy(input_data)
#     input_data_copy["line"][8]["to_status"] = 1
#     json_serialize_to_file(DATA_PATH / "input_data_copy.json", input_data_copy)
#     with pytest.raises(GraphCycleError):
#         ValidatePowerSystemSimulation(str(DATA_PATH / "input_data_copy.json"), str(PATH_META_DATA), str(PATH_EV_ACTIVE_POWER_PROFILE))
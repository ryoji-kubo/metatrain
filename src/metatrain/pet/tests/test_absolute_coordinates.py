import copy

import pytest
import torch
from metatomic.torch import ModelOutput, NeighborListOptions, System

from metatrain.pet import PET, checkpoints
from metatrain.pet.modules.structures import systems_to_batch
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def _dataset_info() -> DatasetInfo:
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def test_absolute_neighbor_image_positions_include_periodic_cell_shift():
    options = NeighborListOptions(cutoff=1.1, full_list=True, strict=True)
    system = System(
        types=torch.tensor([1, 8]),
        positions=torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.2, 0.3]]),
        cell=torch.diag(torch.tensor([1.0, 2.0, 3.0])),
        pbc=torch.tensor([True, True, True]),
    )
    system = get_system_with_neighbor_lists(system, [options])

    species_to_species_index = torch.full((9,), -1, dtype=torch.long)
    species_to_species_index[1] = 0
    species_to_species_index[8] = 1
    batch = systems_to_batch(
        [system],
        options,
        [1, 8],
        species_to_species_index,
        cutoff_function="Bump",
        cutoff_width=0.2,
    )

    centers = batch[11]
    neighbors = batch[12]
    nef_to_edges_neighbor = batch[13]
    cell_shifts = batch[14]
    node_positions = batch[15]
    neighbor_image_positions = batch[16]

    torch.testing.assert_close(node_positions, system.positions)
    actual = neighbor_image_positions[centers, nef_to_edges_neighbor]
    expected = (
        system.positions.index_select(0, neighbors)
        + cell_shifts.to(system.cell.dtype) @ system.cell
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("featurizer_type", ["feedforward", "residual"])
def test_absolute_geometry_mode_runs_with_position_gradients(featurizer_type):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["geometry_mode"] = "absolute"
    hypers["featurizer_type"] = featurizer_type
    hypers["d_pet"] = 8
    hypers["d_head"] = 8
    hypers["d_node"] = 8
    hypers["d_feedforward"] = 8
    hypers["num_heads"] = 1
    hypers["num_attention_layers"] = 1
    hypers["num_gnn_layers"] = 1
    model = PET(hypers, _dataset_info())

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], requires_grad=True
    )
    system = System(
        types=torch.tensor([1, 8]),
        positions=positions,
        cell=torch.zeros((3, 3)),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    result = model([system], {"energy": ModelOutput(sample_kind="system")})

    energy = result["energy"].block().values.sum()
    position_gradients = torch.autograd.grad(energy, positions, create_graph=True)[0]
    assert torch.isfinite(energy)
    assert torch.isfinite(position_gradients).all()
    position_gradients.square().sum().backward()
    assert model.gnn_layers[0].edge_embedder.in_features == 3
    assert isinstance(model.gnn_layers[0].node_position_embedder, torch.nn.Linear)


def test_absolute_geometry_mode_handles_isolated_atom():
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["geometry_mode"] = "absolute"
    model = PET(hypers, _dataset_info())

    positions = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
    system = System(
        types=torch.tensor([1]),
        positions=positions,
        cell=torch.zeros((3, 3)),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    result = model([system], {"energy": ModelOutput(sample_kind="system")})

    energy = result["energy"].block().values.sum()
    position_gradients = torch.autograd.grad(energy, positions)[0]
    assert torch.isfinite(energy)
    assert torch.isfinite(position_gradients).all()


def test_absolute_geometry_mode_torchscript():
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["geometry_mode"] = "absolute"
    hypers["d_pet"] = 8
    hypers["d_head"] = 8
    hypers["d_node"] = 8
    hypers["d_feedforward"] = 8
    hypers["num_heads"] = 1
    hypers["num_attention_layers"] = 1
    hypers["num_gnn_layers"] = 1
    model = PET(hypers, _dataset_info())

    system = System(
        types=torch.tensor([1, 8]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        cell=torch.zeros((3, 3)),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    scripted = torch.jit.script(model)
    result = scripted([system], {"energy": ModelOutput(sample_kind="system")})
    assert torch.isfinite(result["energy"].block().values).all()


def test_v14_checkpoint_defaults_to_relative_geometry():
    checkpoint = {"model_data": {"model_hypers": {}}}
    checkpoints.model_update_v14_v15(checkpoint)
    assert checkpoint["model_data"]["model_hypers"]["geometry_mode"] == "relative"


def test_geometry_mode_default_and_validation():
    defaults = get_default_hypers("pet")
    assert defaults["model"]["geometry_mode"] == "relative"

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["geometry_mode"] = "invalid"
    with pytest.raises(ValueError, match="Unknown geometry mode"):
        PET(hypers, _dataset_info())

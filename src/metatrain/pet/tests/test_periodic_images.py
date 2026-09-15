import copy

import pytest
import torch
from metatomic.torch import ModelOutput, NeighborListOptions, System

from metatrain.pet import PET
from metatrain.pet.modules.nef import get_corresponding_edges
from metatrain.pet.modules.structures import systems_to_batch
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def _periodic_system(positions: torch.Tensor) -> System:
    return System(
        types=torch.full((positions.shape[0],), 6, dtype=torch.long),
        positions=positions,
        cell=torch.diag(torch.tensor([1.0, 0.0, 0.0])),
        pbc=torch.tensor([True, False, False]),
    )


def _batch(
    system: System,
    options: NeighborListOptions,
    mode: str,
    num_neighbors_adaptive=None,
):
    species_to_species_index = torch.full((7,), -1, dtype=torch.long)
    species_to_species_index[6] = 0
    return systems_to_batch(
        [system],
        options,
        [6],
        species_to_species_index,
        cutoff_function="Bump",
        cutoff_width=0.2,
        num_neighbors_adaptive=num_neighbors_adaptive,
        adaptive_cutoff_method="solver",
        neighbor_cell_shift_mode=mode,
    )


def _assert_reverse_closed(
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    cell_shifts: torch.Tensor,
) -> None:
    reverse = get_corresponding_edges(centers, neighbors, cell_shifts)
    edge_index = torch.arange(centers.shape[0])
    torch.testing.assert_close(reverse.index_select(0, reverse), edge_index)
    torch.testing.assert_close(centers, neighbors.index_select(0, reverse))
    torch.testing.assert_close(neighbors, centers.index_select(0, reverse))
    torch.testing.assert_close(cell_shifts, -cell_shifts.index_select(0, reverse))


def test_neighbor_cell_shift_modes_select_expected_edges():
    options = NeighborListOptions(cutoff=1.1, full_list=True, strict=True)
    system = _periodic_system(
        torch.tensor([[0.1, 0.0, 0.0], [0.4, 0.0, 0.0]])
    )
    system = get_system_with_neighbor_lists(system, [options])

    all_batch = _batch(system, options, "all")
    nearest_batch = _batch(system, options, "nearest")
    zero_batch = _batch(system, options, "zero")

    assert all_batch[11].shape[0] == 8
    assert nearest_batch[11].shape[0] == 6
    assert zero_batch[11].shape[0] == 2
    assert torch.all(zero_batch[14] == 0)

    nearest_centers = nearest_batch[11]
    nearest_neighbors = nearest_batch[12]
    nearest_distances = nearest_batch[3][nearest_batch[4]]
    distinct_atoms = nearest_centers != nearest_neighbors
    assert distinct_atoms.sum() == 2
    torch.testing.assert_close(
        nearest_distances[distinct_atoms],
        torch.full((2,), 0.3),
    )

    # The nearest self-image interaction is represented by its reverse-closed
    # +/- shift pair for each atom.
    self_edges = ~distinct_atoms
    assert self_edges.sum() == 4
    assert torch.all(nearest_batch[14][self_edges, 0].abs() == 1)
    _assert_reverse_closed(
        nearest_centers,
        nearest_neighbors,
        nearest_batch[14],
    )


@pytest.mark.parametrize("adaptive_cutoff_method", ["solver", "grid"])
def test_zero_mode_handles_only_cross_boundary_neighbors(adaptive_cutoff_method):
    options = NeighborListOptions(cutoff=0.3, full_list=True, strict=True)
    system = _periodic_system(
        torch.tensor([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]])
    )
    system = get_system_with_neighbor_lists(system, [options])

    species_to_species_index = torch.full((7,), -1, dtype=torch.long)
    species_to_species_index[6] = 0
    nearest_batch = systems_to_batch(
        [system],
        options,
        [6],
        species_to_species_index,
        cutoff_function="Bump",
        cutoff_width=0.1,
        num_neighbors_adaptive=2,
        adaptive_cutoff_method=adaptive_cutoff_method,
        neighbor_cell_shift_mode="nearest",
    )
    zero_batch = systems_to_batch(
        [system],
        options,
        [6],
        species_to_species_index,
        cutoff_function="Bump",
        cutoff_width=0.1,
        num_neighbors_adaptive=2,
        adaptive_cutoff_method=adaptive_cutoff_method,
        neighbor_cell_shift_mode="zero",
    )

    assert nearest_batch[11].shape[0] == 2
    assert torch.all(nearest_batch[14][:, 0].abs() == 1)
    _assert_reverse_closed(
        nearest_batch[11],
        nearest_batch[12],
        nearest_batch[14],
    )

    assert zero_batch[11].numel() == 0
    assert zero_batch[12].numel() == 0
    assert zero_batch[14].shape == (0, 3)
    assert zero_batch[4].shape == (2, 0)


@pytest.mark.parametrize(
    ("mode", "expected_neighbors"),
    [("nearest", 1.0), ("zero", 0.0)],
)
def test_neighbor_cell_shift_mode_runs_through_pet(mode, expected_neighbors):
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["cutoff"] = 0.3
    hypers["num_neighbors_adaptive"] = 2
    hypers["neighbor_cell_shift_mode"] = mode
    model = PET(hypers, dataset_info)

    system = _periodic_system(
        torch.tensor([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]])
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    result = model(
        [system],
        {
            "energy": ModelOutput(sample_kind="system"),
            "mtt::aux::cutoff_stats": ModelOutput(sample_kind="atom"),
        },
    )

    assert torch.isfinite(result["energy"].block().values).all()
    neighbor_counts = result["mtt::aux::cutoff_stats"].block().values[:, 1]
    torch.testing.assert_close(
        neighbor_counts,
        torch.full_like(neighbor_counts, expected_neighbors),
    )


def test_neighbor_cell_shift_mode_default_and_validation():
    defaults = get_default_hypers("pet")
    assert defaults["model"]["neighbor_cell_shift_mode"] == "all"

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["neighbor_cell_shift_mode"] = "invalid"
    with pytest.raises(ValueError, match="Unknown neighbor cell shift mode"):
        PET(hypers, dataset_info)

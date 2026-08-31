import copy
import math

import pytest
import torch
from metatomic.torch import ModelOutput, System

import metatrain.experimental.structure_transformer.model as structure_transformer_model
from metatrain.experimental.structure_transformer import StructureTransformerModel
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


def _model_hypers(**overrides):
    hypers = copy.deepcopy(
        get_default_hypers("experimental.structure_transformer")["model"]
    )
    hypers.update(
        {
            "max_num_elements": 16,
            "embed_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "encoder_hidden_dim": 8,
            "mlp_hidden_dim": 8,
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "residual_dropout": 0.0,
            "mlp_dropout": 0.0,
            "use_rotary_embeddings": False,
            "atom_ordering": "none",
            "edge_vector_head_cutoff": 3.0,
            "edge_vector_head_hidden_dim": 8,
            "edge_vector_head_num_radial_basis": 4,
        }
    )
    hypers.update(overrides)
    return hypers


def _dataset_info():
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={
            "energy": get_energy_target_info(
                "energy",
                {"quantity": "energy", "unit": "eV"},
            ),
            "forces": get_generic_target_info(
                "forces",
                {
                    "quantity": "force",
                    "unit": "eV/Angstrom",
                    "type": {"cartesian": {"rank": 1}},
                    "num_subtargets": 1,
                    "sample_kind": "atom",
                },
            ),
            "stress": get_generic_target_info(
                "stress",
                {
                    "quantity": "pressure",
                    "unit": "GPa",
                    "type": {"cartesian": {"rank": 2}},
                    "num_subtargets": 1,
                    "sample_kind": "system",
                },
            ),
        },
    )


def _system():
    return System(
        types=torch.tensor([6, 8]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.25, 0.0]],
            dtype=torch.float32,
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )


def _three_atom_system():
    return System(
        types=torch.tensor([6, 8, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )


def _three_atom_adaptive_system():
    return System(
        types=torch.tensor([6, 8, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )


def test_default_does_not_request_neighbor_lists():
    model = StructureTransformerModel(_model_hypers(), _dataset_info())

    assert model.requested_neighbor_lists() == []


def test_edge_vector_options_are_inert_when_disabled():
    model = StructureTransformerModel(
        _model_hypers(
            edge_vector_head=False,
            edge_vector_head_cutoff=0.0,
            edge_vector_head_hidden_dim=0,
            edge_vector_head_num_radial_basis=0,
        ),
        _dataset_info(),
    )

    assert model.requested_neighbor_lists() == []


def test_edge_vector_head_requests_neighbor_list():
    hypers = _model_hypers(edge_vector_head=True)
    model = StructureTransformerModel(hypers, _dataset_info())

    requested = model.requested_neighbor_lists()

    assert len(requested) == 1
    assert requested[0].cutoff == hypers["edge_vector_head_cutoff"]
    assert requested[0].full_list


def test_graph_attention_requests_neighbor_list_without_edge_head():
    hypers = _model_hypers(
        graph_attention="smooth_cutoff",
        graph_attention_cutoff=2.5,
        graph_attention_cutoff_width=0.5,
        graph_attention_bias_strength=1.0,
    )
    model = StructureTransformerModel(hypers, _dataset_info())

    requested = model.requested_neighbor_lists()

    assert len(requested) == 1
    assert requested[0].cutoff == hypers["graph_attention_cutoff"]
    assert requested[0].full_list


def test_zero_strength_graph_attention_does_not_request_neighbor_list():
    model = StructureTransformerModel(
        _model_hypers(
            graph_attention="smooth_cutoff",
            graph_attention_bias_strength=0.0,
        ),
        _dataset_info(),
    )

    assert model.requested_neighbor_lists() == []


def test_graph_attention_bias_uses_pet_log_cutoff_factors():
    hypers = _model_hypers(
        graph_attention="smooth_cutoff",
        graph_attention_cutoff=2.5,
        graph_attention_cutoff_width=0.5,
        graph_attention_bias_strength=0.25,
        graph_attention_epsilon=1.0e-12,
    )
    model = StructureTransformerModel(hypers, _dataset_info())
    system = get_system_with_neighbor_lists(
        _three_atom_system(), model.requested_neighbor_lists()
    )

    bias = model._systems_to_graph_attention_bias([system])

    assert bias.shape == (1, 3, 3)
    torch.testing.assert_close(torch.diagonal(bias[0]), torch.zeros(3))
    torch.testing.assert_close(
        bias[0, 0, 1], torch.tensor(0.0), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        bias[0, 1, 0], torch.tensor(0.0), atol=1e-6, rtol=1e-6
    )
    expected_missing_edge_bias = 0.25 * math.log(hypers["graph_attention_epsilon"])
    torch.testing.assert_close(
        bias[0, 0, 2],
        torch.tensor(expected_missing_edge_bias),
        atol=1e-6,
        rtol=1e-6,
    )


def test_graph_attention_adaptive_cutoff_uses_pet_pair_cutoffs(monkeypatch):
    calls = {}

    def fake_grid_adaptive_cutoffs(
        centers,
        edge_distances,
        num_neighbors_adaptive,
        num_nodes,
        max_cutoff,
        min_cutoff=0.5,
        cutoff_width=1.0,
        probe_spacing=None,
        weight_width=None,
    ):
        calls["centers"] = centers
        calls["edge_distances"] = edge_distances
        calls["num_neighbors_adaptive"] = num_neighbors_adaptive
        calls["num_nodes"] = num_nodes
        calls["max_cutoff"] = max_cutoff
        calls["cutoff_width"] = cutoff_width
        return edge_distances.new_full((num_nodes,), 1.5)

    monkeypatch.setattr(
        structure_transformer_model,
        "get_adaptive_cutoffs_grid",
        fake_grid_adaptive_cutoffs,
    )
    hypers = _model_hypers(
        graph_attention="smooth_cutoff",
        graph_attention_cutoff=3.0,
        graph_attention_num_neighbors_adaptive=1,
        graph_attention_adaptive_cutoff_method="grid",
        graph_attention_cutoff_width=0.5,
        graph_attention_bias_strength=1.0,
        graph_attention_epsilon=1.0e-12,
    )
    model = StructureTransformerModel(hypers, _dataset_info())
    system = get_system_with_neighbor_lists(
        _three_atom_adaptive_system(), model.requested_neighbor_lists()
    )

    bias = model._systems_to_graph_attention_bias([system])

    assert calls["num_neighbors_adaptive"] == 1.0
    assert calls["num_nodes"] == 3
    assert calls["max_cutoff"] == 3.0
    assert calls["cutoff_width"] == 0.5
    torch.testing.assert_close(
        bias[0, 0, 1], torch.tensor(0.0), atol=1e-6, rtol=1e-6
    )
    expected_missing_edge_bias = math.log(hypers["graph_attention_epsilon"])
    torch.testing.assert_close(
        bias[0, 0, 2],
        torch.tensor(expected_missing_edge_bias),
        atol=1e-6,
        rtol=1e-6,
    )


def test_graph_attention_forward_without_edge_head():
    model = StructureTransformerModel(
        _model_hypers(
            graph_attention="smooth_cutoff",
            graph_attention_cutoff=3.0,
            graph_attention_cutoff_width=0.5,
            graph_attention_bias_strength=1.0,
        ),
        _dataset_info(),
    )
    system = get_system_with_neighbor_lists(_system(), model.requested_neighbor_lists())

    predictions = model(
        [system],
        {
            "energy": ModelOutput(sample_kind="system"),
            "forces": ModelOutput(sample_kind="atom"),
            "stress": ModelOutput(sample_kind="system"),
        },
    )

    assert torch.isfinite(predictions["energy"].block().values).all()
    assert torch.isfinite(predictions["forces"].block().values).all()
    assert torch.isfinite(predictions["stress"].block().values).all()


def test_edge_vector_head_and_graph_attention_can_share_neighbor_list():
    hypers = _model_hypers(
        edge_vector_head=True,
        edge_vector_head_cutoff=3.0,
        graph_attention="smooth_cutoff",
        graph_attention_cutoff=3.0,
        graph_attention_cutoff_width=0.5,
        graph_attention_bias_strength=1.0,
    )
    model = StructureTransformerModel(hypers, _dataset_info())

    requested = model.requested_neighbor_lists()

    assert len(requested) == 1
    assert model.graph_requested_nl is model.edge_requested_nl

    system = get_system_with_neighbor_lists(_system(), requested)
    predictions = model(
        [system],
        {
            "energy": ModelOutput(sample_kind="system"),
            "forces": ModelOutput(sample_kind="atom"),
            "stress": ModelOutput(sample_kind="system"),
        },
    )

    assert torch.isfinite(predictions["energy"].block().values).all()
    assert torch.isfinite(predictions["forces"].block().values).all()
    assert torch.isfinite(predictions["stress"].block().values).all()


@pytest.mark.parametrize("replace_direct", [False, True])
def test_edge_vector_head_forward_forces_and_stress(replace_direct):
    model = StructureTransformerModel(
        _model_hypers(
            edge_vector_head=True,
            edge_vector_head_replace_direct=replace_direct,
        ),
        _dataset_info(),
    )
    system = get_system_with_neighbor_lists(_system(), model.requested_neighbor_lists())

    predictions = model(
        [system],
        {
            "energy": ModelOutput(sample_kind="system"),
            "forces": ModelOutput(sample_kind="atom"),
            "stress": ModelOutput(sample_kind="system"),
        },
    )

    energy = predictions["energy"].block().values
    forces = predictions["forces"].block().values
    stress = predictions["stress"].block().values

    assert energy.shape == (1, 1)
    assert forces.shape == (2, 3, 1)
    assert stress.shape == (1, 3, 3, 1)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()
    assert torch.isfinite(stress).all()

import copy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.experimental.structure_transformer import StructureTransformerModel
from metatrain.experimental.structure_transformer.modules.transformer import (
    StructureTransformer,
    TransformerData,
)
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info


_ATOMIC_NUMBERS = torch.tensor([6, 8, 1], dtype=torch.long)
_POSITIONS = torch.tensor(
    [
        [0.10, 0.20, 0.30],
        [0.70, 0.35, 0.90],
        [0.40, 0.80, 0.15],
    ],
    dtype=torch.float32,
)
_CELL = torch.eye(3, dtype=torch.float32).unsqueeze(0)


def _v37_kwargs(**overrides):
    kwargs = {
        "max_num_elements": 16,
        "embed_dim": 24,
        "num_heads": 4,
        "num_layers": 2,
        "encoder_hidden_dim": 32,
        "mlp_hidden_dim": 32,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "residual_dropout": 0.0,
        "mlp_dropout": 0.0,
        "position_representation": "fractional",
        "coordinate_encoding": "v37_torus_relative",
        "coord_num_harmonics": 3,
        "coord_encoder_chunk_size": None,
        "coord_encoder_use_checkpoint": False,
        "use_rotary_embeddings": False,
        "use_periodic_rope": True,
        "atom_ordering": "none",
        "center_positions": False,
        "fractional_wrap_eps": 1.0e-6,
        "atom_embedding_type": "scalar",
        "atomic_number_scale": 118.0,
        "atom_scalar_embedding_scale": 1000.0,
        "include_cell_energy": True,
        "include_cell_stress": True,
        "force_readout_type": "mlp",
        "stress_readout_type": "mlp",
        "pair_readout_num_heads": None,
        "pair_readout_hidden_dim": None,
        "pair_readout_num_layers": 1,
        "pair_readout_dropout": None,
        "pair_readout_chunk_size": None,
        "pair_readout_use_checkpoint": False,
        "pair_readout_include_pair_geometry": False,
        "pair_readout_exclude_self": True,
        "regress_forces": True,
        "regress_stress": True,
        "avg_num_nodes": 1.0,
    }
    kwargs.update(overrides)
    return kwargs


def _make_model():
    torch.manual_seed(0)
    model = StructureTransformer(**_v37_kwargs())
    model.eval()
    return model


def _make_pair_readout_model(include_pair_geometry=False, **overrides):
    torch.manual_seed(0)
    model = StructureTransformer(
        **_v37_kwargs(
            force_readout_type="pair_cross_attention",
            stress_readout_type="pair_cross_attention",
            pair_readout_chunk_size=1,
            pair_readout_include_pair_geometry=include_pair_geometry,
            **overrides,
        )
    )
    model.eval()
    return model


def _data(positions=_POSITIONS, atomic_numbers=_ATOMIC_NUMBERS):
    return TransformerData(
        atomic_numbers=atomic_numbers,
        pos=positions,
        batch=torch.zeros(positions.shape[0], dtype=torch.long),
        cell=_CELL,
    )


def _predict(model, data):
    with torch.no_grad():
        return model(data)


def _assert_system_outputs_close(reference, actual):
    torch.testing.assert_close(
        actual["energy"], reference["energy"], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        actual["stress"], reference["stress"], atol=1e-5, rtol=1e-5
    )


def test_v37_backbone_is_periodic_under_integer_fractional_shifts():
    model = _make_model()
    reference = _predict(model, _data())
    integer_shift = torch.tensor(
        [
            [1.0, 0.0, -2.0],
            [-3.0, 2.0, 1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=torch.float32,
    )

    shifted = _predict(model, _data(positions=_POSITIONS + integer_shift))

    _assert_system_outputs_close(reference, shifted)
    torch.testing.assert_close(
        shifted["forces"], reference["forces"], atol=1e-5, rtol=1e-5
    )


def test_v37_backbone_is_invariant_to_global_fractional_translation():
    model = _make_model()
    reference = _predict(model, _data())
    translation = torch.tensor([0.13, 0.27, -0.19], dtype=torch.float32)

    translated = _predict(model, _data(positions=_POSITIONS + translation))

    _assert_system_outputs_close(reference, translated)
    torch.testing.assert_close(
        translated["forces"], reference["forces"], atol=1e-5, rtol=1e-5
    )


def test_v37_backbone_is_permutation_equivariant_for_atom_outputs():
    model = _make_model()
    reference = _predict(model, _data())
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)

    permuted = _predict(
        model,
        _data(
            positions=_POSITIONS.index_select(0, permutation),
            atomic_numbers=_ATOMIC_NUMBERS.index_select(0, permutation),
        ),
    )

    _assert_system_outputs_close(reference, permuted)
    torch.testing.assert_close(
        permuted["forces"],
        reference["forces"].index_select(0, permutation),
        atol=1e-5,
        rtol=1e-5,
    )


def test_v37_chunked_coordinate_encoder_matches_unchunked():
    torch.manual_seed(0)
    full_model = StructureTransformer(**_v37_kwargs(coord_encoder_chunk_size=None))
    chunked_model = StructureTransformer(**_v37_kwargs(coord_encoder_chunk_size=1))
    chunked_model.load_state_dict(full_model.state_dict())
    full_model.eval()
    chunked_model.eval()

    reference = _predict(full_model, _data())
    chunked = _predict(chunked_model, _data())

    _assert_system_outputs_close(reference, chunked)
    torch.testing.assert_close(
        chunked["forces"], reference["forces"], atol=1e-5, rtol=1e-5
    )


def test_v37_checkpointed_coordinate_chunks_backpropagate():
    torch.manual_seed(0)
    model = StructureTransformer(
        **_v37_kwargs(coord_encoder_chunk_size=1, coord_encoder_use_checkpoint=True)
    )
    model.train()

    output = model(_data())
    loss = output["energy"].sum()
    loss = loss + output["forces"].square().sum()
    loss = loss + output["stress"].sum()
    loss.backward()

    grad = model.position_encoder.pair_mlp[0].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0.0


def test_v37_pair_cross_attention_readout_chunked_matches_unchunked():
    torch.manual_seed(0)
    full_model = StructureTransformer(
        **_v37_kwargs(
            force_readout_type="pair_cross_attention",
            stress_readout_type="pair_cross_attention",
            pair_readout_chunk_size=None,
        )
    )
    chunked_model = StructureTransformer(
        **_v37_kwargs(
            force_readout_type="pair_cross_attention",
            stress_readout_type="pair_cross_attention",
            pair_readout_chunk_size=1,
        )
    )
    chunked_model.load_state_dict(full_model.state_dict())
    full_model.eval()
    chunked_model.eval()

    reference = _predict(full_model, _data())
    chunked = _predict(chunked_model, _data())

    _assert_system_outputs_close(reference, chunked)
    torch.testing.assert_close(
        chunked["forces"], reference["forces"], atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("include_pair_geometry", [False, True])
def test_v37_pair_cross_attention_readout_preserves_symmetries(include_pair_geometry):
    model = _make_pair_readout_model(include_pair_geometry=include_pair_geometry)
    reference = _predict(model, _data())

    integer_shift = torch.tensor(
        [
            [1.0, 0.0, -2.0],
            [-3.0, 2.0, 1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=torch.float32,
    )
    shifted = _predict(model, _data(positions=_POSITIONS + integer_shift))
    _assert_system_outputs_close(reference, shifted)
    torch.testing.assert_close(
        shifted["forces"], reference["forces"], atol=1e-5, rtol=1e-5
    )

    translation = torch.tensor([0.13, 0.27, -0.19], dtype=torch.float32)
    translated = _predict(model, _data(positions=_POSITIONS + translation))
    _assert_system_outputs_close(reference, translated)
    torch.testing.assert_close(
        translated["forces"], reference["forces"], atol=1e-5, rtol=1e-5
    )

    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    permuted = _predict(
        model,
        _data(
            positions=_POSITIONS.index_select(0, permutation),
            atomic_numbers=_ATOMIC_NUMBERS.index_select(0, permutation),
        ),
    )
    _assert_system_outputs_close(reference, permuted)
    torch.testing.assert_close(
        permuted["forces"],
        reference["forces"].index_select(0, permutation),
        atol=1e-5,
        rtol=1e-5,
    )


def test_v37_pair_cross_attention_readout_backpropagates_with_checkpointing():
    model = _make_pair_readout_model(pair_readout_use_checkpoint=True)
    model.train()

    output = model(_data())
    loss = output["energy"].sum()
    loss = loss + output["forces"].square().sum()
    loss = loss + output["stress"].square().sum()
    loss.backward()

    force_grad = model.force_pair_readout.layers[0].q_proj.weight.grad
    stress_grad = model.stress_pair_readout.layers[0].q_proj.weight.grad
    assert force_grad is not None
    assert stress_grad is not None
    assert torch.isfinite(force_grad).all()
    assert torch.isfinite(stress_grad).all()
    assert force_grad.abs().sum() > 0.0
    assert stress_grad.abs().sum() > 0.0


def test_v37_hyperparameters_are_exposed_to_config_and_wrapper():
    defaults = get_default_hypers("experimental.structure_transformer")
    hypers = copy.deepcopy(defaults["model"])

    assert defaults["training"]["use_data_augmentation"] is True
    assert hypers["coordinate_encoding"] == "absolute_mlp"
    assert hypers["coord_num_harmonics"] == 4
    assert hypers["coord_encoder_chunk_size"] is None
    assert hypers["coord_encoder_use_checkpoint"] is False
    assert hypers["use_periodic_rope"] is False
    assert hypers["atom_embedding_type"] == "embedding"
    assert hypers["force_readout_type"] == "mlp"
    assert hypers["stress_readout_type"] == "mlp"
    assert hypers["pair_readout_chunk_size"] is None
    assert hypers["pair_readout_include_pair_geometry"] is False

    repo_root = Path(__file__).parents[5]
    v37_config = OmegaConf.load(
        repo_root / "options-structure-transformer-mptrj-salex-direct-160k-v37.yaml"
    )
    assert v37_config.architecture.training.use_data_augmentation is True
    assert v37_config.architecture.model.coord_encoder_chunk_size == 128
    assert v37_config.architecture.model.coord_encoder_use_checkpoint is True
    assert v37_config.architecture.model.force_readout_type == "pair_cross_attention"
    assert v37_config.architecture.model.stress_readout_type == "pair_cross_attention"
    assert v37_config.architecture.model.pair_readout_chunk_size == 64
    assert v37_config.architecture.model.pair_readout_include_pair_geometry is False

    hypers.update(
        _v37_kwargs(
            num_layers=1,
            force_readout_type="pair_cross_attention",
            stress_readout_type="pair_cross_attention",
            pair_readout_chunk_size=1,
        )
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={
            "energy": get_energy_target_info(
                "energy",
                {"quantity": "energy", "unit": "eV"},
            ),
        },
    )

    model = StructureTransformerModel(hypers, dataset_info)

    assert model.transformer.coordinate_encoding == "v37_torus_relative"
    assert model.transformer.coord_num_harmonics == 3
    assert model.transformer.use_periodic_rope
    assert model.transformer.atom_embedding_type == "scalar"
    assert model.transformer.force_readout_type == "pair_cross_attention"
    assert model.transformer.stress_readout_type == "pair_cross_attention"
    assert model.transformer.pair_readout_chunk_size == 1


def test_v37_rejects_index_ordering_and_double_rope():
    with pytest.raises(ValueError, match="atom_ordering='none'"):
        StructureTransformer(**_v37_kwargs(atom_ordering="position_ids"))

    with pytest.raises(ValueError, match="mutually exclusive"):
        StructureTransformer(
            **_v37_kwargs(use_rotary_embeddings=True, use_periodic_rope=True)
        )

    with pytest.raises(ValueError, match="coord_encoder_chunk_size"):
        StructureTransformer(**_v37_kwargs(coord_encoder_chunk_size=0))

    with pytest.raises(ValueError, match="force_readout_type"):
        StructureTransformer(**_v37_kwargs(force_readout_type="attention"))

    with pytest.raises(ValueError, match="pair_readout_chunk_size"):
        StructureTransformer(
            **_v37_kwargs(
                force_readout_type="pair_cross_attention",
                pair_readout_chunk_size=0,
            )
        )

    with pytest.raises(ValueError, match="pair_readout_num_heads"):
        StructureTransformer(
            **_v37_kwargs(
                force_readout_type="pair_cross_attention",
                pair_readout_num_heads=5,
            )
        )

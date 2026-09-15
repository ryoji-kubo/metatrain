"""Behavioral and reference-parity checks for the independent GenT core."""

import ast
import importlib.util
from pathlib import Path

import pytest
import torch

from gent_core import GeneralizedTransformer, GenTBlock, GenTData
from gent_core.gent_layers import precompute_complex_pe


@pytest.mark.parametrize("use_rope", [False, True])
def test_reference_parity(use_rope):
    """Keep the supplied node/edge equations, including endpoint attention order."""
    path = (
        Path(__file__).parents[2] / "structure_transformer_core/gent_layers_original.py"
    )
    spec = importlib.util.spec_from_file_location("gent_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(0)
    original = reference.GenT_RoPE_Block(8, 8, 8, 8, 16, 2, 0.0, 0.0).eval()
    adapted = GenTBlock(8, 8, 8, 8, 16, 2, 0.0, 0.0).eval()
    adapted.load_state_dict(original.state_dict())
    x, e = torch.randn(2, 3, 8), torch.randn(2, 3, 3, 8)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    x = x * mask.unsqueeze(-1)
    e = e * (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
    pe = (
        precompute_complex_pe(3, 4)
        if use_rope
        else torch.ones(3, 2, dtype=torch.complex64)
    )
    expected = original(x, e, pe, mask)
    actual = adapted(x, e, pe if use_rope else None, mask)
    # Reference padding residuals are intentionally fixed in the adapted block.
    torch.testing.assert_close(actual[0][mask], expected[0][mask], atol=1e-6, rtol=1e-5)
    edge_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    torch.testing.assert_close(
        actual[1][edge_mask], expected[1][edge_mask], atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_rows_and_backward(dtype):
    torch.manual_seed(1)
    block = GenTBlock(8, 8, 8, 8, 16, 2, 0.0, 0.0).to(dtype)
    x = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    e = torch.randn(2, 3, 3, 8, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    out_x, out_e = block(x, e, None, mask)
    assert out_x.dtype == dtype and out_e.dtype == dtype
    assert torch.count_nonzero(out_x[~mask]) == 0
    assert torch.count_nonzero(out_e[~(mask.unsqueeze(1) & mask.unsqueeze(2))]) == 0
    (out_x.square().sum() + out_e.square().sum()).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(e.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def _data():
    return GenTData(
        atomic_numbers=torch.tensor([1, 8, 6]),
        num_atoms=torch.tensor([2, 1]),
        edge_vectors=torch.tensor(
            [[1.0, 0.2, 0.0], [-1.0, -0.2, 0.0]], dtype=torch.float64
        ),
        edge_centers=torch.tensor([0, 1]),
        edge_neighbors=torch.tensor([1, 0]),
        cells=torch.eye(3, dtype=torch.float64).repeat(2, 1, 1) * 4,
    )


def _model():
    torch.manual_seed(2)
    return (
        GeneralizedTransformer(
            embed_dim=8,
            num_heads=2,
            num_layers=2,
            num_radial_basis=4,
        )
        .double()
        .eval()
    )


def test_core_script_and_save(tmp_path):
    model = _model()
    scripted = torch.jit.script(model)
    path = tmp_path / "gent-core.pt"
    torch.jit.save(scripted, path)
    loaded = torch.jit.load(path)
    for name, values in model(_data()).items():
        torch.testing.assert_close(loaded(_data())[name], values)


def test_periodic_multiplicity_and_no_neighbor_gradients():
    model, data = _model(), _data()
    duplicated = data._replace(
        edge_vectors=data.edge_vectors.repeat(2, 1),
        edge_centers=data.edge_centers.repeat(2),
        edge_neighbors=data.edge_neighbors.repeat(2),
    )
    assert not torch.allclose(model(data)["energy"], model(duplicated)["energy"])
    empty = data._replace(
        edge_vectors=torch.empty(0, 3, dtype=torch.float64),
        edge_centers=torch.empty(0, dtype=torch.long),
        edge_neighbors=torch.empty(0, dtype=torch.long),
    )
    outputs = model(empty)
    assert torch.count_nonzero(outputs["forces"]) == 0
    assert torch.count_nonzero(outputs["stress"]) == 0
    sum(value.square().sum() for value in outputs.values()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_edge_gradients_against_finite_differences():
    model = _model()
    data = _data()._replace(edge_vectors=_data().edge_vectors.requires_grad_())
    assert torch.autograd.gradcheck(
        lambda edges: model(data._replace(edge_vectors=edges))["energy"],
        (data.edge_vectors,),
        fast_mode=True,
    )
    energy = model(data)["energy"].sum()
    gradient = torch.autograd.grad(energy, data.edge_vectors, create_graph=True)[0]
    gradient.square().sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_heads": 3},
        {"num_heads": 0},
        {"num_layers": 0},
        {"cutoff": 0},
        {"num_radial_basis": 1},
        {"attn_dropout": 1.0},
    ],
)
def test_invalid_hypers(kwargs):
    with pytest.raises(ValueError):
        GeneralizedTransformer(**kwargs)


def test_core_has_no_repository_imports():
    forbidden = {
        "metatrain",
        "metatensor",
        "metatomic",
        "fairchem",
        "structure_transformer_core",
        "torch_geometric",
    }
    for path in Path(__file__).parents[1].glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert (
                    not {alias.name.split(".")[0] for alias in node.names} & forbidden
                )
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden

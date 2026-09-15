import copy

import pytest
import torch
from metatensor.torch import Labels
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)

from metatrain.experimental.gent import GenTModel
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


def hypers(**overrides):
    values = copy.deepcopy(get_default_hypers("experimental.gent")["model"])
    values.update(
        embed_dim=8, num_heads=2, num_layers=2, num_radial_basis=4, cutoff=2.5
    )
    values.update(overrides)
    return values


def dataset_info():
    targets = {
        "energy": get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"})
    }
    for name, quantity, rank, kind, unit in [
        ("non_conservative_force", "force", 1, "atom", "eV/Angstrom"),
        ("non_conservative_stress", "pressure", 2, "system", "eV/Angstrom^3"),
    ]:
        targets[name] = get_generic_target_info(
            name,
            {
                "quantity": quantity,
                "unit": unit,
                "type": {"cartesian": {"rank": rank}},
                "sample_kind": kind,
                "num_subtargets": 1,
            },
        )
    return DatasetInfo(length_unit="Angstrom", atomic_types=[1, 6, 8], targets=targets)


def system(periodic=False, dtype=torch.float64):
    return System(
        types=torch.tensor([1, 8, 6]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.2, 0.1], [0.1, 1.1, 0.3]], dtype=dtype
        ),
        cell=torch.tensor(
            [[2.0, 0.0, 0.0], [0.3, 2.2, 0.0], [0.1, 0.2, 2.4]], dtype=dtype
        )
        if periodic
        else torch.zeros(3, 3, dtype=dtype),
        pbc=torch.tensor([periodic] * 3),
    )


def prepare(model, value):
    return get_system_with_neighbor_lists(value, model.requested_neighbor_lists())


def requests(info):
    return {
        name: ModelOutput(
            quantity=value.quantity, unit=value.unit, sample_kind=value.sample_kind
        )
        for name, value in info.targets.items()
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adapter_core_parity_backward_and_padding(dtype):
    torch.manual_seed(3)
    model = GenTModel(hypers(), dataset_info()).to(dtype)
    first = prepare(model, system(dtype=dtype))
    isolated = prepare(
        model,
        System(
            types=torch.tensor([1]),
            positions=torch.zeros(1, 3, dtype=dtype),
            cell=torch.zeros(3, 3, dtype=dtype),
            pbc=torch.tensor([False] * 3),
        ),
    )
    systems = [isolated, first]
    raw = model.transformer(model._systems_to_gent_data(systems))
    outputs = model(systems, requests(dataset_info()))
    for name, raw_name in [
        ("energy", "energy"),
        ("non_conservative_force", "forces"),
        ("non_conservative_stress", "stress"),
    ]:
        torch.testing.assert_close(
            outputs[name].block().values.flatten(), raw[raw_name].flatten()
        )
        alone = model([isolated], requests(dataset_info()))[name].block().values
        torch.testing.assert_close(outputs[name].block().values[:1], alone)
    sum(value.block().values.square().sum() for value in outputs.values()).backward()
    for parameter in model.transformer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("reflect", [False, True])
def test_permutation_translation_rotation_reflection(periodic, reflect):
    torch.manual_seed(4)
    model = GenTModel(hypers(), dataset_info()).double().eval()
    original = prepare(model, system(periodic))
    reference = model([original], requests(dataset_info()))
    permutation = torch.tensor([2, 0, 1])
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if reflect:
        rotation[:, 0] *= -1
    transformed = System(
        types=original.types[permutation],
        positions=original.positions[permutation] @ rotation + 1.7,
        cell=original.cell @ rotation,
        pbc=original.pbc,
    )
    actual = model([prepare(model, transformed)], requests(dataset_info()))
    torch.testing.assert_close(
        actual["energy"].block().values, reference["energy"].block().values
    )
    forces = reference["non_conservative_force"].block().values.squeeze(-1)
    torch.testing.assert_close(
        actual["non_conservative_force"].block().values.squeeze(-1),
        forces[permutation] @ rotation,
    )
    stress = reference["non_conservative_stress"].block().values.squeeze(-1)
    torch.testing.assert_close(
        actual["non_conservative_stress"].block().values.squeeze(-1),
        rotation.T @ stress @ rotation,
    )
    torch.testing.assert_close(
        forces.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(stress, stress.transpose(1, 2))


def test_periodic_rewrapping_and_self_images():
    model = GenTModel(hypers(), dataset_info()).double().eval()
    original = prepare(model, system(True))
    shifted = System(
        types=original.types,
        positions=original.positions.clone(),
        cell=original.cell,
        pbc=original.pbc,
    )
    shifted.positions[1] += original.cell[0] - 2 * original.cell[1]
    for name, expected in model([original], requests(dataset_info())).items():
        actual = model([prepare(model, shifted)], requests(dataset_info()))[name]
        torch.testing.assert_close(actual.block().values, expected.block().values)
    data = model._systems_to_gent_data([original])
    assert (data.edge_centers == data.edge_neighbors).any()
    assert len(data.edge_centers) > len(original) ** 2


def test_selected_atoms_and_per_atom_energy():
    model = GenTModel(hypers(), dataset_info()).double().eval()
    value = prepare(model, system())
    selected = Labels(["system", "atom"], torch.tensor([[0, 1]]))
    requested = requests(dataset_info())
    requested.pop("non_conservative_stress")
    requested["energy"].sample_kind = "atom"
    full = model([value], requested)
    subset = model([value], requested, selected)
    for name in requested:
        torch.testing.assert_close(
            subset[name].block().values, full[name].block().values[1:2]
        )
    requested["energy"].sample_kind = "system"
    torch.testing.assert_close(
        model([value], requested, selected)["energy"].block().values,
        subset["energy"].block().values,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checkpoint_restart_export(tmp_path, dtype):
    model = GenTModel(hypers(), dataset_info()).to(dtype).eval()
    if dtype == torch.float64:
        with torch.no_grad():
            next(model.transformer.parameters()).add_(1e-10)
    value = prepare(model, system(True, dtype))
    checkpoint = model.get_checkpoint()
    assert checkpoint["architecture_name"] == "experimental.gent"
    for context in ["restart", "export", "finetune"]:
        restored = GenTModel.load_checkpoint(checkpoint, context).to(dtype).eval()
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[key], tensor, rtol=0, atol=0
            )
        restored.restart(dataset_info())
        for name, expected in model([value], requests(dataset_info())).items():
            torch.testing.assert_close(
                restored([value], requests(dataset_info()))[name].block().values,
                expected.block().values,
            )
    exported = model.export()
    assert exported.capabilities().interaction_range == float("inf")
    path = str(tmp_path / "gent.pt")
    exported.save(path)
    loaded = load_atomistic_model(path)
    options = ModelEvaluationOptions(
        length_unit="Angstrom", outputs=requests(dataset_info())
    )
    result = loaded([value], options, check_consistency=True)
    for name, expected in model([value], requests(dataset_info())).items():
        torch.testing.assert_close(result[name].block().values, expected.block().values)


def test_architecture_configuration_and_disabled_heads():
    defaults = get_default_hypers("experimental.gent", base_precision=32)
    check_architecture_options("experimental.gent", defaults)
    with pytest.raises(ValueError, match="regress_forces"):
        GenTModel(hypers(regress_forces=False), dataset_info())
    with pytest.raises(ValueError, match="regress_stress"):
        GenTModel(hypers(regress_stress=False), dataset_info())


@pytest.mark.parametrize("isolated", [False, True])
def test_energy_position_and_strain_gradients(isolated):
    info = dataset_info()
    info.targets["energy"] = get_energy_target_info(
        "energy",
        {"quantity": "energy", "unit": "eV"},
        add_position_gradients=True,
        add_strain_gradients=True,
    )
    model = GenTModel(hypers(), info).double()
    value = system(True)
    if isolated:
        value = System(
            types=torch.tensor([1]),
            positions=torch.zeros(1, 3, dtype=torch.float64),
            cell=torch.eye(3, dtype=torch.float64) * 10,
            pbc=torch.tensor([True] * 3),
        )
    value = prepare(model, value)
    prediction = evaluate_model(
        model, [value], {"energy": info.targets["energy"]}, is_training=True
    )["energy"].block()
    loss = prediction.values.square().sum()
    for name in ["positions", "strain"]:
        gradient = prediction.gradient(name).values
        assert torch.isfinite(gradient).all()
        if isolated:
            assert torch.count_nonzero(gradient) == 0
        loss = loss + gradient.square().sum()
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_forward_backward(dtype):
    torch.manual_seed(6)
    model = GenTModel(hypers(), dataset_info()).to(dtype)
    value = prepare(model, system(True, dtype))
    reference = model([value], requests(dataset_info()))
    model = model.to("cuda")
    result = model([value.to(device="cuda")], requests(dataset_info()))
    for name in reference:
        torch.testing.assert_close(
            result[name].block().values.cpu(),
            reference[name].block().values,
            atol=2e-6,
            rtol=2e-5,
        )
    sum(tmap.block().values.square().sum() for tmap in result.values()).backward()
    for parameter in model.transformer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

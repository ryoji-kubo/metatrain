"""Exercise CLI discovery, PET training, checkpoint restart and final export."""

import os
import subprocess
import sys
from pathlib import Path

import ase
import ase.io
import numpy as np
import torch
from metatomic.torch import load_atomistic_model
from omegaconf import OmegaConf

from .test_model import hypers


def test_cli_training_and_restart(tmp_path):
    frames = []
    for index in range(6):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.8 + 0.04 * index, 0.2, 0.1],
                [0.1, 1.1, 0.3 + 0.03 * index],
            ]
        )
        atoms = ase.Atoms(
            numbers=[1, 6, 8], positions=positions, cell=np.eye(3) * 4, pbc=True
        )
        differences = positions[:, None, :] - positions[None, :, :]
        atoms.info["e"] = float(0.25 * (differences**2).sum() - 3.0)
        atoms.arrays["f"] = -differences.sum(axis=1)
        atoms.info["s"] = (
            0.5
            * np.einsum("ijk,ijl->kl", differences, differences)
            / atoms.get_volume()
        )
        frames.append(atoms)
    train_path, val_path = tmp_path / "train.xyz", tmp_path / "val.xyz"
    ase.io.write(train_path, frames[:4])
    ase.io.write(val_path, frames[4:])
    targets = {
        "energy": {
            "key": "e",
            "quantity": "energy",
            "unit": "eV",
            "forces": False,
            "stress": False,
        },
        "non_conservative_force": {
            "key": "f",
            "quantity": "force",
            "unit": "eV/Angstrom",
            "sample_kind": "atom",
            "type": {"cartesian": {"rank": 1}},
        },
        "non_conservative_stress": {
            "key": "s",
            "quantity": "pressure",
            "unit": "eV/Angstrom^3",
            "sample_kind": "system",
            "type": {"cartesian": {"rank": 2}},
        },
    }
    config = {
        "device": "cpu",
        "base_precision": 32,
        "seed": 5,
        "architecture": {
            "name": "experimental.gent",
            "model": hypers(),
            "training": {
                "num_epochs": 2,
                "num_workers": 0,
                "batch_size": 2,
                "checkpoint_interval": 1,
                "log_interval": 1,
                "scale_targets": True,
                "use_data_augmentation": True,
                "per_structure_targets": ["non_conservative_stress"],
                "loss": {name: {"type": "mse", "weight": 1.0} for name in targets},
            },
        },
        "training_set": {
            "systems": {"read_from": str(train_path), "length_unit": "Angstrom"},
            "targets": targets,
        },
        "validation_set": {
            "systems": {"read_from": str(val_path), "length_unit": "Angstrom"},
            "targets": targets,
        },
        "test_set": {"indices": []},
    }
    options_path = tmp_path / "options.yaml"
    OmegaConf.save(OmegaConf.create(config), options_path)
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    environment["WANDB_MODE"] = "disabled"
    environment["PYTHONPATH"] = (
        str(Path(__file__).parents[4]) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    for output_name, extra in [
        ("trained", []),
        (
            "restarted",
            [
                "--restart",
                str(tmp_path / "trained.ckpt"),
                "-r",
                "architecture.training.num_epochs=3",
            ],
        ),
    ]:
        command = [
            sys.executable,
            "-m",
            "metatrain",
            "train",
            str(options_path),
            "-o",
            str(tmp_path / (output_name + ".pt")),
            *extra,
        ]
        completed = subprocess.run(
            command,
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert load_atomistic_model(
            str(tmp_path / (output_name + ".pt"))
        ).capabilities().atomic_types == [1, 6, 8]
    checkpoint = torch.load(tmp_path / "restarted.ckpt", weights_only=False)
    assert checkpoint["architecture_name"] == "experimental.gent"
    assert checkpoint["epoch"] == 2
    assert checkpoint["optimizer_state_dict"]["state"]

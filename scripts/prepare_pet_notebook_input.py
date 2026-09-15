"""Export one real, fully prepared PET batch for notebooks/pet_isolated.ipynb.

Run from the repository root in the metatrain environment. This script performs
the data work once; the notebook only needs the resulting .pt file and PyTorch.
The existing checkpoint supplies dataset metadata and frozen normalization only.
No learned PET weights or optimizer state are exported.
"""

import argparse
import hashlib
import random
from pathlib import Path

import metatensor.torch as mts
import numpy as np
import torch
from omegaconf import OmegaConf

from metatrain.pet.model import PET
from metatrain.pet.modules.structures import systems_to_batch
from metatrain.utils.additive import get_remove_additive_transform
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.augmentation import RotationalAugmenter
from metatrain.utils.data import (
    CollateFn,
    MaxAtomDistributedBatchSampler,
    get_dataset,
    unpack_batch,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists_transform
from metatrain.utils.omegaconf import expand_dataset_config
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.scaler import get_remove_scale_transform
from metatrain.utils.transfer import batch_to


def main():
    """Prepare and save one batch, without training PET."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("options-pet-oam-l-modern-mptrj-salex-direct-epoch@50.yaml"),
    )
    parser.add_argument(
        "--statistics-checkpoint",
        type=Path,
        default=Path("pet-oam-l-modern-mptrj-salex-direct-ddp.ckpt"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("notebooks/pet_input.pt")
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    architecture = OmegaConf.to_container(
        OmegaConf.merge(get_default_hypers("pet"), config.architecture), resolve=True
    )
    assert config.architecture.name == "pet"
    assert config.base_precision == 32
    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)

    # This is a trusted local training checkpoint, mmap avoids reading its large
    # model/Adam tensors. Only metadata and frozen normalizer buffers are used.
    checkpoint = torch.load(
        args.statistics_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    dataset_info = checkpoint["model_data"]["dataset_info"]
    for key, value in OmegaConf.to_container(config.architecture.model).items():
        assert checkpoint["model_data"]["model_hypers"][key] == value, key
    model = PET(architecture["model"], dataset_info)
    state = checkpoint["model_state_dict"]
    composition = model.additive_models[0].to(dtype=torch.float64)
    composition.model.weights["energy"] = mts.load_buffer(
        state["additive_models.0.energy_composition_buffer"]
    )
    composition.weights_to(device="cpu", dtype=torch.float64)
    scaler = model.scaler.to(dtype=torch.float64)
    for name in dataset_info.targets:
        for attr, suffix in [
            ("scales", "scaler_buffer"),
            ("per_target_scales", "per_target_scaler_buffer"),
            ("per_property_scales", "per_property_scaler_buffer"),
        ]:
            getattr(scaler.model, attr)[name] = mts.load_buffer(
                state[f"scaler.{name}_{suffix}"]
            )
    scaler.scales_to(device="cpu", dtype=torch.float64)

    dataset, targets_info, _ = get_dataset(
        expand_dataset_config(config.training_set)[0]
    )
    assert targets_info == dataset_info.targets
    assert all(not target.gradients for target in targets_info.values())
    training = architecture["training"]
    sampler = MaxAtomDistributedBatchSampler(
        dataset,
        max_atoms=training["max_atoms_per_batch"],
        min_atoms=training["min_atoms_per_batch"],
        shuffle=True,
        drop_last=True,
    )
    indices = list(next(iter(sampler)))
    augmenter = RotationalAugmenter(targets_info)
    transforms = []
    if training.get("use_data_augmentation", True):
        transforms.append(augmenter.apply_random_augmentations)
    transforms.extend(
        [
            get_system_with_neighbor_lists_transform(model.requested_neighbor_lists()),
            get_remove_additive_transform(model.additive_models, targets_info),
            get_remove_scale_transform(scaler),
        ]
    )
    collate = CollateFn(target_keys=list(targets_info), callables=transforms)
    systems, targets, extra = unpack_batch(collate([dataset[i] for i in indices]))
    systems, targets, _ = batch_to(
        systems, targets, extra, dtype=torch.float32, device="cpu"
    )
    prepared = systems_to_batch(
        systems,
        model.requested_nl,
        model.atomic_types,
        model.species_to_species_index,
        model.cutoff_function,
        model.cutoff_width,
        model.num_neighbors_adaptive,
        model.adaptive_cutoff_method,
        model.neighbor_cell_shift_mode,
    )
    names = [
        "element_indices_nodes", "element_indices_neighbors", "edge_vectors",
        "edge_distances", "padding_mask", "reverse_neighbor_index", "cutoff_factors",
        "system_indices", "sample_labels", "species", "atomic_cutoffs_stats",
        "centers", "neighbors", "nef_to_edges_neighbor", "cell_shifts",
        "node_positions", "neighbor_image_positions",
    ]
    packed = dict(zip(names, prepared, strict=True))
    input_names = names[:7] + names[-2:]
    inputs = {name: packed[name].detach().contiguous() for name in input_names}
    # Targets are now exactly in loss space: residual energy per atom, direct
    # force per atom, and stress per structure, with per-target scales removed.
    loss_targets = average_by_num_atoms(
        targets, systems, training["per_structure_targets"]
    )
    # The memmap reader keeps dataset-wide system IDs, while PET outputs batch
    # IDs. Verify row order before relabeling; preserve source IDs in provenance.
    dataset_ids = torch.tensor(indices, dtype=torch.int32)
    for name, target in loss_targets.items():
        block = target.block()
        if "atom" in block.samples.names:
            local_samples = packed["sample_labels"]
        else:
            local_samples = mts.Labels.range("system", len(systems))
        expected = local_samples.values.clone()
        expected[:, 0] = dataset_ids[expected[:, 0].long()]
        assert torch.equal(block.samples.values, expected), name
        assert torch.isfinite(block.values).all(), name
        loss_targets[name] = mts.TensorMap(
            target.keys,
            [mts.TensorBlock(
                block.values, local_samples, block.components, block.properties
            )],
        )
    ones = {
        name: mts.TensorMap(
            target.keys,
            [mts.TensorBlock(
                torch.ones_like(b.values), b.samples, b.components, b.properties
            ) for b in target.blocks()],
        )
        for name, target in loss_targets.items()
    }
    prediction_scales = scaler(
        systems, ones, remove=False,
        use_per_target_scales=False, use_per_property_scales=True,
    )
    assert all(len(t) == 1 for t in loss_targets.values())
    assert sum(len(s) for s in systems) <= training["max_atoms_per_batch"]
    payload = {
        "format_version": 2,
        "config": OmegaConf.to_container(config, resolve=True),
        "architecture": architecture,
        "provenance": {
            "config_name": args.config.name,
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "dataset": str(config.training_set.systems.read_from),
            "dataset_size": len(dataset),
            "dataset_indices": [int(i) for i in indices],
            "statistics_checkpoint": args.statistics_checkpoint.name,
            "statistics_checkpoint_epoch": checkpoint["epoch"],
            "seed": seed,
            "augmentation": bool(training.get("use_data_augmentation", True)),
            "torch_version": str(torch.__version__),
            "description": (
                "First atom-capped batch, frozen after augmentation and packing; "
                "fresh PET weights in notebook."
            ),
        },
        "dataset_info": {
            "length_unit": dataset_info.length_unit,
            "atomic_types": dataset_info.atomic_types,
            "targets": {
                name: {
                    "quantity": info.quantity,
                    "unit": info.unit,
                    "sample_kind": info.sample_kind,
                    "shape": list(loss_targets[name].block().values.shape[1:]),
                }
                for name, info in dataset_info.targets.items()
            },
        },
        "inputs": inputs,
        "system_indices": packed["system_indices"],
        "sample_labels": packed["sample_labels"].values,
        "cells": torch.stack([system.cell for system in systems]),
        "num_atoms": torch.tensor([len(system) for system in systems]),
        "loss_targets": {
            name: target.block().values.detach().contiguous()
            for name, target in loss_targets.items()
        },
        "target_samples": {
            name: target.block().samples.values for name, target in loss_targets.items()
        },
        "prediction_scales": {
            k: v.block().values.detach().contiguous()
            for k, v in prediction_scales.items()
        },
        "atomic_cutoffs": packed["atomic_cutoffs_stats"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    # Verify the artifact contains only tensors and ordinary Python containers.
    reloaded = torch.load(args.output, weights_only=True, map_location="cpu")
    assert reloaded["cells"].shape == (len(systems), 3, 3)
    assert all(isinstance(v, torch.Tensor) for v in reloaded["loss_targets"].values())
    print(f"Saved {args.output}: {args.output.stat().st_size / 2**20:.2f} MiB")
    print(f"{len(systems)} structures, {sum(len(s) for s in systems)} atoms")
    for name, value in inputs.items():
        print(f"{name}: {tuple(value.shape)} {value.dtype}")


if __name__ == "__main__":
    main()

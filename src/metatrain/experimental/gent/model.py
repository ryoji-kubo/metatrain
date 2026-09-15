import logging
from typing import Any, Dict, List, Literal, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.operations._add import _add_block_block
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from gent_core import GeneralizedTransformer, GenTData
from metatrain.utils.abc import ModelInterface
from metatrain.utils.additive import CompositionModel
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.scaler import Scaler

from .documentation import ModelHypers


class GenTModel(ModelInterface[ModelHypers]):
    """Metatrain boundary for the independent tensor-only GenT architecture."""

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "GenT layers by Shao Yanming and Xavier Bresson, Sep 10, 2026; "
                "adapted in src/gent_core from gent_layers_original.py"
            ]
        }
    )

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        self.atomic_types = dataset_info.atomic_types
        if any(at < 1 or at >= hypers["max_num_elements"] for at in self.atomic_types):
            raise ValueError("Dataset atomic types must be in [1, max_num_elements)")
        self.target_infos = dict(dataset_info.targets)
        self.has_new_targets = False
        self.finetune_config: Dict[str, Any] = {}
        self.transformer = GeneralizedTransformer(**hypers)
        self.requested_nl = NeighborListOptions(
            cutoff=hypers["cutoff"], full_list=True, strict=True
        )
        self.outputs: Dict[str, ModelOutput] = {}
        self._target_to_raw_output: Dict[str, str] = {}
        for name, info in self.target_infos.items():
            raw_name = self._classify_target(name, info)
            if raw_name == "forces" and not hypers["regress_forces"]:
                raise ValueError("Force targets require regress_forces=True")
            if raw_name == "stress" and not hypers["regress_stress"]:
                raise ValueError("Stress targets require regress_stress=True")
            if raw_name in self._target_to_raw_output.values():
                raise ValueError("GenT supports only one target per physical quantity")
            self._target_to_raw_output[name] = raw_name
            self.outputs[name] = ModelOutput(
                quantity=info.quantity,
                unit=info.unit,
                sample_kind="atom" if raw_name == "energy" else info.sample_kind,
                description=info.description,
            )
        composition_model = CompositionModel(
            hypers={},
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    name: info
                    for name, info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(name, info)
                },
            ),
        )
        self.additive_models = torch.nn.ModuleList([composition_model])
        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def _classify_target(self, target_name: str, target_info: TargetInfo) -> str:
        if len(target_info.layout) != 1:
            raise ValueError(
                "GenTModel currently supports one-block targets only. "
                f"Target {target_name!r} has {len(target_info.layout)} blocks."
            )
        block = target_info.layout.block(0)
        if len(block.properties) != 1:
            raise ValueError(
                "GenTModel currently supports one-property targets only. "
                f"Target {target_name!r} has {len(block.properties)} properties."
            )

        if target_info.is_scalar and target_info.sample_kind == "system":
            if target_info.quantity == "energy" or target_name == "energy":
                return "energy"

        if (
            target_info.is_cartesian
            and target_info.sample_kind == "atom"
            and len(block.components) == 1
        ):
            if target_info.quantity == "force" or "force" in target_name:
                return "forces"

        if (
            target_info.is_cartesian
            and target_info.sample_kind == "system"
            and len(block.components) == 2
        ):
            if (
                target_info.quantity in {"pressure", "stress"}
                or "stress" in target_name
            ):
                return "stress"

        raise ValueError(
            "GenTModel only supports direct energy, atom-level "
            f"Cartesian force, and system-level Cartesian rank-2 stress targets. "
            f"Could not map target {target_name!r}."
        )

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def restart(self, dataset_info: DatasetInfo) -> "GenTModel":
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        if len(new_atomic_types) > 0:
            raise ValueError(
                "GenTModel does not support adding new atomic types "
                f"on restart. New types: {new_atomic_types}."
            )

        new_targets = [
            key for key in merged_info.targets if key not in self.dataset_info.targets
        ]
        if len(new_targets) > 0:
            raise ValueError(
                "GenTModel does not yet support adding new targets "
                f"on restart. New targets: {new_targets}."
            )

        self.dataset_info = merged_info
        self.target_infos = dict(merged_info.targets)
        self.additive_models[0] = self.additive_models[0].restart(
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            )
        )
        self.scaler = self.scaler.restart(dataset_info)
        self.has_new_targets = False
        return self

    def _systems_to_gent_data(self, systems: List[System]) -> GenTData:
        if len(systems) == 0:
            raise ValueError("GenT requires at least one system")
        device = systems[0].positions.device
        dtype = systems[0].positions.dtype
        vectors: List[torch.Tensor] = []
        centers: List[torch.Tensor] = []
        neighbors: List[torch.Tensor] = []
        offset = 0
        for system in systems:
            if len(system) == 0:
                raise ValueError("GenT does not support empty structures")
            nl = system.get_neighbor_list(self.requested_nl)
            indices = nl.samples.values.to(device=device, dtype=torch.long)
            # Neighbor vectors include cell shifts and registered autograd links.
            vectors.append(nl.values.squeeze(-1).to(device=device, dtype=dtype))
            centers.append(indices[:, 0] + offset)
            neighbors.append(indices[:, 1] + offset)
            offset += len(system)
        return GenTData(
            atomic_numbers=torch.cat([system.types for system in systems]).long(),
            num_atoms=torch.tensor([len(system) for system in systems], device=device),
            edge_vectors=torch.cat(vectors),
            edge_centers=torch.cat(centers),
            edge_neighbors=torch.cat(neighbors),
            cells=torch.stack([system.cell for system in systems]),
        )

    def _system_samples(self, systems: List[System], device: torch.device) -> Labels:
        return Labels(
            names=["system"],
            values=torch.arange(len(systems), dtype=torch.int64, device=device).reshape(
                -1, 1
            ),
            assume_unique=True,
        )

    def _atom_samples(self, systems: List[System], device: torch.device) -> Labels:
        sample_values = []
        for i_system, system in enumerate(systems):
            sample_values.append(
                torch.stack(
                    [
                        torch.full(
                            (len(system),),
                            i_system,
                            dtype=torch.int64,
                            device=device,
                        ),
                        torch.arange(len(system), dtype=torch.int64, device=device),
                    ],
                    dim=1,
                )
            )
        return Labels(
            names=["system", "atom"],
            values=torch.cat(sample_values, dim=0),
            assume_unique=True,
        )

    def _to_tensormap(
        self,
        target_name: str,
        values: torch.Tensor,
        samples: Labels,
    ) -> TensorMap:
        target_info = self.target_infos[target_name]
        layout_block = target_info.layout.block(0)
        device = values.device
        block = TensorBlock(
            values=values,
            samples=samples,
            components=[
                component.to(device=device) for component in layout_block.components
            ],
            properties=layout_block.properties.to(device=device),
        )
        return TensorMap(target_info.layout.keys.to(device=device), [block])

    def _raw_outputs_to_tensormaps(
        self,
        systems: List[System],
        raw_outputs: Dict[str, torch.Tensor],
        requested_outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        device = systems[0].positions.device
        system_samples = self._system_samples(systems, device)
        atom_samples = self._atom_samples(systems, device)
        predictions: Dict[str, TensorMap] = {}
        for name, output in requested_outputs.items():
            if name not in self._target_to_raw_output:
                raise ValueError("Unsupported GenT output requested: " + name)
            raw_name = self._target_to_raw_output[name]
            if raw_name == "energy":
                tmap = self._to_tensormap(
                    name, raw_outputs["atomic_energy"].reshape(-1, 1), atom_samples
                )
                if selected_atoms is not None:
                    tmap = mts.slice(tmap, axis="samples", selection=selected_atoms)
                if output.sample_kind == "system":
                    tmap = mts.sum_over_samples(tmap, sample_names=["atom"])
                elif output.sample_kind != "atom":
                    raise ValueError(
                        "GenT energy sample_kind must be 'atom' or 'system'"
                    )
                predictions[name] = tmap
            elif raw_name == "forces":
                if output.sample_kind != "atom":
                    raise ValueError("GenT direct forces require sample_kind='atom'")
                tmap = self._to_tensormap(
                    name, raw_outputs["forces"].reshape(-1, 3, 1), atom_samples
                )
                if selected_atoms is not None:
                    tmap = mts.slice(tmap, axis="samples", selection=selected_atoms)
                predictions[name] = tmap
            else:
                if output.sample_kind != "system":
                    raise ValueError("GenT direct stress requires sample_kind='system'")
                if selected_atoms is not None:
                    raise ValueError("GenT direct stress requires the full structure")
                predictions[name] = self._to_tensormap(
                    name, raw_outputs["stress"].reshape(-1, 3, 3, 1), system_samples
                )
        return predictions

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        raw_outputs = self.transformer(self._systems_to_gent_data(systems))
        return_dict = self._raw_outputs_to_tensormaps(
            systems, raw_outputs, outputs, selected_atoms
        )

        if not self.training:
            return_dict = self.scaler(
                systems,
                return_dict,
                selected_atoms=selected_atoms,
                use_per_target_scales=True,
                use_per_property_scales=True,
            )
            for additive_model in self.additive_models:
                outputs_for_additive_model: Dict[str, ModelOutput] = {}
                for name, output in outputs.items():
                    if name in additive_model.outputs:
                        outputs_for_additive_model[name] = output
                additive_contributions = additive_model(
                    systems,
                    outputs_for_additive_model,
                    selected_atoms,
                )
                for name in additive_contributions:
                    output_blocks: List[TensorBlock] = []
                    for key, block in return_dict[name].items():
                        if key in additive_contributions[name].keys:
                            output_blocks.append(
                                _add_block_block(
                                    block,
                                    additive_contributions[name]
                                    .block(key)
                                    .to(device=block.device, dtype=block.dtype),
                                )
                            )
                        else:
                            output_blocks.append(block.copy(deep=False))
                    return_dict[name] = TensorMap(return_dict[name].keys, output_blocks)

        return return_dict

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "GenTModel":
        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint['best_epoch']}")
            model_state_dict = checkpoint["best_model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        model_data = checkpoint["model_data"]
        model = cls(
            hypers=model_data["model_hypers"],
            dataset_info=model_data["dataset_info"],
        )
        state_dict = dict(model_state_dict)
        model.finetune_config = state_dict.pop("finetune_config", {})
        # Preserve checkpoint dtypes, including float64 composition/scaler buffers.
        model.load_state_dict(state_dict, assign=True)
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()
        model.metadata = merge_metadata(model.metadata, checkpoint.get("metadata"))
        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for GenTModel")

        self.to(dtype)
        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            # Global attention couples every atom in each input structure.
            interaction_range=float("inf"),
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )
        metadata = merge_metadata(self.metadata, metadata)
        return AtomisticModel(self.eval(), metadata, capabilities)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        if (
            checkpoint.get("model_ckpt_version", cls.__checkpoint_version__)
            != cls.__checkpoint_version__
        ):
            raise RuntimeError("No checkpoint upgrade path is available for GenTModel.")
        checkpoint["model_ckpt_version"] = cls.__checkpoint_version__
        return checkpoint

    def get_checkpoint(self) -> Dict[str, Any]:
        model_state_dict = self.state_dict()
        model_state_dict["finetune_config"] = self.finetune_config
        return {
            "architecture_name": "experimental.gent",
            "model_ckpt_version": self.__checkpoint_version__,
            "model_data": {
                "model_hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": self.state_dict(),
        }

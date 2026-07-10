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

from metatrain.utils.abc import ModelInterface
from metatrain.utils.additive import CompositionModel
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.scaler import Scaler

from .documentation import ModelHypers
from .modules.transformer import StructureTransformer, TransformerData


class StructureTransformerModel(ModelInterface[ModelHypers]):
    """Metatrain wrapper around the direct structure transformer from equiformer_v3."""

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "Adapted from /home/ryoji/equivarient/equiformer_v3/experimental/"
                "models/transformer/transformer.py"
            ]
        }
    )

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        self.atomic_types = dataset_info.atomic_types
        self.target_infos = dict(dataset_info.targets)
        self.target_names = list(self.target_infos.keys())
        self.symmetrize_stress = bool(hypers["symmetrize_stress"])
        self.has_new_targets = False
        self.finetune_config: Dict[str, Any] = {}

        transformer_hypers = dict(hypers)
        transformer_hypers.pop("symmetrize_stress", None)
        self.transformer = StructureTransformer(**transformer_hypers)

        self.outputs: Dict[str, ModelOutput] = {}
        self._target_to_raw_output: Dict[str, str] = {}
        for target_name, target_info in self.target_infos.items():
            self.outputs[target_name] = ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                sample_kind=target_info.sample_kind,
                description=target_info.description,
            )
            self._target_to_raw_output[target_name] = self._classify_target(
                target_name, target_info
            )

        composition_model = CompositionModel(
            hypers={},
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        self.additive_models = torch.nn.ModuleList([composition_model])
        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)

    def _classify_target(self, target_name: str, target_info: TargetInfo) -> str:
        block = target_info.layout.block(0)
        if len(target_info.layout) != 1:
            raise ValueError(
                "StructureTransformerModel currently supports one-block targets only. "
                f"Target {target_name!r} has {len(target_info.layout)} blocks."
            )
        if len(block.properties) != 1:
            raise ValueError(
                "StructureTransformerModel currently supports one-property targets only. "
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
            if target_info.quantity in {"pressure", "stress"} or "stress" in target_name:
                return "stress"

        raise ValueError(
            "StructureTransformerModel only supports direct energy, atom-level "
            f"Cartesian force, and system-level Cartesian rank-2 stress targets. "
            f"Could not map target {target_name!r}."
        )

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        # The copied transformer is a global sequence model and does not consume local
        # neighbor lists. Returning an empty list lets the PET trainer keep the same
        # collate stack while avoiding unnecessary neighbor-list construction.
        return []

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def restart(self, dataset_info: DatasetInfo) -> "StructureTransformerModel":
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        if len(new_atomic_types) > 0:
            raise ValueError(
                "StructureTransformerModel does not support adding new atomic types "
                f"on restart. New types: {new_atomic_types}."
            )

        new_targets = [
            key for key in merged_info.targets if key not in self.dataset_info.targets
        ]
        if len(new_targets) > 0:
            raise ValueError(
                "StructureTransformerModel does not yet support adding new targets "
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

    def _systems_to_transformer_data(self, systems: List[System]) -> TransformerData:
        device = systems[0].positions.device
        positions = torch.cat([system.positions for system in systems], dim=0)
        atomic_numbers = torch.cat([system.types for system in systems], dim=0)
        batch = torch.cat(
            [
                torch.full((len(system),), i, dtype=torch.long, device=device)
                for i, system in enumerate(systems)
            ],
            dim=0,
        )
        cells = torch.stack([system.cell for system in systems], dim=0)
        return TransformerData(
            atomic_numbers=atomic_numbers,
            pos=positions,
            batch=batch,
            cell=cells,
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
            components=[component.to(device=device) for component in layout_block.components],
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
        for target_name in requested_outputs:
            if target_name not in self._target_to_raw_output:
                raise ValueError("Unsupported output requested")

            raw_name = self._target_to_raw_output[target_name]
            if raw_name not in raw_outputs:
                raise ValueError(
                    "Transformer did not produce a required raw output"
                )

            raw = raw_outputs[raw_name]
            if raw_name == "energy":
                values = raw.reshape(-1, 1)
                predictions[target_name] = self._to_tensormap(
                    target_name, values, system_samples
                )
            elif raw_name == "forces":
                values = raw.reshape(-1, 3, 1)
                tmap = self._to_tensormap(target_name, values, atom_samples)
                if selected_atoms is not None:
                    tmap = mts.slice(tmap, axis="samples", selection=selected_atoms)
                predictions[target_name] = tmap
            elif raw_name == "stress":
                stress = raw.reshape(-1, 3, 3, 1)
                if self.symmetrize_stress:
                    stress = (stress + stress.transpose(1, 2)) / 2.0
                predictions[target_name] = self._to_tensormap(
                    target_name, stress, system_samples
                )
            else:
                raise ValueError("Unsupported raw output")

        return predictions

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        data = self._systems_to_transformer_data(systems)
        raw_outputs = self.transformer(data)
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
    ) -> "StructureTransformerModel":
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
        model.load_state_dict(state_dict)
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()
        model.metadata = merge_metadata(model.metadata, checkpoint.get("metadata"))
        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for StructureTransformerModel")

        self.to(dtype)
        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            # This is a global sequence model rather than a finite-cutoff local model.
            # Use 0.0 here to avoid advertising a PET-like locality radius.
            interaction_range=0.0,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )
        metadata = merge_metadata(self.metadata, metadata)
        return AtomisticModel(self.eval(), metadata, capabilities)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        if checkpoint.get("model_ckpt_version", cls.__checkpoint_version__) != cls.__checkpoint_version__:
            raise RuntimeError(
                "No checkpoint upgrade path is available for StructureTransformerModel."
            )
        checkpoint["model_ckpt_version"] = cls.__checkpoint_version__
        return checkpoint

    def get_checkpoint(self) -> Dict[str, Any]:
        model_state_dict = self.state_dict()
        model_state_dict["finetune_config"] = self.finetune_config
        return {
            "architecture_name": "experimental.structure_transformer",
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

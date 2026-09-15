import torch

from metatrain.experimental.structure_transformer.modules.coordinate_utils import (
    cartesian_to_fractional_dense as legacy_cartesian_to_fractional_dense,
)
from metatrain.experimental.structure_transformer.modules.transformer import (
    StructureTransformer as LegacyStructureTransformer,
)
from metatrain.experimental.structure_transformer.modules.transformer import (
    TransformerData as LegacyTransformerData,
)
from structure_transformer_core import StructureTransformer, TransformerData
from structure_transformer_core.coordinate_utils import cartesian_to_fractional_dense


def test_legacy_module_paths_reexport_shared_core():
    assert LegacyStructureTransformer is StructureTransformer
    assert LegacyTransformerData is TransformerData
    assert legacy_cartesian_to_fractional_dense is cartesian_to_fractional_dense


def test_shared_core_forward_smoke():
    torch.manual_seed(0)
    model = StructureTransformer(
        max_num_elements=16,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        encoder_hidden_dim=24,
        mlp_hidden_dim=24,
        dropout=0.0,
        attn_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
        regress_forces=True,
        regress_stress=True,
    )
    data = TransformerData(
        atomic_numbers=torch.tensor([6, 8, 1], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.1], [0.0, 0.7, 0.2]],
            dtype=torch.float32,
        ),
        batch=torch.zeros(3, dtype=torch.long),
        cell=torch.eye(3, dtype=torch.float32).unsqueeze(0),
    )

    outputs = model(data)

    assert outputs["energy"].shape == (1,)
    assert outputs["forces"].shape == (3, 3)
    assert outputs["stress"].shape == (1, 9)

"""
Experimental Structure Transformer
==================================

A metatrain wrapper around the direct structure transformer from the local
``equiformer_v3`` repository. This architecture intentionally reuses the PET trainer
so that data preprocessing, composition baselines, target scaling, augmentation,
losses, and metrics can be compared against PET with the neural architecture swapped.
"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.pet.documentation import TrainerHypers


class ModelHypers(TypedDict):
    """Hyperparameters for the experimental structure transformer."""

    max_num_elements: int = 128
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    encoder_hidden_dim: Optional[int] = None
    mlp_hidden_dim: Optional[int] = None
    dropout: float = 0.05
    attn_dropout: Optional[float] = None
    residual_dropout: Optional[float] = None
    mlp_dropout: Optional[float] = None
    position_scale: float = 1.0
    position_representation: Literal["cartesian", "fractional"] = "cartesian"
    use_rotary_embeddings: bool = False
    atom_ordering: Literal["none", "position_ids", "permute"] = "none"
    center_positions: bool = True
    include_cell_energy: bool = True
    include_cell_stress: bool = True
    regress_forces: bool = True
    regress_stress: bool = True
    direct_prediction: bool = True
    avg_num_nodes: float = 1.0
    symmetrize_stress: bool = False
    """If true, symmetrize the 3x3 stress prediction before returning it."""

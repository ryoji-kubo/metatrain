"""Generalized Transformer (Experimental)
====================================

GenT with the PET training pipeline.

Geometry is encoded from full periodic neighbor lists into dense GenT edge states.
Energy is a sum of atom contributions; direct forces and symmetric stresses use
invariant edge coefficients and Cartesian edge vectors. These direct outputs are
not derivatives of energy. Sequence-index RoPE is not used for atomic structures.
"""

from typing import Optional

from typing_extensions import TypedDict

from metatrain.pet.documentation import TrainerHypers  # noqa: F401


class ModelHypers(TypedDict):
    """Hyperparameters for GenT; dense edge storage scales as B * N_max^2 * D."""

    max_num_elements: int = 128
    """Exclusive upper bound for atomic numbers in the embedding table."""
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    hidden_dim_sa: Optional[int] = None
    """Node self-attention width; None uses embed_dim."""
    hidden_dim_ca_n: Optional[int] = None
    """Edge-to-node attention width; None uses embed_dim."""
    hidden_dim_ca_e: Optional[int] = None
    """Node-to-edge attention width; None uses embed_dim."""
    mlp_hidden_dim: Optional[int] = None
    """Gated SiLU MLP width; None uses int(2.25 * embed_dim)."""
    attn_dropout: float = 0.0
    residual_dropout: float = 0.0
    cutoff: float = 5.0
    """Geometry neighbor cutoff in the dataset length unit; attention remains global."""
    num_radial_basis: int = 32
    regress_forces: bool = True
    """Enable a direct, nonconservative atom force head."""
    regress_stress: bool = True
    """Enable a direct symmetric system stress head."""

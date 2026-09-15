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
    coordinate_encoding: Literal["absolute_mlp", "v37_torus_relative"] = "absolute_mlp"
    """Coordinate path: old absolute MLP or v37 relative torus features."""

    coord_num_harmonics: int = 4
    """Number of integer Fourier harmonics in the v37 coordinate encoder."""

    coord_encoder_chunk_size: Optional[int] = None
    """Receiver-atom chunk size for the v37 all-pairs coordinate encoder."""

    coord_encoder_use_checkpoint: bool = False
    """If true, checkpoint v37 coordinate chunks to reduce training memory."""

    use_rotary_embeddings: bool = False
    use_periodic_rope: bool = False
    """If true, use v37 integer-harmonic fractional-coordinate RoPE."""

    atom_ordering: Literal["none", "position_ids", "permute"] = "none"
    center_positions: bool = True
    fractional_wrap_eps: float = 1.0e-6
    """Tolerance used when canonicalizing fractional coordinates near 0 and 1."""

    atom_embedding_type: Literal["embedding", "scalar"] = "embedding"
    """Atom encoder: embedding table or v37-style scalar Fourier embedding."""

    atomic_number_scale: float = 118.0
    """Scale applied before the scalar atomic-number Fourier embedding."""

    atom_scalar_embedding_scale: float = 1000.0
    """Frequency scale used by the scalar atomic-number Fourier embedding."""

    include_cell_energy: bool = True
    include_cell_stress: bool = True
    regress_forces: bool = True
    regress_stress: bool = True
    edge_vector_head: bool = False
    """If true, add a local edge-vector readout for force and stress prediction."""

    edge_vector_head_cutoff: float = 10.0
    """Cutoff radius for the optional edge-vector prediction head."""

    edge_vector_head_hidden_dim: int = 256
    """Hidden dimension used inside the optional edge-vector prediction head."""

    edge_vector_head_num_radial_basis: int = 16
    """Number of Gaussian radial basis functions for the edge-vector head."""

    edge_vector_head_replace_direct: bool = False
    """If true, use only the edge-vector head for force and stress outputs."""

    force_readout_type: Literal["mlp", "pair_cross_attention"] = "mlp"
    """Prediction head used before the final force vector projection."""

    stress_readout_type: Literal["mlp", "pair_cross_attention"] = "mlp"
    """Prediction head used before the final per-atom stress projection."""

    pair_readout_num_heads: Optional[int] = None
    """Attention heads for pair_cross_attention readouts; defaults to num_heads."""

    pair_readout_hidden_dim: Optional[int] = None
    """SwiGLU hidden width for pair_cross_attention readouts."""

    pair_readout_num_layers: int = 1
    """Number of receiver-wise cross-attention layers in each enabled readout."""

    pair_readout_dropout: Optional[float] = None
    """Dropout inside pair_cross_attention readouts; defaults to dropout."""

    pair_readout_chunk_size: Optional[int] = None
    """Receiver-atom chunk size for pair_cross_attention readouts."""

    pair_readout_use_checkpoint: bool = False
    """If true, checkpoint pair_cross_attention readout chunks during training."""

    pair_readout_include_pair_geometry: bool = False
    """If true, add periodic Fourier pair-geometry bias to readout attention."""

    pair_readout_exclude_self: bool = True
    """If true, atom readout attention masks self-pairs when other atoms exist."""

    graph_attention: Literal["none", "binary", "smooth_cutoff"] = "none"
    """Optional PET-style graph prior applied as an additive attention bias."""

    graph_attention_cutoff: float = 4.5
    """Neighbor-list cutoff radius used for graph attention."""

    graph_attention_num_neighbors_adaptive: Optional[int] = None
    """PET-style target effective neighbor count for adaptive graph cutoffs."""

    graph_attention_adaptive_cutoff_method: Literal["grid", "solver"] = "solver"
    """PET adaptive cutoff method used when adaptive graph neighbors are set."""

    graph_attention_cutoff_width: float = 0.5
    """Width of the PET cutoff transition used for smooth graph attention."""

    graph_attention_cutoff_function: Literal["Bump", "Cosine"] = "Bump"
    """PET cutoff function used for smooth graph attention."""

    graph_attention_bias_strength: float = 1.0
    """Multiplier lambda on the PET-style log-cutoff attention bias."""

    graph_attention_epsilon: float = 1.0e-15
    """Minimum cutoff factor before taking log for graph attention."""

    direct_prediction: bool = True
    avg_num_nodes: float = 1.0
    symmetrize_stress: bool = False
    """If true, symmetrize the 3x3 stress prediction before returning it."""

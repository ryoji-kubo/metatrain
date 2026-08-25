import math
from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .coordinate_utils import cartesian_to_fractional_dense

try:
    from torch_geometric.utils import to_dense_batch as _pyg_to_dense_batch
except ImportError:
    _pyg_to_dense_batch = None


class TransformerData(NamedTuple):
    atomic_numbers: torch.Tensor
    pos: torch.Tensor
    batch: torch.Tensor
    cell: torch.Tensor


def _to_dense_batch_torch(
    values: torch.Tensor,
    batch: torch.Tensor,
    fill_value: float = 0.0,
    batch_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 0:
        if values.shape[0] == 0:
            batch_size = 0
        else:
            batch_size = int(batch.max().item()) + 1

    if values.shape[0] == 0:
        if values.dim() == 1:
            return values.new_full((batch_size, 0), fill_value), torch.empty(
                (batch_size, 0), dtype=torch.bool, device=values.device
            )
        return (
            values.new_full(
                (batch_size, 0, values.size(1)),
                fill_value,
            ),
            torch.empty((batch_size, 0), dtype=torch.bool, device=values.device),
        )

    counts = torch.bincount(batch, minlength=batch_size)
    if counts.shape[0] != batch_size:
        raise ValueError("batch contains a system index outside the cell batch")
    max_count = int(counts.max().item())
    if values.dim() == 1:
        dense = values.new_full((batch_size, max_count), fill_value)
    else:
        dense = values.new_full((batch_size, max_count, values.size(1)), fill_value)
    mask = torch.zeros(
        (batch_size, max_count), dtype=torch.bool, device=values.device
    )

    for i_system in range(batch_size):
        atom_indices = torch.nonzero(batch == i_system).reshape(-1)
        n_atoms = atom_indices.numel()
        if n_atoms == 0:
            continue
        dense[i_system, :n_atoms] = values.index_select(0, atom_indices)
        mask[i_system, :n_atoms] = True

    return dense, mask


@torch.jit.unused
def _to_dense_batch_pyg(
    values: torch.Tensor,
    batch: torch.Tensor,
    fill_value: float = 0.0,
    batch_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _pyg_to_dense_batch is None:
        return _to_dense_batch_torch(values, batch, fill_value, batch_size)
    if batch_size < 0:
        return _pyg_to_dense_batch(values, batch, fill_value=fill_value)
    return _pyg_to_dense_batch(
        values,
        batch,
        fill_value=fill_value,
        batch_size=batch_size,
    )


def _to_dense_batch(
    values: torch.Tensor,
    batch: torch.Tensor,
    fill_value: float = 0.0,
    batch_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return _to_dense_batch_torch(values, batch, fill_value, batch_size)
    return _to_dense_batch_pyg(values, batch, fill_value, batch_size)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _build_complex_pe(
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    k = torch.arange(head_dim // 2, device=device).float()
    omegas = 10000 ** (-2 * k / head_dim)
    positions = torch.arange(max_seq_len, device=device)
    angles = torch.outer(positions, omegas).float()
    return torch.polar(torch.ones_like(angles), angles)


def _apply_rotary_emb(
    x: torch.Tensor,
    pe_cplx: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = x.size()
    x_pair = x.view(batch_size, num_heads, seq_len, head_dim // 2, 2)
    x_complex = torch.view_as_complex(x_pair.float())
    if position_ids is None:
        pe = pe_cplx[:seq_len, :].view(1, 1, seq_len, head_dim // 2)
    else:
        pe = pe_cplx[position_ids].unsqueeze(1)
    x_rotated = x_complex * pe
    x_real = torch.view_as_real(x_rotated)
    return x_real.view(batch_size, num_heads, seq_len, head_dim).type_as(x)


def _canonical_frac_coords(frac_coords: torch.Tensor, eps: float) -> torch.Tensor:
    frac_coords = torch.remainder(frac_coords, 1.0)
    if eps > 0.0:
        frac_coords = torch.where(
            (frac_coords < eps) | (frac_coords > 1.0 - eps),
            torch.zeros_like(frac_coords),
            frac_coords,
        )
    return frac_coords


def _scalar_embedding(
    scalars: torch.Tensor,
    dim: int,
    scalar_scale: float,
) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("scalar embedding requires an even embedding dimension")
    half_dim = dim // 2
    dtype = scalars.dtype
    frequencies = 10000.0 ** (
        -torch.arange(half_dim, device=scalars.device, dtype=dtype) / half_dim
    )
    angles = scalar_scale * scalars.unsqueeze(-1) * frequencies
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def _precompute_periodic_3d_rope_harmonics(
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim < 2:
        raise ValueError("periodic coordinate RoPE requires head_dim >= 2")
    n_pairs = head_dim // 2
    pair_index = torch.arange(n_pairs, dtype=torch.long)
    pair_axis = pair_index % 3
    harmonic = (pair_index // 3 + 1).float()
    return pair_axis, harmonic


def _apply_periodic_3d_rotary_emb(
    x: torch.Tensor,
    frac_coords: torch.Tensor,
    pair_axis: torch.Tensor,
    harmonic: torch.Tensor,
    wrap_eps: float,
) -> torch.Tensor:
    num_atoms = x.shape[-2]
    head_dim = x.shape[-1]
    n_pairs = head_dim // 2
    rotary_dim = 2 * n_pairs
    if num_atoms == 0 or rotary_dim == 0:
        return x

    pair_axis = pair_axis.to(device=frac_coords.device)
    harmonic = harmonic.to(device=frac_coords.device, dtype=torch.float32)
    frac_coords = _canonical_frac_coords(frac_coords.float(), wrap_eps)

    angles = (
        2.0
        * math.pi
        * frac_coords[..., pair_axis]
        * harmonic.view(1, 1, n_pairs)
    )
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)

    x_rot = x[..., :rotary_dim].float()
    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos
    y_rot = torch.stack((y_even, y_odd), dim=-1).flatten(-2)

    if rotary_dim < head_dim:
        y = torch.cat((y_rot, x[..., rotary_dim:].float()), dim=-1)
    else:
        y = y_rot
    return y.to(dtype=x.dtype)


class TorusRelativeCoordEncoder(nn.Module):
    """Periodic, translation-invariant coordinate encoder on the fractional torus."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_harmonics: int = 4,
        wrap_eps: float = 1.0e-6,
        chunk_size: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if num_harmonics < 1:
            raise ValueError("num_harmonics must be >= 1")
        if wrap_eps < 0.0:
            raise ValueError("wrap_eps must be non-negative")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive when provided")
        self.embed_dim = embed_dim
        self.num_harmonics = num_harmonics
        self.wrap_eps = wrap_eps
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint
        periodic_feature_dim = 3 * 2 * num_harmonics
        self.pair_mlp = nn.Sequential(
            nn.Linear(periodic_feature_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def _forward_receiver_chunk(
        self,
        u: torch.Tensor,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        num_atoms = u.shape[1]
        receiver_coords = u[:, start:end, :]
        chunk_atoms = end - start
        delta = receiver_coords.unsqueeze(2) - u.unsqueeze(1)
        harmonics = torch.arange(
            1,
            self.num_harmonics + 1,
            device=u.device,
            dtype=u.dtype,
        )
        angles = 2.0 * math.pi * delta.unsqueeze(-1) * harmonics.view(
            1, 1, 1, 1, -1
        )
        pair_features = torch.cat((angles.cos(), angles.sin()), dim=-1)
        pair_features = pair_features.flatten(start_dim=-2)

        neighbor_features = atom_features.to(dtype=pair_features.dtype).unsqueeze(1)
        neighbor_features = neighbor_features.expand(-1, chunk_atoms, -1, -1)
        pair_input = torch.cat((pair_features, neighbor_features), dim=-1)
        messages = self.pair_mlp(pair_input)

        receiver_mask = atom_mask[:, start:end]
        valid_pair = receiver_mask.unsqueeze(2) & atom_mask.unsqueeze(1)
        receiver_index = torch.arange(start, end, device=atom_mask.device)
        neighbor_index = torch.arange(num_atoms, device=atom_mask.device)
        self_pair = receiver_index.view(1, chunk_atoms, 1) == neighbor_index.view(
            1, 1, num_atoms
        )
        valid_pair = valid_pair & ~self_pair
        messages = messages.masked_fill(~valid_pair.unsqueeze(-1), 0.0)
        denom = valid_pair.sum(dim=2).clamp_min(1).unsqueeze(-1)
        coord_features = messages.sum(dim=2) / denom.to(dtype=messages.dtype)
        coord_features = coord_features * receiver_mask.unsqueeze(-1).to(
            dtype=coord_features.dtype
        )
        return coord_features.to(dtype=atom_features.dtype)

    def forward(
        self,
        frac_coords: torch.Tensor,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        if frac_coords.ndim != 3 or frac_coords.shape[-1] != 3:
            raise ValueError("frac_coords must have shape [B, N, 3]")
        if atom_features.ndim != 3:
            raise ValueError("atom_features must have shape [B, N, D]")
        if atom_features.shape[:2] != frac_coords.shape[:2]:
            raise ValueError("atom_features and frac_coords must share [B, N]")
        if atom_features.shape[-1] != self.embed_dim:
            raise ValueError("atom_features has the wrong embedding dimension")

        batch_size, num_atoms, _ = frac_coords.shape
        if num_atoms == 0:
            return atom_features.new_zeros((batch_size, 0, self.embed_dim))

        u = _canonical_frac_coords(frac_coords.float(), self.wrap_eps)
        chunk_size = num_atoms if self.chunk_size is None else self.chunk_size
        chunks = []
        for start in range(0, num_atoms, chunk_size):
            end = min(start + chunk_size, num_atoms)
            if self.use_checkpoint and self.training and atom_features.requires_grad:

                def chunk_fn(
                    u_: torch.Tensor,
                    atom_features_: torch.Tensor,
                    atom_mask_: torch.Tensor,
                    chunk_start: int = start,
                    chunk_end: int = end,
                ) -> torch.Tensor:
                    return self._forward_receiver_chunk(
                        u_, atom_features_, atom_mask_, chunk_start, chunk_end
                    )

                chunk = activation_checkpoint(
                    chunk_fn,
                    u,
                    atom_features,
                    atom_mask,
                    use_reentrant=False,
                )
            else:
                chunk = self._forward_receiver_chunk(
                    u,
                    atom_features,
                    atom_mask,
                    start,
                    end,
                )
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)


def _atom_reordering(
    frac_coords: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    values = (
        100.0 * frac_coords[:, :, 0]
        + 10.0 * frac_coords[:, :, 1]
        + frac_coords[:, :, 2]
    )
    values = values.masked_fill(~atom_mask, torch.inf)
    return torch.argsort(values, dim=1, stable=True)


def _invert_permutation(index_order: torch.Tensor) -> torch.Tensor:
    inverse_order = torch.empty_like(index_order)
    positions = torch.arange(
        index_order.shape[1],
        device=index_order.device,
    ).unsqueeze(0).expand_as(index_order)
    inverse_order.scatter_(1, index_order, positions)
    return inverse_order


def _gather_atom_features(
    x: torch.Tensor,
    index_order: torch.Tensor,
) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=index_order.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
    )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        use_rotary_embeddings: bool,
        use_periodic_rope: bool = False,
        fractional_wrap_eps: float = 1.0e-6,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        head_dim = embed_dim // num_heads
        if use_rotary_embeddings and use_periodic_rope:
            raise ValueError(
                "use_rotary_embeddings and use_periodic_rope are mutually exclusive"
            )
        if use_rotary_embeddings and head_dim % 2 != 0:
            raise ValueError("RoPE requires an even per-head dimension")
        if use_periodic_rope and head_dim < 2:
            raise ValueError("periodic coordinate RoPE requires head_dim >= 2")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_periodic_rope = use_periodic_rope
        self.fractional_wrap_eps = fractional_wrap_eps
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        if self.use_periodic_rope:
            pair_axis, harmonic = _precompute_periodic_3d_rope_harmonics(head_dim)
            self.register_buffer("periodic_rope_pair_axis", pair_axis, persistent=False)
            self.register_buffer("periodic_rope_harmonic", harmonic, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        rotary_pe_cplx: Optional[torch.Tensor] = None,
        rotary_position_ids: Optional[torch.Tensor] = None,
        periodic_rope_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rotary_embeddings:
            if rotary_pe_cplx is None:
                raise ValueError("rotary_pe_cplx is required when RoPE is enabled")
            q_for_scores = _apply_rotary_emb(q, rotary_pe_cplx, rotary_position_ids)
            k_for_scores = _apply_rotary_emb(k, rotary_pe_cplx, rotary_position_ids)
        elif self.use_periodic_rope:
            if periodic_rope_coords is None:
                raise ValueError(
                    "periodic_rope_coords is required when periodic RoPE is enabled"
                )
            if periodic_rope_coords.shape[:2] != (batch_size, seq_len - 1):
                raise ValueError(
                    "periodic_rope_coords must have shape [B, seq_len - 1, 3]"
                )
            q_for_scores = q.clone()
            k_for_scores = k.clone()
            q_for_scores[:, :, 1:, :] = _apply_periodic_3d_rotary_emb(
                q[:, :, 1:, :],
                periodic_rope_coords,
                self.periodic_rope_pair_axis,
                self.periodic_rope_harmonic,
                self.fractional_wrap_eps,
            )
            k_for_scores[:, :, 1:, :] = _apply_periodic_3d_rotary_emb(
                k[:, :, 1:, :],
                periodic_rope_coords,
                self.periodic_rope_pair_axis,
                self.periodic_rope_harmonic,
                self.fractional_wrap_eps,
            )
        else:
            q_for_scores = q
            k_for_scores = k

        attn = torch.matmul(
            q_for_scores.float(), k_for_scores.float().transpose(-2, -1)
        )
        if self.use_periodic_rope:
            unrotated_attn = torch.matmul(q.float(), k.float().transpose(-2, -1))
            attn[:, :, 0, :] = unrotated_attn[:, :, 0, :]
            attn[:, :, :, 0] = unrotated_attn[:, :, :, 0]
        attn = attn * self.scale
        attn = attn.masked_fill(
            ~valid_mask.view(batch_size, 1, 1, seq_len),
            -1.0e30,
        )
        attn = F.softmax(attn, dim=-1).to(v.dtype)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2.0 * embed_dim / 3.0)
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
        residual_dropout: float,
        mlp_hidden_dim: Optional[int],
        mlp_dropout: float,
        use_rotary_embeddings: bool,
        use_periodic_rope: bool = False,
        fractional_wrap_eps: float = 1.0e-6,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim,
            num_heads,
            attn_dropout,
            use_rotary_embeddings,
            use_periodic_rope,
            fractional_wrap_eps,
        )
        self.attn_drop = nn.Dropout(residual_dropout)
        self.mlp_norm = RMSNorm(embed_dim)
        self.mlp = SwiGLU(embed_dim, mlp_hidden_dim)
        self.mlp_drop = nn.Dropout(mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        rotary_pe_cplx: Optional[torch.Tensor] = None,
        rotary_position_ids: Optional[torch.Tensor] = None,
        periodic_rope_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn_drop(
            self.attn(
                self.attn_norm(x),
                valid_mask,
                rotary_pe_cplx,
                rotary_position_ids,
                periodic_rope_coords,
            )
        )
        x = x + self.mlp_drop(self.mlp(self.mlp_norm(x)))
        return x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)


class StructureTransformer(nn.Module):
    """
    LLaMA-style direct predictor for structures, copied from the local equiformer_v3
    transformer implementation and made self-contained for metatrain.

    Atoms are sequence tokens. Each token receives an atomic-number embedding and an
    encoded position. The cell is encoded as a per-structure token and as a per-layer
    sequence condition.
    """

    def __init__(
        self,
        max_num_elements: int = 128,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        encoder_hidden_dim: Optional[int] = None,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.05,
        attn_dropout: Optional[float] = None,
        residual_dropout: Optional[float] = None,
        mlp_dropout: Optional[float] = None,
        position_scale: float = 1.0,
        position_representation: str = "cartesian",
        coordinate_encoding: str = "absolute_mlp",
        coord_num_harmonics: int = 4,
        coord_encoder_chunk_size: Optional[int] = None,
        coord_encoder_use_checkpoint: bool = False,
        use_rotary_embeddings: bool = False,
        use_periodic_rope: bool = False,
        atom_ordering: str = "none",
        center_positions: bool = True,
        fractional_wrap_eps: float = 1.0e-6,
        atom_embedding_type: str = "embedding",
        atomic_number_scale: float = 118.0,
        atom_scalar_embedding_scale: float = 1000.0,
        include_cell_energy: bool = True,
        include_cell_stress: bool = True,
        regress_forces: bool = True,
        regress_stress: bool = False,
        edge_vector_head: bool = False,
        edge_vector_head_cutoff: float = 10.0,
        edge_vector_head_hidden_dim: int = 256,
        edge_vector_head_num_radial_basis: int = 16,
        edge_vector_head_replace_direct: bool = False,
        direct_prediction: bool = True,
        avg_num_nodes: float = 1.0,
        d: Optional[int] = None,
    ):
        super().__init__()
        if d is not None:
            embed_dim = d
        if not direct_prediction:
            raise ValueError(
                "StructureTransformer currently supports direct_prediction only"
            )
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        valid_position_representations = {"cartesian", "fractional"}
        if position_representation not in valid_position_representations:
            raise ValueError(
                "position_representation must be one of "
                f"{sorted(valid_position_representations)}, "
                f"got {position_representation!r}"
            )
        valid_atom_orderings = {"none", "position_ids", "permute"}
        if atom_ordering not in valid_atom_orderings:
            raise ValueError(
                "atom_ordering must be one of "
                f"{sorted(valid_atom_orderings)}, got {atom_ordering!r}"
            )
        valid_coordinate_encodings = {"absolute_mlp", "v37_torus_relative"}
        if coordinate_encoding not in valid_coordinate_encodings:
            raise ValueError(
                "coordinate_encoding must be one of "
                f"{sorted(valid_coordinate_encodings)}, got {coordinate_encoding!r}"
            )
        valid_atom_embedding_types = {"embedding", "scalar"}
        if atom_embedding_type not in valid_atom_embedding_types:
            raise ValueError(
                "atom_embedding_type must be one of "
                f"{sorted(valid_atom_embedding_types)}, got {atom_embedding_type!r}"
            )
        if use_rotary_embeddings and use_periodic_rope:
            raise ValueError(
                "use_rotary_embeddings and use_periodic_rope are mutually exclusive"
            )
        if (
            coordinate_encoding == "v37_torus_relative" or use_periodic_rope
        ) and atom_ordering != "none":
            raise ValueError(
                "v37_torus_relative coordinates and periodic RoPE require "
                "atom_ordering='none' to avoid index/order canonicalization"
            )
        if coord_num_harmonics < 1:
            raise ValueError("coord_num_harmonics must be >= 1")
        if coord_encoder_chunk_size is not None and coord_encoder_chunk_size <= 0:
            raise ValueError("coord_encoder_chunk_size must be positive when provided")
        if fractional_wrap_eps < 0.0:
            raise ValueError("fractional_wrap_eps must be non-negative")
        if atom_embedding_type == "scalar" and embed_dim % 2 != 0:
            raise ValueError("scalar atom embedding requires an even embed_dim")
        if atomic_number_scale <= 0.0:
            raise ValueError("atomic_number_scale must be positive")
        if atom_scalar_embedding_scale <= 0.0:
            raise ValueError("atom_scalar_embedding_scale must be positive")
        if edge_vector_head:
            if edge_vector_head_cutoff <= 0.0:
                raise ValueError("edge_vector_head_cutoff must be positive")
            if edge_vector_head_hidden_dim <= 0:
                raise ValueError("edge_vector_head_hidden_dim must be positive")
            if edge_vector_head_num_radial_basis <= 0:
                raise ValueError("edge_vector_head_num_radial_basis must be positive")

        self.max_num_elements = max_num_elements
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.position_scale = position_scale
        self.position_representation = position_representation
        self.coordinate_encoding = coordinate_encoding
        self.coord_num_harmonics = coord_num_harmonics
        self.coord_encoder_chunk_size = coord_encoder_chunk_size
        self.coord_encoder_use_checkpoint = coord_encoder_use_checkpoint
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_periodic_rope = use_periodic_rope
        self.atom_ordering = atom_ordering
        self.center_positions = center_positions
        self.fractional_wrap_eps = float(fractional_wrap_eps)
        self.atom_embedding_type = atom_embedding_type
        self.atomic_number_scale = float(atomic_number_scale)
        self.atom_scalar_embedding_scale = float(atom_scalar_embedding_scale)
        self.include_cell_energy = include_cell_energy
        self.include_cell_stress = include_cell_stress
        self.regress_forces = regress_forces
        self.regress_stress = regress_stress
        self.edge_vector_head = edge_vector_head
        self.edge_vector_head_cutoff = float(edge_vector_head_cutoff)
        self.edge_vector_head_hidden_dim = edge_vector_head_hidden_dim
        self.edge_vector_head_num_radial_basis = edge_vector_head_num_radial_basis
        self.edge_vector_head_replace_direct = edge_vector_head_replace_direct
        self.avg_num_nodes = avg_num_nodes

        if encoder_hidden_dim is None:
            encoder_hidden_dim = int(1.5 * embed_dim)
        if attn_dropout is None:
            attn_dropout = dropout
        if residual_dropout is None:
            residual_dropout = dropout
        if mlp_dropout is None:
            mlp_dropout = dropout

        if self.atom_embedding_type == "embedding":
            self.atom_embedding = nn.Embedding(
                max_num_elements, embed_dim, padding_idx=0
            )
        else:
            self.atom_scalar_encoder = _build_mlp(
                embed_dim,
                encoder_hidden_dim,
                embed_dim,
                dropout,
            )
        if self.coordinate_encoding == "v37_torus_relative":
            self.position_encoder = TorusRelativeCoordEncoder(
                embed_dim=embed_dim,
                hidden_dim=encoder_hidden_dim,
                num_harmonics=coord_num_harmonics,
                wrap_eps=self.fractional_wrap_eps,
                chunk_size=coord_encoder_chunk_size,
                use_checkpoint=coord_encoder_use_checkpoint,
            )
        else:
            self.position_encoder = _build_mlp(
                3, encoder_hidden_dim, embed_dim, dropout
            )
        self.cell_token_encoder = _build_mlp(9, encoder_hidden_dim, embed_dim, dropout)
        self.cell_condition_encoder = _build_mlp(
            9, encoder_hidden_dim, embed_dim, dropout
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    residual_dropout=residual_dropout,
                    mlp_hidden_dim=mlp_hidden_dim,
                    mlp_dropout=mlp_dropout,
                    use_rotary_embeddings=use_rotary_embeddings,
                    use_periodic_rope=use_periodic_rope,
                    fractional_wrap_eps=self.fractional_wrap_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(embed_dim)

        self.atom_energy_head = _build_mlp(embed_dim, encoder_hidden_dim, 1, dropout)
        self.cell_energy_head = _build_mlp(embed_dim, encoder_hidden_dim, 1, dropout)

        if self.regress_forces:
            self.force_head = _build_mlp(embed_dim, encoder_hidden_dim, 3, dropout)
        if self.regress_stress:
            self.atom_stress_head = _build_mlp(
                embed_dim, encoder_hidden_dim, 9, dropout
            )
            self.cell_stress_head = _build_mlp(
                embed_dim, encoder_hidden_dim, 9, dropout
            )
        if self.edge_vector_head:
            radial_centers = torch.linspace(
                0.0,
                self.edge_vector_head_cutoff,
                self.edge_vector_head_num_radial_basis,
            )
            self.register_buffer("edge_radial_centers", radial_centers)
            self.edge_radial_width = self.edge_vector_head_cutoff / max(
                self.edge_vector_head_num_radial_basis - 1,
                1,
            )
            self.edge_feature_projection = nn.Linear(
                embed_dim,
                self.edge_vector_head_hidden_dim,
            )
            edge_input_dim = (
                2 * self.edge_vector_head_hidden_dim
                + self.edge_vector_head_num_radial_basis
            )
            if self.regress_forces:
                self.edge_force_head = _build_mlp(
                    edge_input_dim,
                    self.edge_vector_head_hidden_dim,
                    1,
                    dropout,
                )
            if self.regress_stress:
                self.edge_stress_head = _build_mlp(
                    edge_input_dim,
                    self.edge_vector_head_hidden_dim,
                    1,
                    dropout,
                )

    def _encode_atoms(
        self,
        atomic_numbers_dense: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.atom_embedding_type == "embedding":
            return self.atom_embedding(atomic_numbers_dense)
        scaled_atomic_numbers = atomic_numbers_dense.to(dtype=dtype)
        scaled_atomic_numbers = scaled_atomic_numbers / self.atomic_number_scale
        atom_embedding = _scalar_embedding(
            scaled_atomic_numbers,
            self.embed_dim,
            self.atom_scalar_embedding_scale,
        )
        return self.atom_scalar_encoder(atom_embedding)

    def _check_edge_inputs(
        self,
        edge_vectors: Optional[torch.Tensor],
        edge_centers: Optional[torch.Tensor],
        edge_neighbors: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_vectors is None or edge_centers is None or edge_neighbors is None:
            raise ValueError(
                "edge_vector_head=True requires neighbor-list edge vectors, centers, "
                "and neighbors"
            )
        return edge_vectors, edge_centers, edge_neighbors

    def _edge_features(
        self,
        atom_features: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_centers: torch.Tensor,
        edge_neighbors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_vectors = edge_vectors.to(dtype=atom_features.dtype)
        edge_lengths = torch.linalg.norm(edge_vectors, dim=-1).clamp_min(1.0e-12)
        unit_vectors = edge_vectors / edge_lengths.unsqueeze(-1)
        centers = self.edge_radial_centers.to(
            device=edge_vectors.device,
            dtype=edge_vectors.dtype,
        )
        radial = torch.exp(
            -0.5
            * ((edge_lengths.unsqueeze(-1) - centers) / self.edge_radial_width).pow(2)
        )
        projected_atoms = self.edge_feature_projection(atom_features)
        center_features = projected_atoms.index_select(0, edge_centers)
        neighbor_features = projected_atoms.index_select(0, edge_neighbors)
        features = torch.cat((center_features, neighbor_features, radial), dim=-1)
        return features, unit_vectors, edge_lengths

    def _edge_force_readout(
        self,
        atom_features: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_centers: torch.Tensor,
        edge_neighbors: torch.Tensor,
    ) -> torch.Tensor:
        forces = atom_features.new_zeros((atom_features.shape[0], 3))
        if edge_vectors.shape[0] == 0:
            return forces

        edge_features, unit_vectors, _ = self._edge_features(
            atom_features,
            edge_vectors,
            edge_centers,
            edge_neighbors,
        )
        edge_force = self.edge_force_head(edge_features).squeeze(-1)
        forces.index_add_(0, edge_centers, edge_force.unsqueeze(-1) * unit_vectors)
        return forces

    def _edge_stress_readout(
        self,
        atom_features: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_centers: torch.Tensor,
        edge_neighbors: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        stress = atom_features.new_zeros((batch_size, 3, 3))
        if edge_vectors.shape[0] == 0:
            return stress.reshape(batch_size, 9)

        edge_features, unit_vectors, _ = self._edge_features(
            atom_features,
            edge_vectors,
            edge_centers,
            edge_neighbors,
        )
        edge_stress = self.edge_stress_head(edge_features).squeeze(-1)
        dyads = unit_vectors.unsqueeze(-1) * unit_vectors.unsqueeze(-2)
        edge_systems = batch.index_select(0, edge_centers)
        stress.index_add_(0, edge_systems, edge_stress.view(-1, 1, 1) * dyads)

        edge_counts = atom_features.new_zeros(batch_size)
        edge_counts.index_add_(
            0,
            edge_systems,
            atom_features.new_ones(edge_systems.shape[0]),
        )
        stress = stress / edge_counts.clamp_min(1.0).view(-1, 1, 1)
        return stress.reshape(batch_size, 9)

    def _dense_inputs(
        self,
        data: TransformerData,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        atomic_numbers = data.atomic_numbers.long()
        if atomic_numbers.numel() > 0 and atomic_numbers.min() <= 0:
            raise ValueError(
                "StructureTransformer expects atomic_numbers to be real elements >= 1; "
                "0 is reserved for padding."
            )
        batch = data.batch.long()

        cell = data.cell
        if cell.dim() == 2 and cell.shape == (3, 3):
            cell = cell.unsqueeze(0)
        batch_size = cell.shape[0]

        pos_dense, atom_mask = _to_dense_batch(
            data.pos, batch, batch_size=batch_size
        )
        atomic_numbers_dense, _ = _to_dense_batch(
            atomic_numbers,
            batch,
            fill_value=0.0,
            batch_size=batch_size,
        )
        atomic_numbers_dense = atomic_numbers_dense.long()

        cell = cell.reshape(batch_size, 3, 3)

        needs_fractional = (
            self.position_representation == "fractional"
            or self.atom_ordering != "none"
            or self.coordinate_encoding == "v37_torus_relative"
            or self.use_periodic_rope
        )
        frac_pos_dense: Optional[torch.Tensor] = None
        if needs_fractional:
            frac_pos_dense = cartesian_to_fractional_dense(pos_dense, cell)

        if self.position_representation == "fractional":
            if frac_pos_dense is None:
                raise RuntimeError("fractional coordinates were not computed")
            pos_dense = frac_pos_dense

        atom_index_order: Optional[torch.Tensor] = None
        inverse_atom_index_order: Optional[torch.Tensor] = None
        rotary_position_ids: Optional[torch.Tensor] = None
        output_atom_mask = atom_mask
        if self.atom_ordering != "none":
            if frac_pos_dense is None:
                raise RuntimeError("atom ordering requires fractional coordinates")
            atom_index_order = _atom_reordering(frac_pos_dense, atom_mask)
            inverse_atom_index_order = _invert_permutation(atom_index_order)

            if self.atom_ordering == "permute":
                pos_dense = _gather_atom_features(pos_dense, atom_index_order)
                frac_pos_dense = _gather_atom_features(frac_pos_dense, atom_index_order)
                atomic_numbers_dense = torch.gather(
                    atomic_numbers_dense,
                    dim=1,
                    index=atom_index_order,
                )
                atom_mask = torch.gather(atom_mask, dim=1, index=atom_index_order)
            elif self.atom_ordering == "position_ids":
                atom_ranks = inverse_atom_index_order + 1
                atom_ranks = atom_ranks.masked_fill(~atom_mask, 0)
                cell_ranks = torch.zeros(
                    batch_size,
                    1,
                    device=atom_ranks.device,
                    dtype=atom_ranks.dtype,
                )
                rotary_position_ids = torch.cat((cell_ranks, atom_ranks), dim=1)

        if self.center_positions:
            mask = atom_mask.unsqueeze(-1).to(dtype=pos_dense.dtype)
            count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            center = (pos_dense * mask).sum(dim=1, keepdim=True) / count
            pos_dense = (pos_dense - center) * mask

        pos_dense = pos_dense / self.position_scale
        if frac_pos_dense is None:
            frac_pos_dense = pos_dense.new_zeros(pos_dense.shape)
        else:
            frac_pos_dense = _canonical_frac_coords(
                frac_pos_dense,
                self.fractional_wrap_eps,
            )
        cell_flat = cell.reshape(batch_size, 9) / self.position_scale

        return (
            pos_dense,
            frac_pos_dense,
            atomic_numbers_dense,
            atom_mask,
            output_atom_mask,
            cell_flat,
            rotary_position_ids,
            inverse_atom_index_order if self.atom_ordering == "permute" else None,
        )

    def forward(
        self,
        data: TransformerData,
        edge_vectors: Optional[torch.Tensor] = None,
        edge_centers: Optional[torch.Tensor] = None,
        edge_neighbors: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        (
            pos_dense,
            frac_pos_dense,
            atomic_numbers_dense,
            atom_mask,
            output_atom_mask,
            cell_flat,
            rotary_position_ids,
            inverse_atom_index_order,
        ) = self._dense_inputs(data)
        batch_size, max_atoms, _ = pos_dense.shape

        atom_identity = self._encode_atoms(atomic_numbers_dense, pos_dense.dtype)
        if self.coordinate_encoding == "v37_torus_relative":
            atom_positions = self.position_encoder(
                frac_pos_dense,
                atom_identity,
                atom_mask,
            )
        else:
            atom_positions = self.position_encoder(pos_dense)
        atom_tokens = atom_positions + atom_identity
        atom_tokens = atom_tokens * atom_mask.unsqueeze(-1).to(dtype=atom_tokens.dtype)

        cell_token = self.cell_token_encoder(cell_flat).unsqueeze(1)
        tokens = torch.cat((cell_token, atom_tokens), dim=1)

        cell_mask = torch.ones(batch_size, 1, device=atom_mask.device, dtype=torch.bool)
        valid_mask = torch.cat((cell_mask, atom_mask), dim=1)
        cell_condition = self.cell_condition_encoder(cell_flat).unsqueeze(1)
        rotary_pe_cplx: Optional[torch.Tensor] = None
        if self.use_rotary_embeddings:
            rotary_pe_cplx = _build_complex_pe(
                tokens.shape[1],
                self.embed_dim // self.num_heads,
                tokens.device,
            )

        for block in self.blocks:
            tokens = tokens + cell_condition
            tokens = block(
                tokens,
                valid_mask,
                rotary_pe_cplx,
                rotary_position_ids,
                frac_pos_dense if self.use_periodic_rope else None,
            )

        tokens = self.norm(tokens)
        cell_features = tokens[:, 0, :]
        atom_features = tokens[:, 1 : max_atoms + 1, :]
        if inverse_atom_index_order is not None:
            atom_features = _gather_atom_features(
                atom_features, inverse_atom_index_order
            )

        atom_energy = self.atom_energy_head(atom_features).squeeze(-1)
        atom_energy = atom_energy * output_atom_mask.to(dtype=atom_energy.dtype)
        energy = atom_energy.sum(dim=1) / self.avg_num_nodes
        if self.include_cell_energy:
            energy = energy + self.cell_energy_head(cell_features).squeeze(-1)

        outputs = {"energy": energy}
        if self.regress_forces:
            forces = self.force_head(atom_features)[output_atom_mask]
            if self.edge_vector_head:
                edge_vectors, edge_centers, edge_neighbors = self._check_edge_inputs(
                    edge_vectors,
                    edge_centers,
                    edge_neighbors,
                )
                edge_forces = self._edge_force_readout(
                    atom_features[output_atom_mask],
                    edge_vectors,
                    edge_centers,
                    edge_neighbors,
                )
                if self.edge_vector_head_replace_direct:
                    forces = edge_forces
                else:
                    forces = forces + edge_forces
            outputs["forces"] = forces
        if self.regress_stress:
            atom_stress = self.atom_stress_head(atom_features)
            stress_mask = output_atom_mask.unsqueeze(-1).to(dtype=atom_stress.dtype)
            num_atoms = stress_mask.sum(dim=1).clamp_min(1.0)
            stress = (atom_stress * stress_mask).sum(dim=1) / num_atoms
            if self.include_cell_stress:
                stress = stress + self.cell_stress_head(cell_features)
            if self.edge_vector_head:
                edge_vectors, edge_centers, edge_neighbors = self._check_edge_inputs(
                    edge_vectors,
                    edge_centers,
                    edge_neighbors,
                )
                edge_stress = self._edge_stress_readout(
                    atom_features[output_atom_mask],
                    edge_vectors,
                    edge_centers,
                    edge_neighbors,
                    data.batch.long(),
                    batch_size,
                )
                if self.edge_vector_head_replace_direct:
                    stress = edge_stress
                else:
                    stress = stress + edge_stress
            outputs["stress"] = stress
        return outputs

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        no_wd = set()
        for name, module in self.named_modules():
            if isinstance(module, (nn.Embedding, RMSNorm, nn.LayerNorm)):
                for parameter_name, _ in module.named_parameters(recurse=False):
                    no_wd.add(f"{name}.{parameter_name}" if name else parameter_name)

        for name, _ in self.named_parameters():
            if name.endswith("bias"):
                no_wd.add(name)
        return no_wd

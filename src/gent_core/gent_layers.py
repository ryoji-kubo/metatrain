"""Generalized Transformer layers adapted from Shao Yanming and Xavier Bresson.

Source: ``structure_transformer_core/gent_layers_original.py`` (Sep 10, 2026).
The three attention paths and multiplicative node update follow that reference.
RoPE is optional: atomistic callers omit sequence positions to preserve permutation
equivariance. Padding is zeroed after residuals, including fully masked rows.
Only PyTorch is required; the source file is not imported at runtime.
"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate low precision inputs in float32, but retain float64 accuracy.
        work = x if x.dtype == torch.float64 else x.float()
        normalized = work * torch.rsqrt(work.square().mean(-1, True) + self.eps)
        return self.weight * normalized.to(x.dtype)


class MLP_SiLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear3 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear3(F.silu(self.linear1(x)) * self.linear2(x))


def precompute_complex_pe(len_max: int, hidden_dim: int) -> torch.Tensor:
    """Return the reference sequence-index RoPE phases, initially on the CPU."""
    k = torch.arange(hidden_dim // 2).float()
    angles = torch.outer(torch.arange(len_max).float(), 10000 ** (-2 * k / hidden_dim))
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_pe(x: torch.Tensor, pe_cplx: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels; leave an odd final channel unchanged."""
    width = (x.shape[-1] // 2) * 2
    paired = x[..., :width].reshape(x.shape[0], x.shape[1], x.shape[2], width // 2, 2)
    phases = pe_cplx[: x.shape[2], : width // 2].to(x.device)
    real = phases.real.to(x.dtype)
    imag = phases.imag.to(x.dtype)
    rotated = torch.stack(
        (
            paired[..., 0] * real - paired[..., 1] * imag,
            paired[..., 0] * imag + paired[..., 1] * real,
        ),
        dim=-1,
    ).flatten(-2)
    return torch.cat((rotated, x[..., width:]), dim=-1)


def apply_rotary_pe_edge(
    e: torch.Tensor, pe_cplx: torch.Tensor, dim_to_rotate: str
) -> torch.Tensor:
    """Apply node-index RoPE along either endpoint of a dense edge tensor."""
    bs, heads, ni, nj, d = e.shape
    if dim_to_rotate == "i":
        flat = e.permute(0, 3, 1, 2, 4).reshape(bs * nj, heads, ni, d)
        return (
            apply_rotary_pe(flat, pe_cplx)
            .reshape(bs, nj, heads, ni, d)
            .permute(0, 2, 3, 1, 4)
        )
    elif dim_to_rotate == "j":
        flat = e.permute(0, 2, 1, 3, 4).reshape(bs * ni, heads, nj, d)
        return (
            apply_rotary_pe(flat, pe_cplx)
            .reshape(bs, ni, heads, nj, d)
            .permute(0, 2, 1, 3, 4)
        )
    raise ValueError("dim_to_rotate must be 'i' or 'j'")


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize valid entries without NaNs in fully masked rows or their gradients."""
    scores = scores.masked_fill(~mask, float("-inf"))
    scores = torch.where(
        mask.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores)
    )
    return torch.softmax(scores, dim=-1).masked_fill(~mask, 0.0)


class SA_RoPE_Layer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim_sa: int, num_heads: int, dropout_att: float
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_sa = hidden_dim_sa
        self.d_head = hidden_dim_sa // num_heads
        self.q_node_self = nn.Linear(input_dim, hidden_dim_sa, bias=False)
        self.k_node_self = nn.Linear(input_dim, hidden_dim_sa, bias=False)
        self.v_node_self = nn.Linear(input_dim, hidden_dim_sa, bias=False)
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        self.dropout_att = dropout_att
        self.proj_node = nn.Linear(hidden_dim_sa, input_dim, bias=False)

    def forward(
        self, x: torch.Tensor, pe_cplx: Optional[torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        bs, n, _ = x.shape
        q = self.q_norm(
            self.q_node_self(x).view(bs, n, self.num_heads, self.d_head).transpose(1, 2)
        )
        k = self.k_norm(
            self.k_node_self(x).view(bs, n, self.num_heads, self.d_head).transpose(1, 2)
        )
        v = self.v_node_self(x).view(bs, n, self.num_heads, self.d_head).transpose(1, 2)
        if pe_cplx is not None:
            q = apply_rotary_pe(q, pe_cplx)
            k = apply_rotary_pe(k, pe_cplx)
        valid = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(1)
        # Explicit attention also supports double backward on CPU and CUDA.
        weights = masked_softmax(q @ k.transpose(-1, -2) / self.d_head**0.5, valid)
        weights = F.dropout(weights, self.dropout_att, self.training)
        result = (weights @ v).transpose(1, 2).reshape(bs, n, self.d_sa)
        return self.proj_node(result).masked_fill(~mask.unsqueeze(-1), 0.0)


class CAn_RoPE_Layer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim_ca_n: int, num_heads: int, dropout_att: float
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_ca_n = hidden_dim_ca_n
        self.d_head = hidden_dim_ca_n // num_heads
        self.q_node_cross = nn.Linear(input_dim, hidden_dim_ca_n, bias=False)
        self.k_edge_cross = nn.Linear(input_dim, hidden_dim_ca_n, bias=False)
        self.v_edge_cross = nn.Linear(input_dim, hidden_dim_ca_n, bias=False)
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        self.drop_att = nn.Dropout(dropout_att)
        self.proj_node = nn.Linear(hidden_dim_ca_n, input_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        pe_cplx: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        bs, n, _ = x.shape
        q = self.q_norm(
            self.q_node_cross(x)
            .view(bs, n, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        k = self.k_norm(
            self.k_edge_cross(e)
            .view(bs, n, n, self.num_heads, self.d_head)
            .permute(0, 3, 1, 2, 4)
        )
        v = (
            self.v_edge_cross(e)
            .view(bs, n, n, self.num_heads, self.d_head)
            .permute(0, 3, 1, 2, 4)
        )
        if pe_cplx is not None:
            q = apply_rotary_pe(q, pe_cplx)
            k = apply_rotary_pe_edge(k, pe_cplx, "j")
        valid = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(1)
        scores = torch.einsum("bhid,bhijd->bhij", q, k) / self.d_head**0.5
        weights = self.drop_att(masked_softmax(scores, valid))
        result = torch.einsum("bhij,bhijd->bhid", weights, v)
        result = result.transpose(1, 2).reshape(bs, n, self.d_ca_n)
        return self.proj_node(result).masked_fill(~mask.unsqueeze(-1), 0.0)


class CAe_RoPE_Layer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim_ca_e: int, num_heads: int, dropout_att: float
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_ca_e = hidden_dim_ca_e
        self.d_head = hidden_dim_ca_e // num_heads
        self.q_edge_cross = nn.Linear(input_dim, hidden_dim_ca_e, bias=False)
        self.k_node_cross = nn.Linear(input_dim, hidden_dim_ca_e, bias=False)
        self.v_node_cross = nn.Linear(input_dim, hidden_dim_ca_e, bias=False)
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        self.drop_att = nn.Dropout(dropout_att)
        self.proj_edge = nn.Linear(hidden_dim_ca_e, input_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        pe_cplx: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        bs, n, _ = x.shape
        k = self.k_norm(
            self.k_node_cross(x)
            .view(bs, n, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self.v_node_cross(x)
            .view(bs, n, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        q = self.q_norm(
            self.q_edge_cross(e)
            .view(bs, n, n, self.num_heads, self.d_head)
            .permute(0, 3, 1, 2, 4)
        )
        qi, qj = q, q
        if pe_cplx is not None:
            k = apply_rotary_pe(k, pe_cplx)
            qi = apply_rotary_pe_edge(q, pe_cplx, "i")
            qj = apply_rotary_pe_edge(q, pe_cplx, "j")
        scores = (
            torch.stack(
                (
                    torch.einsum("bhijd,bhjd->bhij", qi, k),
                    torch.einsum("bhijd,bhid->bhij", qj, k),
                ),
                dim=-1,
            )
            / self.d_head**0.5
        )
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)
        weights = self.drop_att(masked_softmax(scores, valid[:, None, :, :, None]))
        result = weights[..., 0:1] * v.unsqueeze(2) + weights[..., 1:2] * v.unsqueeze(3)
        result = result.permute(0, 2, 3, 1, 4).reshape(bs, n, n, self.d_ca_e)
        return self.proj_edge(result).masked_fill(~valid.unsqueeze(-1), 0.0)


class GenTBlock(nn.Module):
    """Coupled node/edge attention with the reference multiplicative node gate."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_dim_sa: int,
        hidden_dim_ca_n: int,
        hidden_dim_ca_e: int,
        ffn_dim: int,
        num_heads: int,
        dropout_att: float,
        dropout_lnr: float,
    ):
        super().__init__()
        for width in (hidden_dim_sa, hidden_dim_ca_n, hidden_dim_ca_e):
            if num_heads <= 0 or width <= 0 or width % num_heads != 0:
                raise ValueError(
                    "Attention widths must be positive and divisible by num_heads"
                )
        if hidden_dim <= 0 or ffn_dim <= 0:
            raise ValueError("hidden_dim and ffn_dim must be positive")
        if not 0.0 <= dropout_att < 1.0 or not 0.0 <= dropout_lnr < 1.0:
            raise ValueError("Dropout probabilities must lie in [0, 1)")
        self.RMSNorm_node_x_in = RMSNorm(hidden_dim)
        self.RMSNorm_node_e_in = RMSNorm(hidden_dim)
        self.CAn_layer = CAn_RoPE_Layer(
            hidden_dim, hidden_dim_ca_n, num_heads, dropout_att
        )
        self.SA_layer = SA_RoPE_Layer(hidden_dim, hidden_dim_sa, num_heads, dropout_att)
        self.proj_SA = nn.Linear(hidden_dim, hidden_dim)
        self.proj_CAn = nn.Linear(hidden_dim, hidden_dim)
        self.RMSNorm_SA = RMSNorm(hidden_dim)
        self.RMSNorm_CAn = RMSNorm(hidden_dim)
        self.drop_x = nn.Dropout(dropout_lnr)
        self.RMSNorm_x_out = RMSNorm(hidden_dim)
        self.MLP_x_out = MLP_SiLU(hidden_dim, ffn_dim, hidden_dim)
        self.RMSNorm_edge_x_in = RMSNorm(hidden_dim)
        self.RMSNorm_edge_e_in = RMSNorm(hidden_dim)
        self.CAe_layer = CAe_RoPE_Layer(
            hidden_dim, hidden_dim_ca_e, num_heads, dropout_att
        )
        self.drop_e = nn.Dropout(dropout_lnr)
        self.RMSNorm_e_out = RMSNorm(hidden_dim)
        self.MLP_e_out = MLP_SiLU(hidden_dim, ffn_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        pe_cplx: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_invalid = ~mask.unsqueeze(-1)
        e_invalid = ~(mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
        x = x.masked_fill(x_invalid, 0.0)
        e = e.masked_fill(e_invalid, 0.0)
        xn, en = self.RMSNorm_node_x_in(x), self.RMSNorm_node_e_in(e)
        sa = self.SA_layer(xn, pe_cplx, mask)
        ca = self.CAn_layer(xn, en, pe_cplx, mask)
        mixed = self.proj_SA(self.RMSNorm_SA(sa)) * self.proj_CAn(self.RMSNorm_CAn(ca))
        x = x + self.drop_x(mixed)
        x = (x + self.MLP_x_out(self.RMSNorm_x_out(x))).masked_fill(x_invalid, 0.0)
        ca_edge = self.CAe_layer(
            self.RMSNorm_edge_x_in(x), self.RMSNorm_edge_e_in(e), pe_cplx, mask
        )
        e = e + self.drop_e(ca_edge)
        e = (e + self.MLP_e_out(self.RMSNorm_e_out(e))).masked_fill(e_invalid, 0.0)
        return x, e

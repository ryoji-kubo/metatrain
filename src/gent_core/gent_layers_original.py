"""
Generalized Transformer (GenT) layers, with support for dense graph attention,
graph size masking, and RoPE (Rotary Positional Encoding) integration.

Dependencies: Python 3.10.18, PyTorch 2.6.0+cu126
Version: Sep 10, 2026

Author: Shao Yanming, Xavier Bresson
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Root Mean Square (RMS) Layer Normalization
# ---------------------------------------------------------------------------

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square (RMS) Layer Normalization.
    """
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        norm_x_float = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        output = self.weight * norm_x_float.type_as(x) 
        return output


# ---------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP) Layers
# ---------------------------------------------------------------------------

class MLP_SiLU(nn.Module):
    """
    SiLU activation based Multi-Layer Perceptron (MLP) layer.
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else int(2.25 * input_dim)
        output_dim = output_dim if output_dim is not None else hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear3 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear3(F.silu(self.linear1(x)) * self.linear2(x))


# ---------------------------------------------------------------------------
# Rotary Positional Encoding (RoPE) Utilities
# ---------------------------------------------------------------------------

def precompute_complex_pe(len_max, hidden_dim):
    """
    Precompute complex positional embeddings for RoPE.
    """
    k = torch.arange(hidden_dim // 2).float()  # [d//2]
    omegas = 10000 ** (-2 * k / hidden_dim)
    pos = torch.arange(len_max)  # [len_max]
    angles = torch.outer(pos, omegas).float()  # [len_max, d//2]
    return torch.polar(torch.ones_like(angles), angles)


@torch._dynamo.disable()
def apply_rotary_pe(x, pe_cplx):
    """
    Apply RoPE to the input tensor.
    """
    bs, head_num, n, d = x.size()
    y = x.view(bs, head_num, n, d // 2, 2)
    z = torch.view_as_complex(y.float())
    z = z * pe_cplx[:n, :].view(1, 1, n, d // 2)
    y = torch.view_as_real(z)
    return y.view(bs, head_num, n, d).type_as(x)


def apply_rotary_pe_edge(e, pe_cplx, dim_to_rotate):
    """
    Apply RoPE to the edge features.
    """
    bs, head_num, n_i, n_j, d = e.size()  # [bs, head_num, n_i, n_j, d] & n_i = n_j
    if dim_to_rotate == "i":
        e_reshaped = e.permute(0, 3, 1, 2, 4).reshape(bs * n_j, head_num, n_i, d)
        e_rotated = apply_rotary_pe(e_reshaped, pe_cplx)
        return e_rotated.view(bs, n_j, head_num, n_i, d).permute(0, 2, 3, 1, 4)
    elif dim_to_rotate == "j":
        e_reshaped = e.permute(0, 2, 1, 3, 4).reshape(bs * n_i, head_num, n_j, d)
        e_rotated = apply_rotary_pe(e_reshaped, pe_cplx)
        return e_rotated.view(bs, n_i, head_num, n_j, d).permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Unsupported `dim_to_rotate`: {dim_to_rotate}")


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention from Node to Node Layers
# ---------------------------------------------------------------------------


class SA_RoPE_Layer(nn.Module):
    """
    Self-attention from node to node with RoPE.
    """
    def __init__(self, input_dim, hidden_dim_sa, num_heads, dropout_att):
        super().__init__()
        # Layer hyperparameters
        assert hidden_dim_sa % num_heads == 0, "`hidden_dim_sa` must be divisible by `num_heads`."
        self.d_in = input_dim; self.d_sa = hidden_dim_sa
        self.d_head = hidden_dim_sa // num_heads; self.num_heads = num_heads
        # Attention projection layers
        self.q_node_self = nn.Linear(input_dim, hidden_dim_sa, bias=False)
        self.k_node_self = nn.Linear(input_dim, hidden_dim_sa, bias=False)
        self.v_node_self = nn.Linear(input_dim, hidden_dim_sa, bias=False)
        self.dropout_att = dropout_att
        # Normalization layers
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        # Output projection
        self.proj_node = nn.Linear(hidden_dim_sa, input_dim, bias=False)

    def forward(self, x, pe_cplx, mask):
        bs, n, _ = x.size(); h = self.num_heads; d_h = self.d_head
        x_mask = mask.unsqueeze(-1).float()  # [bs, n, 1]
        # Q/K/V projections and RMSNorm
        q_ns = self.q_node_self(x).view(bs, n, h, d_h).transpose(1, 2)
        k_ns = self.k_node_self(x).view(bs, n, h, d_h).transpose(1, 2)
        v_ns = self.v_node_self(x).view(bs, n, h, d_h).transpose(1, 2)
        q_ns = self.q_norm(q_ns); k_ns = self.k_norm(k_ns)
        # RoPE application
        q_ns = apply_rotary_pe(q_ns, pe_cplx)
        k_ns = apply_rotary_pe(k_ns, pe_cplx)
        # Attention computation
        att_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(1)  # [bs, 1, n, n]
        x_sa = F.scaled_dot_product_attention(
            q_ns, k_ns, v_ns, attn_mask=att_mask,
            dropout_p=self.dropout_att if self.training else 0.0,
        ).nan_to_num(0.0)
        # Reshape and project output
        x_sa_concat = x_sa.transpose(1, 2).reshape(bs, n, self.d_sa)
        return self.proj_node(x_sa_concat) * x_mask


# ---------------------------------------------------------------------------
# Multi-Head Cross-Attention from Edge to Node Layers
# ---------------------------------------------------------------------------

class CAn_RoPE_Layer(nn.Module):
    """
    Cross-attention from edge to node with RoPE.
    """
    def __init__(self, input_dim, hidden_dim_ca_n, num_heads, dropout_att):
        super().__init__()
        # Layer hyperparameters
        assert hidden_dim_ca_n % num_heads == 0, "`hidden_dim_ca_n` must be divisible by `num_heads`."
        self.d_in = input_dim; self.d_ca_n = hidden_dim_ca_n
        self.d_head = hidden_dim_ca_n // num_heads; self.num_heads = num_heads
        self.sqrt_d = self.d_head ** 0.5
        # Attention projection layers
        self.q_node_cross = nn.Linear(input_dim, hidden_dim_ca_n, bias=False)
        self.k_edge_cross = nn.Linear(input_dim, hidden_dim_ca_n, bias=False)
        self.v_edge_cross = nn.Linear(input_dim, hidden_dim_ca_n, bias=False)
        self.drop_att = nn.Dropout(dropout_att)
        # Normalization layers
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        # Output projection
        self.proj_node = nn.Linear(hidden_dim_ca_n, input_dim, bias=False)

    def forward(self, x, e, pe_cplx, mask):
        bs, n, _ = x.size(); h = self.num_heads; d_h = self.d_head
        x_mask = mask.unsqueeze(-1).float()  # [bs, n, 1]
        e_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [bs, n, n]
        # Q/K/V projections and RMSNorm
        q_nc = self.q_node_cross(x).view(bs, n, h, d_h).transpose(1, 2)
        k_ec = self.k_edge_cross(e).view(bs, n, n, h, d_h).permute(0, 3, 1, 2, 4)
        v_ec = self.v_edge_cross(e).view(bs, n, n, h, d_h).permute(0, 3, 1, 2, 4)
        q_nc = self.q_norm(q_nc); k_ec = self.k_norm(k_ec)
        # RoPE application
        d_rope = (self.d_head // 2) * 2
        if d_rope > 0:
            pe_cplx_sliced = pe_cplx[:, : d_rope // 2]
            q_nc_rope = apply_rotary_pe(q_nc[..., :d_rope].contiguous(), pe_cplx_sliced)
            k_ec_rope = apply_rotary_pe_edge(k_ec[..., :d_rope].contiguous(), pe_cplx_sliced, dim_to_rotate="j")
            if d_rope < self.d_head:
                q_nc = torch.cat((q_nc_rope, q_nc[..., d_rope:]), dim=-1)
                k_ec = torch.cat((k_ec_rope, k_ec[..., d_rope:]), dim=-1)
            else:
                q_nc = q_nc_rope; k_ec = k_ec_rope
        # Attention computation
        scores_ca_e2n = torch.einsum("bhid,bhijd->bhij", q_nc, k_ec) / self.sqrt_d
        scores_ca_e2n.masked_fill_(~e_mask.unsqueeze(1), float("-inf"))
        softmax_ca_e2n = torch.softmax(scores_ca_e2n, dim=-1).nan_to_num(0.0)
        att_ca_e2n = self.drop_att(softmax_ca_e2n) if self.training else softmax_ca_e2n
        x_ca_e2n = torch.einsum("bhij,bhijd->bhid", att_ca_e2n, v_ec)
        # Reshape and project output
        x_ca_e2n_concat = x_ca_e2n.transpose(1, 2).reshape(bs, n, self.d_ca_n)
        return self.proj_node(x_ca_e2n_concat) * x_mask


# ---------------------------------------------------------------------------
# Multi-Head Cross-Attention from Node to Edge Layers
# ---------------------------------------------------------------------------

class CAe_RoPE_Layer(nn.Module):
    """
    Cross-attention from node to edge with RoPE.
    """
    def __init__(self, input_dim, hidden_dim_ca_e, num_heads, dropout_att):
        super().__init__()
        # Layer hyperparameters
        assert hidden_dim_ca_e % num_heads == 0, "`hidden_dim_ca_e` must be divisible by `num_heads`."
        self.d_in = input_dim; self.d_ca_e = hidden_dim_ca_e
        self.d_head = hidden_dim_ca_e // num_heads; self.num_heads = num_heads
        self.sqrt_d = self.d_head ** 0.5
        # Attention projection layers
        self.q_edge_cross = nn.Linear(input_dim, hidden_dim_ca_e, bias=False)
        self.k_node_cross = nn.Linear(input_dim, hidden_dim_ca_e, bias=False)
        self.v_node_cross = nn.Linear(input_dim, hidden_dim_ca_e, bias=False)
        self.drop_att = nn.Dropout(dropout_att)
        # Normalization layers
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)
        # Output projection
        self.proj_edge = nn.Linear(hidden_dim_ca_e, input_dim, bias=False)

    def forward(self, x, e, pe_cplx, mask):
        bs, n, _ = x.size(); h = self.num_heads; d_h = self.d_head
        e_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [bs, n, n]
        # Q/K/V projections and RMSNorm
        k_nc = self.k_node_cross(x).view(bs, n, h, d_h).transpose(1, 2)
        v_nc = self.v_node_cross(x).view(bs, n, h, d_h).transpose(1, 2)
        q_ec = self.q_edge_cross(e).view(bs, n, n, h, d_h).permute(0, 3, 1, 2, 4)
        q_ec = self.q_norm(q_ec); k_nc = self.k_norm(k_nc)
        # RoPE application
        d_rope = (self.d_head // 2) * 2
        if d_rope > 0:
            pe_cplx_sliced = pe_cplx[:, : d_rope // 2]
            k_nc_rope = apply_rotary_pe(k_nc[..., :d_rope].contiguous(), pe_cplx_sliced)
            q_ec_rope_i = apply_rotary_pe_edge(q_ec[..., :d_rope].contiguous(), pe_cplx_sliced, dim_to_rotate="i")
            q_ec_rope_j = apply_rotary_pe_edge(q_ec[..., :d_rope].contiguous(), pe_cplx_sliced, dim_to_rotate="j")
            if d_rope < self.d_head:
                k_nc_rotated = torch.cat((k_nc_rope, k_nc[..., d_rope:]), dim=-1)
                q_ec_rotated_i = torch.cat((q_ec_rope_i, q_ec[..., d_rope:]), dim=-1)
                q_ec_rotated_j = torch.cat((q_ec_rope_j, q_ec[..., d_rope:]), dim=-1)
            else:
                k_nc_rotated = k_nc_rope
                q_ec_rotated_i = q_ec_rope_i
                q_ec_rotated_j = q_ec_rope_j
        else:
            k_nc_rotated = k_nc
            q_ec_rotated_i = q_ec
            q_ec_rotated_j = q_ec
        # Attention computation
        scores_i = torch.einsum("bhijd,bhjd->bhij", q_ec_rotated_i, k_nc_rotated) / self.sqrt_d
        scores_j = torch.einsum("bhijd,bhid->bhij", q_ec_rotated_j, k_nc_rotated) / self.sqrt_d
        scores_i.masked_fill_(~e_mask.unsqueeze(1), float("-inf"))
        scores_j.masked_fill_(~e_mask.unsqueeze(1), float("-inf"))
        scores_ca_n2e = torch.stack([scores_i, scores_j], dim=-1)
        softmax_ca_n2e = torch.softmax(scores_ca_n2e, dim=-1).nan_to_num(0.0)
        att_ca_n2e = self.drop_att(softmax_ca_n2e) if self.training else softmax_ca_n2e
        att_i = att_ca_n2e[..., 0:1]
        att_j = att_ca_n2e[..., 1:2]
        e_ca_n2e = att_i * v_nc.unsqueeze(2) + att_j * v_nc.unsqueeze(3)
        # Reshape and project output
        e_ca_n2e_concat = e_ca_n2e.permute(0, 2, 3, 1, 4).reshape(bs, n, n, self.d_ca_e)
        return self.proj_edge(e_ca_n2e_concat) * e_mask.unsqueeze(-1).float()


# ---------------------------------------------------------------------------
# Generalized Transformer Block
# ---------------------------------------------------------------------------

class GenT_RoPE_Block(nn.Module):
    def __init__(
        self, hidden_dim, hidden_dim_sa, hidden_dim_ca_n, hidden_dim_ca_e,
        ffn_dim, num_heads, dropout_att, dropout_lnr,
    ):
        super().__init__()
        # Node attention computation
        self.RMSNorm_node_x_in = RMSNorm(hidden_dim)
        self.RMSNorm_node_e_in = RMSNorm(hidden_dim)
        self.CAn_layer = CAn_RoPE_Layer(
            input_dim=hidden_dim,
            hidden_dim_ca_n=hidden_dim_ca_n,
            num_heads=num_heads,
            dropout_att=dropout_att,
        )
        self.SA_layer = SA_RoPE_Layer(
            input_dim=hidden_dim,
            hidden_dim_sa=hidden_dim_sa,
            num_heads=num_heads,
            dropout_att=dropout_att,
        )
        self.proj_SA = nn.Linear(hidden_dim, hidden_dim)
        self.proj_CAn = nn.Linear(hidden_dim, hidden_dim)
        self.RMSNorm_SA = RMSNorm(hidden_dim)
        self.RMSNorm_CAn = RMSNorm(hidden_dim)
        self.drop_x = nn.Dropout(dropout_lnr)
        self.RMSNorm_x_out = RMSNorm(hidden_dim)
        self.MLP_x_out = MLP_SiLU(
            input_dim=hidden_dim,
            hidden_dim=ffn_dim,
            output_dim=hidden_dim,
        )
        # Edge attention computation
        self.RMSNorm_edge_x_in = RMSNorm(hidden_dim)
        self.RMSNorm_edge_e_in = RMSNorm(hidden_dim)
        self.CAe_layer = CAe_RoPE_Layer(
            input_dim=hidden_dim,
            hidden_dim_ca_e=hidden_dim_ca_e,
            num_heads=num_heads,
            dropout_att=dropout_att,
        )
        self.drop_e = nn.Dropout(dropout_lnr)
        self.RMSNorm_e_out = RMSNorm(hidden_dim)
        self.MLP_e_out = MLP_SiLU(
            input_dim=hidden_dim,
            hidden_dim=ffn_dim,
            output_dim=hidden_dim,
        )

    def forward(self, x, e, pe_cplx, mask):
        x_mask = mask.unsqueeze(-1).float()  # [bs, n, 1]
        e_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).float()  # [bs, n, n, 1]
        # Node attention computation
        x_norm = self.RMSNorm_node_x_in(x)
        e_norm = self.RMSNorm_node_e_in(e)
        x_SA = self.SA_layer(x_norm, pe_cplx, mask)
        x_CAn = self.CAn_layer(x_norm, e_norm, pe_cplx, mask)
        x_SA_proj = self.proj_SA(self.RMSNorm_SA(x_SA))
        x_CAn_proj = self.proj_CAn(self.RMSNorm_CAn(x_CAn))
        x_mix = x_SA_proj * x_CAn_proj
        x = x + self.drop_x(x_mix)
        x_norm = self.RMSNorm_x_out(x)
        x = x + self.MLP_x_out(x_norm) * x_mask
        # Edge attention computation
        x_norm = self.RMSNorm_edge_x_in(x)
        e_norm = self.RMSNorm_edge_e_in(e)
        e_CAe = self.CAe_layer(x_norm, e_norm, pe_cplx, mask)
        e = e + self.drop_e(e_CAe)
        e_norm = self.RMSNorm_e_out(e)
        e = e + self.MLP_e_out(e_norm) * e_mask
        return x, e

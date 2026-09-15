from __future__ import annotations

import math
from typing import Optional

import torch


DEFAULT_MIN_PROBE_CUTOFF = 0.5
DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH = 1.0


def cutoff_func_bump(
    values: torch.Tensor,
    cutoff: torch.Tensor,
    width: float,
) -> torch.Tensor:
    """Smooth bump cutoff that is 1 inside and 0 outside the cutoff shell."""
    scaled_values = (values - (cutoff - width)) / width
    clamped = scaled_values.clamp(1.0e-6, 1.0 - 1.0e-6)
    return 0.5 * (1.0 + torch.tanh(1.0 / torch.tan(torch.pi * clamped)))


def cutoff_func_cosine(
    values: torch.Tensor,
    cutoff: torch.Tensor,
    width: float,
) -> torch.Tensor:
    """Cosine cutoff that is 1 inside and 0 outside the cutoff shell."""
    scaled_values = (values - (cutoff - width)) / width
    clamped = scaled_values.clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(torch.pi * clamped))


def _n_total(
    r_per_atom: torch.Tensor,
    edge_distances: torch.Tensor,
    centers: torch.Tensor,
    num_nodes: int,
    cutoff_width: float,
    inv_max_cutoff: float,
    num_neighbors_adaptive: float,
) -> torch.Tensor:
    per_edge = cutoff_func_bump(edge_distances, r_per_atom[centers], cutoff_width)
    n = torch.zeros(num_nodes, dtype=edge_distances.dtype, device=edge_distances.device)
    n.index_add_(0, centers, per_edge)
    x = r_per_atom * inv_max_cutoff
    return n + num_neighbors_adaptive * x.pow(3)


def _n_total_and_dn_dr(
    r_per_atom: torch.Tensor,
    edge_distances: torch.Tensor,
    centers: torch.Tensor,
    num_nodes: int,
    cutoff_width: float,
    inv_max_cutoff: float,
    num_neighbors_adaptive: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r_per_edge = r_per_atom[centers]
    scaled = (edge_distances - (r_per_edge - cutoff_width)) / cutoff_width
    active = (scaled > 0.0) & (scaled < 1.0)
    smaller = scaled <= 0.0

    safe = scaled.clamp(1.0e-6, 1.0 - 1.0e-6)
    s = math.pi * safe
    sin_s = torch.sin(s)
    cot_s = torch.cos(s) / sin_s
    tanh_cot = torch.tanh(cot_s)

    f_active = 0.5 * (1.0 + tanh_cot)
    f = torch.where(active, f_active, smaller.to(scaled.dtype))

    sech_sq = 1.0 - tanh_cot * tanh_cot
    df_dr = ((0.5 * math.pi / cutoff_width) * sech_sq / (sin_s * sin_s)) * active.to(
        scaled.dtype
    )

    n = torch.zeros(num_nodes, dtype=edge_distances.dtype, device=edge_distances.device)
    n.index_add_(0, centers, f)
    dn = torch.zeros(
        num_nodes,
        dtype=edge_distances.dtype,
        device=edge_distances.device,
    )
    dn.index_add_(0, centers, df_dr)

    x = r_per_atom * inv_max_cutoff
    n = n + num_neighbors_adaptive * x.pow(3)
    dn = dn + 3.0 * num_neighbors_adaptive * x.pow(2) * inv_max_cutoff

    return n, dn


def get_adaptive_cutoffs_solver(
    centers: torch.Tensor,
    edge_distances: torch.Tensor,
    num_neighbors_adaptive: float,
    num_nodes: int,
    max_cutoff: float,
    cutoff_width: float = DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH,
) -> torch.Tensor:
    """Adaptive per-atom cutoffs from a Newton-bisection root solve."""
    edge_distances_d = edge_distances.detach()
    inv_max_cutoff = 1.0 / max_cutoff

    r_lo = torch.zeros(
        num_nodes,
        dtype=edge_distances.dtype,
        device=edge_distances.device,
    )
    r_hi = torch.full(
        (num_nodes,),
        max_cutoff,
        dtype=edge_distances.dtype,
        device=edge_distances.device,
    )
    r = 0.5 * r_hi
    for _ in range(10):
        n, dn = _n_total_and_dn_dr(
            r,
            edge_distances_d,
            centers,
            num_nodes,
            cutoff_width,
            inv_max_cutoff,
            num_neighbors_adaptive,
        )
        f = n - num_neighbors_adaptive
        below = f <= 0
        r_lo = torch.where(below, r, r_lo)
        r_hi = torch.where(below, r_hi, r)
        r_newton = r - f / dn.clamp_min(1.0e-6)
        inside = (r_newton >= r_lo) & (r_newton <= r_hi)
        r_mid = 0.5 * (r_lo + r_hi)
        r = torch.where(inside, r_newton, r_mid)

    _, dn_root = _n_total_and_dn_dr(
        r,
        edge_distances_d,
        centers,
        num_nodes,
        cutoff_width,
        inv_max_cutoff,
        num_neighbors_adaptive,
    )
    n_residual = (
        _n_total(
            r,
            edge_distances,
            centers,
            num_nodes,
            cutoff_width,
            inv_max_cutoff,
            num_neighbors_adaptive,
        )
        - num_neighbors_adaptive
    )
    min_cutoff_factor = 1.0 / 16.0
    return (r - n_residual / dn_root.clamp_min(1.0e-6)).clamp(
        max_cutoff * min_cutoff_factor,
        max_cutoff,
    )


def get_effective_num_neighbors(
    edge_distances: torch.Tensor,
    probe_cutoffs: torch.Tensor,
    centers: torch.Tensor,
    num_nodes: int,
    width: float = DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH,
) -> torch.Tensor:
    """Effective smooth neighbor count for each atom and probe cutoff."""
    weights = cutoff_func_bump(
        edge_distances.unsqueeze(0),
        probe_cutoffs.unsqueeze(1),
        width,
    )
    probe_num_neighbors = torch.zeros(
        (len(probe_cutoffs), num_nodes),
        dtype=edge_distances.dtype,
        device=edge_distances.device,
    )
    probe_num_neighbors.index_add_(1, centers, weights)
    return probe_num_neighbors.T


def get_gaussian_cutoff_weights(
    effective_num_neighbors: torch.Tensor,
    num_neighbors_adaptive: float,
    width: Optional[float] = None,
) -> torch.Tensor:
    """Gaussian weights over probe cutoffs centered at the target neighbor count."""
    if effective_num_neighbors.numel() == 0:
        return torch.empty_like(effective_num_neighbors)

    diff = effective_num_neighbors - num_neighbors_adaptive
    x = torch.linspace(
        0,
        1,
        effective_num_neighbors.shape[1],
        device=effective_num_neighbors.device,
        dtype=effective_num_neighbors.dtype,
    )
    diff = diff + num_neighbors_adaptive * x.pow(3).unsqueeze(0)
    if width is None:
        eps = 1.0e-12
        if diff.shape[-1] == 1:
            width_t = diff.abs() * 0.5 + eps
        else:
            (width_t,) = torch.gradient(diff, dim=-1)
            width_t = width_t.abs().clamp_min(eps)
    else:
        width_t = torch.ones_like(diff) * width

    logw = -0.5 * (diff / width_t).pow(2)
    weights = torch.exp(logw - logw.max())
    return weights / weights.sum(dim=1, keepdim=True)


def get_adaptive_cutoffs_grid(
    centers: torch.Tensor,
    edge_distances: torch.Tensor,
    num_neighbors_adaptive: float,
    num_nodes: int,
    max_cutoff: float,
    min_cutoff: float = DEFAULT_MIN_PROBE_CUTOFF,
    cutoff_width: float = DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH,
    probe_spacing: Optional[float] = None,
    weight_width: Optional[float] = None,
) -> torch.Tensor:
    """Adaptive per-atom cutoffs from a discrete probe-cutoff grid."""
    if probe_spacing is None:
        probe_spacing = cutoff_width / 4.0
    probe_cutoffs = torch.arange(
        min_cutoff,
        max_cutoff,
        probe_spacing,
        device=edge_distances.device,
        dtype=edge_distances.dtype,
    )
    effective_num_neighbors = get_effective_num_neighbors(
        edge_distances,
        probe_cutoffs,
        centers,
        num_nodes,
        width=cutoff_width,
    )
    cutoffs_weights = get_gaussian_cutoff_weights(
        effective_num_neighbors,
        num_neighbors_adaptive,
        width=weight_width,
    )
    return probe_cutoffs @ cutoffs_weights.T


def _graph_pair_cutoffs(
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    edge_distances: torch.Tensor,
    num_atoms: int,
    graph_attention_cutoff: float,
    graph_attention_num_neighbors_adaptive: Optional[float],
    graph_attention_adaptive_cutoff_method: str,
    graph_attention_cutoff_width: float,
) -> torch.Tensor:
    if graph_attention_num_neighbors_adaptive is None:
        return edge_distances.new_full(edge_distances.shape, graph_attention_cutoff)

    cutoff_method = graph_attention_adaptive_cutoff_method.lower()
    if cutoff_method == "solver":
        atomic_cutoffs = get_adaptive_cutoffs_solver(
            centers,
            edge_distances,
            graph_attention_num_neighbors_adaptive,
            num_atoms,
            graph_attention_cutoff,
            cutoff_width=graph_attention_cutoff_width,
        )
    elif cutoff_method == "grid":
        atomic_cutoffs = get_adaptive_cutoffs_grid(
            centers,
            edge_distances,
            graph_attention_num_neighbors_adaptive,
            num_atoms,
            graph_attention_cutoff,
            cutoff_width=graph_attention_cutoff_width,
        )
    else:
        raise ValueError("invalid graph attention adaptive cutoff method")

    return (atomic_cutoffs[centers] + atomic_cutoffs[neighbors]) / 2.0


def _graph_cutoff_factors(
    edge_distances: torch.Tensor,
    pair_cutoffs: torch.Tensor,
    graph_attention: str,
    graph_attention_cutoff_function: str,
    graph_attention_cutoff_width: float,
) -> torch.Tensor:
    if graph_attention == "binary":
        return edge_distances.new_ones(edge_distances.shape)

    cutoff_function = graph_attention_cutoff_function.lower()
    if cutoff_function == "bump":
        return cutoff_func_bump(
            edge_distances,
            pair_cutoffs,
            graph_attention_cutoff_width,
        )
    if cutoff_function == "cosine":
        return cutoff_func_cosine(
            edge_distances,
            pair_cutoffs,
            graph_attention_cutoff_width,
        )
    raise ValueError("invalid graph attention cutoff function")


def _scatter_max_graph_factors(
    dense_factors: torch.Tensor,
    systems: torch.Tensor,
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    edge_factors: torch.Tensor,
) -> None:
    if centers.numel() == 0:
        return

    max_atoms = dense_factors.shape[1]
    flat_index = systems * max_atoms * max_atoms + centers * max_atoms + neighbors
    flat_factors = dense_factors.reshape(-1)
    if hasattr(flat_factors, "scatter_reduce_"):
        flat_factors.scatter_reduce_(
            0,
            flat_index,
            edge_factors,
            reduce="amax",
            include_self=True,
        )
        return

    # Compatibility path for older PyTorch builds without scatter_reduce_.
    for index, value in zip(flat_index.tolist(), edge_factors):
        flat_factors[index] = torch.maximum(flat_factors[index], value)


def _local_atom_indices(
    batch: torch.Tensor,
    batch_size: int,
    atom_counts: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.numel() > 0 and batch.min() < 0:
        raise ValueError("batch indices must be non-negative")

    if atom_counts is None:
        atom_counts = torch.bincount(batch, minlength=batch_size)
        if atom_counts.shape[0] != batch_size:
            raise ValueError("batch contains a system index outside the cell batch")
    else:
        atom_counts = atom_counts.to(
            device=batch.device,
            dtype=torch.long,
        ).reshape(-1)
        if atom_counts.numel() != batch_size:
            raise ValueError("atom_counts must have one entry per system")
        if atom_counts.numel() > 0 and atom_counts.min() < 0:
            raise ValueError("atom_counts must be non-negative")
        if int(atom_counts.sum().item()) != batch.numel():
            raise ValueError("atom_counts must sum to the number of atoms")

    if batch.numel() == 0:
        return batch, atom_counts
    if batch.max() >= batch_size:
        raise ValueError("batch contains a system index outside the cell batch")
    if torch.any(batch[1:] < batch[:-1]):
        raise ValueError("batch must be sorted by system")

    atom_offsets = torch.cumsum(atom_counts, dim=0) - atom_counts
    global_indices = torch.arange(
        batch.numel(),
        device=batch.device,
        dtype=batch.dtype,
    )
    local_indices = global_indices - atom_offsets.index_select(0, batch)
    local_counts = atom_counts.index_select(0, batch)
    if torch.any(local_indices < 0) or torch.any(local_indices >= local_counts):
        raise ValueError("atom_counts must match the grouped batch layout")
    return local_indices, atom_counts


def build_dense_graph_attention_bias(
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    edge_distances: torch.Tensor,
    batch: torch.Tensor,
    *,
    batch_size: Optional[int] = None,
    max_atoms: Optional[int] = None,
    atom_counts: Optional[torch.Tensor] = None,
    graph_attention: str,
    graph_attention_cutoff: float,
    graph_attention_num_neighbors_adaptive: Optional[float],
    graph_attention_adaptive_cutoff_method: str,
    graph_attention_cutoff_width: float,
    graph_attention_cutoff_function: str,
    graph_attention_bias_strength: float,
    graph_attention_epsilon: float,
) -> torch.Tensor:
    """Build dense atom-to-atom attention log-bias from sparse graph edges.

    The returned tensor has shape ``[batch_size, max_atoms, max_atoms]`` and is
    ready to pass to ``StructureTransformer.forward(graph_attention_bias=...)``.
    Missing edges receive ``bias_strength * log(epsilon)``; self edges receive
    zero bias. If ``atom_counts`` is provided, it is used as the fast path for
    global-to-local atom indexing; the batch is expected to be grouped by system.
    """
    if graph_attention not in {"binary", "smooth_cutoff"}:
        raise ValueError("graph_attention must be 'binary' or 'smooth_cutoff'")
    if graph_attention_cutoff <= 0.0:
        raise ValueError("graph_attention_cutoff must be positive")
    if graph_attention_cutoff_width <= 0.0:
        raise ValueError("graph_attention_cutoff_width must be positive")
    if graph_attention_bias_strength < 0.0:
        raise ValueError("graph_attention_bias_strength must be non-negative")
    if graph_attention_epsilon <= 0.0:
        raise ValueError("graph_attention_epsilon must be positive")
    if (
        graph_attention_num_neighbors_adaptive is not None
        and graph_attention_num_neighbors_adaptive <= 0.0
    ):
        raise ValueError(
            "graph_attention_num_neighbors_adaptive must be positive when provided"
        )

    batch = batch.long().reshape(-1)
    centers = centers.to(device=batch.device, dtype=torch.long).reshape(-1)
    neighbors = neighbors.to(device=batch.device, dtype=torch.long).reshape(-1)
    edge_distances = edge_distances.to(device=batch.device).reshape(-1)
    if (
        centers.numel() != neighbors.numel()
        or centers.numel() != edge_distances.numel()
    ):
        raise ValueError("centers, neighbors, and edge_distances must align")

    if batch_size is None:
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    local_indices, atom_counts = _local_atom_indices(
        batch,
        batch_size,
        atom_counts,
    )
    if max_atoms is None:
        max_atoms = int(atom_counts.max().item()) if atom_counts.numel() > 0 else 0

    if atom_counts.numel() > 0 and max_atoms < int(atom_counts.max().item()):
        raise ValueError("max_atoms must be at least the largest system atom count")

    cutoff_factors = torch.zeros(
        (batch_size, max_atoms, max_atoms),
        device=batch.device,
        dtype=edge_distances.dtype,
    )
    if batch_size > 0 and max_atoms > 0:
        diagonal = torch.arange(max_atoms, device=batch.device)
        valid_diagonal = diagonal.unsqueeze(0) < atom_counts.unsqueeze(1)
        cutoff_factors[:, diagonal, diagonal] = valid_diagonal.to(
            dtype=cutoff_factors.dtype,
        )

    if centers.numel() > 0:
        if centers.max() >= batch.numel() or neighbors.max() >= batch.numel():
            raise ValueError("edge indices must refer to atoms in batch")
        edge_systems = batch.index_select(0, centers)
        neighbor_systems = batch.index_select(0, neighbors)
        if not torch.all(edge_systems == neighbor_systems):
            raise ValueError("graph attention edges must stay within one system")

        local_centers = local_indices.index_select(0, centers)
        local_neighbors = local_indices.index_select(0, neighbors)
        graph_edge_distances = edge_distances + 1.0e-15
        pair_cutoffs = _graph_pair_cutoffs(
            centers,
            neighbors,
            graph_edge_distances,
            batch.numel(),
            graph_attention_cutoff,
            graph_attention_num_neighbors_adaptive,
            graph_attention_adaptive_cutoff_method,
            graph_attention_cutoff_width,
        )
        if graph_attention_num_neighbors_adaptive is not None:
            keep = torch.nonzero(graph_edge_distances <= pair_cutoffs).squeeze(-1)
            edge_systems = edge_systems.index_select(0, keep)
            local_centers = local_centers.index_select(0, keep)
            local_neighbors = local_neighbors.index_select(0, keep)
            graph_edge_distances = graph_edge_distances.index_select(0, keep)
            pair_cutoffs = pair_cutoffs.index_select(0, keep)

        edge_factors = _graph_cutoff_factors(
            graph_edge_distances,
            pair_cutoffs,
            graph_attention,
            graph_attention_cutoff_function,
            graph_attention_cutoff_width,
        )
        _scatter_max_graph_factors(
            cutoff_factors,
            edge_systems,
            local_centers,
            local_neighbors,
            edge_factors,
        )

    cutoff_factors = cutoff_factors.clamp_min(graph_attention_epsilon)
    return graph_attention_bias_strength * torch.log(cutoff_factors)


__all__ = [
    "build_dense_graph_attention_bias",
    "cutoff_func_bump",
    "cutoff_func_cosine",
    "get_adaptive_cutoffs_grid",
    "get_adaptive_cutoffs_solver",
]

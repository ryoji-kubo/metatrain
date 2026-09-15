"""Atomistic GenT encoders, dense batching and readouts, independent of metatrain."""

import math
from typing import Dict, NamedTuple, Optional

import torch
from torch import nn

from .gent_layers import GenTBlock, MLP_SiLU, RMSNorm


class GenTData(NamedTuple):
    """Tensor contract for a batch of structures.

    Atoms are concatenated in structure order. ``num_atoms`` has shape [B],
    ``atomic_numbers`` [A], and ``cells`` [B, 3, 3], with row-vector cells.
    Edges form a directed *full* neighbor list (both orientations) including all
    periodic images inside the cutoff. Indices are global into the concatenated
    atoms. ``edge_vectors`` [E, 3] point from center to neighbor image; differentiable
    callers must keep their dependency on positions/cells. No zero-shift self edges
    are needed. Repeated center/neighbor indices are expected for periodic images.
    """

    atomic_numbers: torch.Tensor
    num_atoms: torch.Tensor
    edge_vectors: torch.Tensor
    edge_centers: torch.Tensor
    edge_neighbors: torch.Tensor
    cells: torch.Tensor


class GeneralizedTransformer(nn.Module):
    """Dense GenT with invariant scalar states and equivariant direct readouts.

    Attention is global within each structure. Neighbor lists encode geometry;
    they do not limit the receptive field. Distinct periodic image encodings are
    summed into dense pair states and remain separate in force/stress readouts.
    Sequence-index RoPE is deliberately absent from the atomistic model.
    """

    def __init__(
        self,
        max_num_elements: int = 128,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        hidden_dim_sa: Optional[int] = None,
        hidden_dim_ca_n: Optional[int] = None,
        hidden_dim_ca_e: Optional[int] = None,
        mlp_hidden_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        cutoff: float = 5.0,
        num_radial_basis: int = 32,
        regress_forces: bool = True,
        regress_stress: bool = True,
    ):
        super().__init__()
        if max_num_elements <= 1 or embed_dim <= 0 or num_layers <= 0:
            raise ValueError(
                "max_num_elements > 1, embed_dim > 0 and num_layers > 0 are required"
            )
        if not math.isfinite(cutoff) or cutoff <= 0 or num_radial_basis < 2:
            raise ValueError(
                "cutoff must be finite and positive; num_radial_basis must be >= 2"
            )
        self.cutoff = float(cutoff)
        self.embed_dim = embed_dim
        self.max_num_elements = max_num_elements
        self.regress_forces = regress_forces
        self.regress_stress = regress_stress
        ffn_dim = int(2.25 * embed_dim) if mlp_hidden_dim is None else mlp_hidden_dim
        self.atom_embedding = nn.Embedding(max_num_elements, embed_dim)
        self.self_edge_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer(
            "radial_centers", torch.linspace(0.0, cutoff, num_radial_basis)
        )
        self.radial_width = (num_radial_basis - 1) / cutoff
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_radial_basis, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.layers = nn.ModuleList(
            [
                GenTBlock(
                    embed_dim,
                    embed_dim if hidden_dim_sa is None else hidden_dim_sa,
                    embed_dim if hidden_dim_ca_n is None else hidden_dim_ca_n,
                    embed_dim if hidden_dim_ca_e is None else hidden_dim_ca_e,
                    ffn_dim,
                    num_heads,
                    attn_dropout,
                    residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.node_norm = RMSNorm(embed_dim)
        self.edge_norm = RMSNorm(embed_dim)
        self.energy_head = MLP_SiLU(embed_dim, ffn_dim, 1)
        # Separate heads keep direct force/stress training independent of energy.
        self.force_head = MLP_SiLU(embed_dim + num_radial_basis, ffn_dim, 1)
        self.stress_head = MLP_SiLU(embed_dim + num_radial_basis, ffn_dim, 1)
        if not regress_forces:
            self.force_head.requires_grad_(False)
        if not regress_stress:
            self.stress_head.requires_grad_(False)

    def forward(self, data: GenTData) -> Dict[str, torch.Tensor]:
        counts = data.num_atoms
        if counts.numel() == 0 or bool(torch.any(counts <= 0)):
            raise ValueError("GenT requires a nonempty batch of nonempty structures")
        if int(counts.sum()) != data.atomic_numbers.numel():
            raise ValueError("num_atoms must sum to the number of atomic_numbers")
        if bool(torch.any(data.atomic_numbers < 1)) or bool(
            torch.any(data.atomic_numbers >= self.max_num_elements)
        ):
            raise ValueError("atomic_numbers must be in [1, max_num_elements)")
        batch_size = counts.numel()
        max_atoms = int(counts.max())
        device = data.atomic_numbers.device
        batch = torch.repeat_interleave(torch.arange(batch_size, device=device), counts)
        offsets = torch.cumsum(counts, 0) - counts
        local = torch.arange(batch.numel(), device=device) - offsets[batch]
        flat_atoms = batch * max_atoms + local
        mask = torch.arange(max_atoms, device=device).unsqueeze(0) < counts.unsqueeze(1)
        embeddings = self.atom_embedding(data.atomic_numbers)
        x = embeddings.new_zeros((batch_size * max_atoms, self.embed_dim))
        x = x.index_copy(0, flat_atoms, embeddings).view(
            batch_size, max_atoms, self.embed_dim
        )

        centers, neighbors = data.edge_centers, data.edge_neighbors
        if bool(torch.any(batch[centers] != batch[neighbors])):
            raise ValueError("Edges can not connect atoms from different structures")
        pair_indices = (
            batch[centers] * max_atoms + local[centers]
        ) * max_atoms + local[neighbors]
        distances = torch.linalg.vector_norm(data.edge_vectors, dim=-1)
        # C2 envelope: values and their first two derivatives vanish at the cutoff.
        fraction = (distances / self.cutoff).clamp(0.0, 1.0)
        envelope = 1.0 - 10.0 * fraction**3 + 15.0 * fraction**4 - 6.0 * fraction**5
        radial = torch.exp(
            -(
                (distances.unsqueeze(-1) - self.radial_centers) * self.radial_width
            ).square()
        )
        encoded = self.edge_encoder(radial) * envelope.unsqueeze(-1)
        e = x.new_zeros((batch_size * max_atoms * max_atoms, self.embed_dim))
        e = e.index_add(0, pair_indices, encoded).view(
            batch_size, max_atoms, max_atoms, self.embed_dim
        )
        diagonal = torch.eye(max_atoms, device=device, dtype=x.dtype)
        e = e + diagonal[None, :, :, None] * self.self_edge_embedding
        for layer in self.layers:
            x, e = layer(x, e, None, mask)

        atom_energies = self.energy_head(self.node_norm(x)).squeeze(-1)
        atom_energies = atom_energies.masked_fill(~mask, 0.0)
        result = {
            "energy": atom_energies.sum(dim=1),
            "atomic_energy": atom_energies[mask],
        }
        pair_states = self.edge_norm(e).reshape(-1, self.embed_dim)[pair_indices]
        readout = torch.cat((pair_states, radial), dim=-1)
        if self.regress_forces:
            coefficients = self.force_head(readout) * envelope.unsqueeze(-1)
            directions = data.edge_vectors / distances.clamp_min(1e-12).unsqueeze(-1)
            pair_forces = 0.5 * coefficients * directions
            forces = x.new_zeros((batch.numel(), 3))
            # Both endpoints receive opposite contributions, even when e_ij != e_ji.
            forces = forces.index_add(0, centers, pair_forces)
            forces = forces.index_add(0, neighbors, -pair_forces)
            result["forces"] = forces
        if self.regress_stress:
            coefficients = self.stress_head(readout) * envelope.unsqueeze(-1)
            dyads = data.edge_vectors.unsqueeze(-1) * data.edge_vectors.unsqueeze(-2)
            edge_stress = 0.5 * coefficients.unsqueeze(-1) * dyads
            stress = x.new_zeros((batch_size, 3, 3)).index_add(
                0, batch[centers], edge_stress
            )
            volume = torch.linalg.det(data.cells).abs()
            # Nonperiodic structures with zero cells use a unit normalization volume.
            volume = torch.where(volume > 1e-12, volume, torch.ones_like(volume))
            result["stress"] = stress / volume[:, None, None]
        return result

# PET Periodic Edge Multiplicity and a Structure Transformer Limitation

This note explains a subtle but important difference between PET-style graph
representations and the Structure Transformer graph-attention prior. The short
version is:

```math
\text{PET edge object} = (i,j,\eta),
\qquad
\text{Structure Transformer attention object} = (i,j).
```

The extra periodic-image index `eta` can matter. It lets PET represent multiple
geometrically distinct interactions between the same atom indices `i` and `j`.
The Structure Transformer graph-attention prior currently collapses those
interactions into one scalar attention bias.

## PET Constructs a Periodic Neighbor Graph

PET requests a neighbor list with a cutoff:

```python
self.requested_nl = NeighborListOptions(
    cutoff=self.cutoff,
    full_list=True,
    strict=True,
)
```

This is in `src/metatrain/pet/model.py`. The important point is that the graph
is constructed before the PET module sees the batch. PET does not learn the
periodic image index. It receives it from the neighbor-list construction.

For a structure with Cartesian atom positions

```math
x_i \in \mathbb{R}^3
```

and lattice matrix

```math
L \in \mathbb{R}^{3\times 3},
```

a periodic image of atom `j` is

```math
x_j^{(\eta)} = x_j + \eta L,
\qquad
\eta=(\eta_1,\eta_2,\eta_3)\in\mathbb{Z}^3.
```

With the row-vector convention used in the code, `eta L` is implemented as
`cell_shifts @ cell`. The directed periodic edge vector from center atom `i` to
that image of neighbor atom `j` is

```math
r_{ij}^{\eta}
= x_j+\eta L-x_i,
\qquad
\rho_{ij}^{\eta}
= \left\|r_{ij}^{\eta}\right\|_2.
```

The neighbor-list graph is conceptually

```math
E
=
\left\{
(i,j,\eta):
\rho_{ij}^{\eta}\le r_{\mathrm{cut}}
\right\}.
```

The set of all integer shifts is infinite, but the graph constructor only needs
to enumerate the finite set of image cells that can possibly fall within
`r_cut`. For non-periodic structures, or for pairs that do not cross a periodic
boundary, the relevant shift is usually `eta = (0,0,0)`.

PET extracts the neighbor-list samples in
`src/metatrain/pet/modules/structures.py`:

```python
neighbor_list = system.get_neighbor_list(neighbor_list_options)
nl_values = neighbor_list.samples.values

centers = nl_values[:, 0]
neighbors = nl_values[:, 1]
cell_shifts = nl_values[:, 2:]
```

Then PET computes the actual Cartesian edge vectors and distances:

```python
cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
edge_vectors = positions[neighbors] - positions[centers] + cell_contributions
edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15
```

Mathematically, this is exactly

```math
\texttt{edge\_vectors}_{e}
=
r_{ij}^{\eta}
=
x_j-x_i+\eta L.
```

## The Same Atom Pair Can Have Multiple Periodic Edges

For fixed atom indices `i` and `j`, define the set of periodic image shifts that
survive the cutoff:

```math
M_{ij}
=
\left\{
\eta\in\mathbb{Z}^3:
(i,j,\eta)\in E
\right\}.
```

If the unit cell is small, or the cutoff is large, then

```math
|M_{ij}| > 1
```

can occur. In that case the same atom index `j` appears as multiple periodic
neighbors of atom `i`:

```math
(i,j,\eta_1),\quad
(i,j,\eta_2),\quad
\ldots
```

These are not duplicate edges in a physical sense. They correspond to different
periodic images:

```math
r_{ij}^{\eta_1}\ne r_{ij}^{\eta_2},
\qquad
\rho_{ij}^{\eta_1}\ne \rho_{ij}^{\eta_2}
\quad\text{in general}.
```

Even if two image distances are similar, their directions can differ:

```math
\hat r_{ij}^{\eta}
=
\frac{r_{ij}^{\eta}}{\rho_{ij}^{\eta}}.
```

So PET's local environment around atom `i` is better viewed as an edge-image
set

```math
\mathcal{N}(i)
=
\{(j,\eta):(i,j,\eta)\in E\},
```

not merely as a set of neighboring atom indices

```math
\{j:(i,j)\ \text{is connected}\}.
```

## Why This Matters Physically

A local energy model with periodic interactions naturally sums over periodic
edge images:

```math
E
\approx
\sum_i E_i,
\qquad
E_i
=
\sum_{(j,\eta)\in\mathcal{N}(i)}
\phi_\theta
\left(
A_i,A_j,r_{ij}^{\eta}
\right).
```

If only the shortest image is kept, this becomes

```math
E_i^{\mathrm{min}}
=
\sum_j
\phi_\theta
\left(
A_i,A_j,r_{ij}^{\eta^\star(i,j)}
\right),
```

where

```math
\eta^\star(i,j)
=
\operatorname*{argmin}_{\eta\in\mathbb{Z}^3}
\left\|x_j+\eta L-x_i\right\|_2.
```

This minimum-image approximation is safe only when the cutoff and cell geometry
guarantee that at most one image of atom `j` lies inside the cutoff for each
center atom `i`.

For force and stress, the loss can be more serious because displacement
directions enter directly. If an edge contribution is schematically

```math
e_{ij}^{\eta}
=
\phi_\theta(\rho_{ij}^{\eta}),
```

then a force contribution depends on

```math
\frac{\partial \rho_{ij}^{\eta}}{\partial x_i}
=
-\frac{r_{ij}^{\eta}}{\rho_{ij}^{\eta}}.
```

Thus the force direction is tied to the image-specific vector
`r_ij^eta`. A virial-like stress contribution also depends on edge geometry:

```math
\sigma
\sim
\frac{1}{V}
\sum_{i,j,\eta}
r_{ij}^{\eta}\otimes F_{ij}^{\eta}.
```

Dropping periodic image multiplicity can therefore remove both:

```math
\text{how many images interact}
\quad\text{and}\quad
\text{which directions those interactions point}.
```

## PET Preserves the Image-Edge Object

PET keeps each neighbor-list row as a distinct edge. In other words, PET can
construct features over

```math
(i,j,\eta)
```

rather than only over

```math
(i,j).
```

This means PET can distinguish environments such as

```math
\mathcal{N}_1(i,j)
=
\{(j,\eta_1)\}
```

from

```math
\mathcal{N}_2(i,j)
=
\{(j,\eta_1),(j,\eta_2)\},
```

even if both environments share the same closest image. The multiplicity itself
is part of the representation.

This is one reason PET's representation is richer than a pure atom-token
transformer. It uses the graph not merely as a mask, but as a set of persistent
geometric interaction objects.

## Structure Transformer Collapses Image Edges to Atom-Pair Biases

The Structure Transformer sequence is

```math
Z=[z_{\mathrm{cell}},h_1,h_2,\ldots,h_N].
```

Its atom-token attention has one logit for each atom-index pair:

```math
S_{ij}^{(\ell,r)}
=
\frac{
(q_i^{(\ell,r)})^T k_j^{(\ell,r)}
}{\sqrt{d_h}}
+
B_{ij}.
```

With the PET-style graph-attention prior, the bias is

```math
B_{ij}
=
\lambda\log(\max(c_{ij},\epsilon)).
```

But because there is only one attention entry for atom pair `(i,j)`, all
periodic-image edges for that pair must be summarized into one scalar:

```math
c_{ij}
=
\max_{\eta\in M_{ij}}
c\left(\rho_{ij}^{\eta}\right).
```

This is implemented in
`src/metatrain/experimental/structure_transformer/model.py`:

```python
flat_index = centers * num_atoms + neighbors
flat_factors = dense_factors.reshape(-1)
flat_factors.scatter_reduce_(
    0,
    flat_index,
    edge_factors,
    reduce="amax",
    include_self=True,
)
```

The resulting dense atom-atom bias is then added to the transformer attention
matrix in `src/metatrain/experimental/structure_transformer/modules/transformer.py`:

```python
attn = torch.matmul(
    q_for_scores.float(), k_for_scores.float().transpose(-2, -1)
)
attn = attn * self.scale
attn = attn + attention_bias.to(
    device=attn.device,
    dtype=attn.dtype,
).unsqueeze(1)
```

So the Structure Transformer receives a graph prior of the form

```math
\alpha_{ij}
=
\operatorname{softmax}_j
\left(
\frac{q_i^T k_j}{\sqrt{d_h}}
+
\lambda\log(c_{ij})
\right),
```

where `c_ij` is a collapsed atom-pair connection strength.

## What Information Is Lost by the Collapse?

The collapse map is approximately

```math
\left\{
\left(
\rho_{ij}^{\eta},
r_{ij}^{\eta},
c(\rho_{ij}^{\eta})
\right)
:
\eta\in M_{ij}
\right\}
\longmapsto
\max_{\eta\in M_{ij}} c(\rho_{ij}^{\eta}).
```

This preserves a useful fact:

```math
\text{there exists a strong local connection between } i \text{ and } j.
```

But it discards:

```math
|M_{ij}|
\quad
\text{the number of periodic image edges},
```

```math
\{r_{ij}^{\eta}:\eta\in M_{ij}\}
\quad
\text{the image-specific displacement vectors},
```

```math
\{\rho_{ij}^{\eta}:\eta\in M_{ij}\}
\quad
\text{the non-maximum distances},
```

and

```math
\{\eta:\eta\in M_{ij}\}
\quad
\text{the periodic image identities}.
```

For example, suppose two environments have the same strongest edge:

```math
\mathcal{E}_1(i,j)
=
\{r_{ij}^{\eta_1}\},
```

```math
\mathcal{E}_2(i,j)
=
\{r_{ij}^{\eta_1},r_{ij}^{\eta_2}\}.
```

If

```math
c(\|r_{ij}^{\eta_1}\|)
\ge
c(\|r_{ij}^{\eta_2}\|),
```

then both environments produce the same graph-attention scalar:

```math
\max_{\eta\in M_{ij}}c(\rho_{ij}^{\eta})
=
c(\|r_{ij}^{\eta_1}\|).
```

The attention prior cannot distinguish the one-image case from the two-image
case, even though PET can.

## Potential Limitation for Structure Transformer

This suggests a concrete limitation of the current atom-token Structure
Transformer:

```math
\text{atom-token attention has } O(N^2) \text{ pair slots,}
```

but the periodic graph may contain

```math
O(|E|)
=
O\left(\sum_{i,j}|M_{ij}|\right)
```

image-specific interaction objects. When some `|M_ij| > 1`, the atom-token
attention matrix does not have enough slots to represent every periodic image
interaction independently.

This bottleneck may be acceptable for energy if the nearest image dominates:

```math
\sum_{\eta\in M_{ij}}
\phi_\theta(r_{ij}^{\eta})
\approx
\phi_\theta(r_{ij}^{\eta^\star}).
```

It is less obviously acceptable for force and stress, where the directions and
outer products of the individual edge vectors matter:

```math
F_i
\text{ depends on }
\left\{
\frac{r_{ij}^{\eta}}{\rho_{ij}^{\eta}}
\right\}_{j,\eta},
\qquad
\sigma
\text{ depends on }
\left\{
r_{ij}^{\eta}\otimes F_{ij}^{\eta}
\right\}_{i,j,\eta}.
```

Therefore, the PET-style graph-attention prior is a useful ablation, but it is
not equivalent to giving the Structure Transformer PET's full graph
representation. It injects local connectivity into attention while preserving
the atom-token architecture, but it still compresses image-edge multiplicity
into a single scalar per atom pair.

## Possible Node-Only Follow-Ups

If the research goal is to remain close to a node-token transformer, possible
less-invasive summaries of periodic image multiplicity include:

```math
c_{ij}^{\max}
=
\max_{\eta\in M_{ij}} c(\rho_{ij}^{\eta}),
```

```math
c_{ij}^{\sum}
=
\sum_{\eta\in M_{ij}} c(\rho_{ij}^{\eta}),
```

```math
m_{ij}
=
|M_{ij}|,
```

and directional moments such as

```math
\mu_{ij}
=
\sum_{\eta\in M_{ij}}
c(\rho_{ij}^{\eta})r_{ij}^{\eta},
```

```math
Q_{ij}
=
\sum_{\eta\in M_{ij}}
c(\rho_{ij}^{\eta})
r_{ij}^{\eta}(r_{ij}^{\eta})^T.
```

These still summarize edges into atom-pair features, but they retain more of
the lost multiplicity and geometry than a single max cutoff factor. The tradeoff
is that adding vector or tensor summaries begins to move the model closer to a
geometric graph model, which is exactly the boundary this ablation is trying to
understand.

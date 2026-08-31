# PET-Style Graph Attention Bias for Structure Transformer

This note documents the graph-prior attention path added to the experimental
Structure Transformer. The goal is to study how much local graph information
helps energy, force, and stress prediction while keeping the backbone a global
atom-token transformer.

The implementation reuses PET's central idea: construct smooth cutoff factors
from neighbor-list edge distances, then inject those factors into attention as
an additive log-bias.

## PET Reference Point

PET computes directed neighbor-list edge vectors

```math
r_{ij}^{\eta}=x_j+\eta L-x_i,
\qquad
\rho_{ij}^{\eta}=\|r_{ij}^{\eta}\|,
```

where `eta in Z^3` is the periodic cell shift attached to the neighbor-list
edge. In PET this happens in `systems_to_batch` in
`src/metatrain/pet/modules/structures.py`:

```python
edge_vectors = positions[neighbors] - positions[centers] + cell_contributions
edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15
```

PET then evaluates one of the cutoff functions from
`src/metatrain/pet/modules/utilities.py`. For cutoff radius `r_c` and smoothing
width `w`, define

```math
s(\rho)=\frac{\rho-(r_c-w)}{w}.
```

The PET bump cutoff is

```math
c_{\mathrm{bump}}(\rho)
=
\frac{1}{2}
\left[
1+\tanh\left(
\frac{1}{\tan(\pi\operatorname{clip}(s(\rho),\epsilon,1-\epsilon))}
\right)
\right].
```

The PET cosine cutoff is

```math
c_{\mathrm{cos}}(\rho)
=
\frac{1}{2}
\left[
1+\cos\left(\pi\operatorname{clip}(s(\rho),0,1)\right)
\right].
```

Both satisfy approximately

```math
c(\rho)=1\quad\rho\le r_c-w,
\qquad
c(\rho)\to 0\quad\rho\to r_c.
```

PET's attention block then converts the cutoff factor into an additive log-mask:

```python
attn_weights = torch.clamp(cutoff_factors[:, None, :, :], self.epsilon)
attn_weights = torch.log(attn_weights)
```

and uses it in attention:

```python
attention_weights = (
    torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5 * temperature)
) + attn_mask
attention_weights = attention_weights.softmax(dim=-1)
```

Mathematically, PET attention uses

```math
\alpha_{ij}
=
\operatorname{softmax}_j
\left(
\frac{q_i^T k_j}{\sqrt{d_h}\,T}+\log c_{ij}
\right),
```

or equivalently

```math
\alpha_{ij}
=
\frac{c_{ij}\exp(q_i^T k_j/(\sqrt{d_h}T))}
{\sum_k c_{ik}\exp(q_i^T k_k/(\sqrt{d_h}T))}.
```

So the cutoff factor is a multiplicative graph prior on attention probability.
It is not applied after attention; it changes the softmax competition itself.

## Structure Transformer Adaptation

The Structure Transformer sequence is

```math
Z=[z_{\mathrm{cell}},h_1,h_2,\ldots,h_N].
```

Graph connectivity naturally applies only to atom-atom entries. The lattice token
has no pair distance, so its row and column are left un-biased:

```math
C^{\mathrm{seq}}=
\begin{bmatrix}
1 & 1 & \cdots & 1\\
1 & c_{11} & \cdots & c_{1N}\\
\vdots & \vdots & \ddots & \vdots\\
1 & c_{N1} & \cdots & c_{NN}
\end{bmatrix}.
```

The transformer consumes an additive bias

```math
B^{\mathrm{seq}}_{ab}
=\lambda\log\left(\max(C^{\mathrm{seq}}_{ab},\epsilon)\right),
```

where `lambda = graph_attention_bias_strength`. The attention score in every
transformer block becomes

```math
S_{ab}^{(\ell,r)}
=
\frac{(q_a^{(\ell,r)})^Tk_b^{(\ell,r)}}{\sqrt{d_h}}
+B^{\mathrm{seq}}_{ab}.
```

The attention weights are

```math
\alpha_{ab}^{(\ell,r)}
=\operatorname{softmax}_b S_{ab}^{(\ell,r)}.
```

The useful ablation axis is therefore

```math
\lambda=0 \Rightarrow \text{original all-to-all transformer},
```

```math
0<\lambda<1 \Rightarrow \text{weak graph prior},
```

```math
\lambda=1 \Rightarrow \text{PET-strength log-cutoff prior}.
```

## Dense Atom-Atom Bias Construction

The wrapper `StructureTransformerModel` now requests a neighbor list whenever

```math
\texttt{graph\_attention}\ne\texttt{none}
\quad\text{and}\quad
\lambda>0.
```

The relevant code is in
`src/metatrain/experimental/structure_transformer/model.py`:

```python
self.graph_attention = self.transformer.graph_attention
self.uses_graph_attention = (
    self.graph_attention != "none"
    and self.transformer.graph_attention_bias_strength > 0.0
)
self.graph_requested_nl = NeighborListOptions(
    cutoff=graph_attention_cutoff,
    full_list=True,
    strict=True,
)
```

The helper `_systems_to_graph_attention_bias` constructs

```math
G\in\mathbb{R}^{B\times N_{\max}\times N_{\max}},
\qquad
G_{bij}=\lambda\log(\max(c_{bij},\epsilon)).
```

For each structure, it initializes

```math
c_{ii}=1,
\qquad
c_{ij}=0\quad i\ne j,
```

then fills neighbor-list entries. If `graph_attention: binary`, every real edge
gets

```math
c_{ij}=1.
```

If `graph_attention: smooth_cutoff`, the code calls PET's actual cutoff
functions:

```python
if cutoff_function == "bump":
    return cutoff_func_bump(edge_distances, pair_cutoffs, cutoff_width)
if cutoff_function == "cosine":
    return cutoff_func_cosine(edge_distances, pair_cutoffs, cutoff_width)
```

If several periodic images produce edges for the same atom pair `(i,j)`, the
code collapses them by max strength:

```math
c_{ij}=\max_{\eta:(i,j,\eta)\in\mathcal{E}} c(\rho_{ij}^{\eta}).
```

This corresponds to the nearest or strongest periodic image controlling the
single dense attention entry `A_ij`. In code this is `_scatter_max_graph_factors`:

```python
flat_index = centers * num_atoms + neighbors
flat_factors.scatter_reduce_(
    0,
    flat_index,
    edge_factors,
    reduce="amax",
    include_self=True,
)
```

Finally, the wrapper passes `graph_attention_bias` into `StructureTransformer`:

```python
raw_outputs = self.transformer(
    data,
    graph_attention_bias=graph_attention_bias,
)
```

## Attention Injection Point

`MultiHeadAttention.forward` in
`src/metatrain/experimental/structure_transformer/modules/transformer.py` accepts

```python
attention_bias: Optional[torch.Tensor] = None
```

with shape

```math
B\times (N_{\max}+1)\times (N_{\max}+1).
```

After the usual query-key dot product and RoPE handling, it adds the bias before
padding is masked and before softmax:

```python
attn = attn * self.scale
if attention_bias is not None:
    attn = attn + attention_bias.unsqueeze(1)
attn = attn.masked_fill(~valid_mask.view(batch_size, 1, 1, seq_len), -1.0e30)
attn = F.softmax(attn, dim=-1)
```

`StructureTransformer.forward` receives the atom-only dense bias

```math
G\in\mathbb{R}^{B\times N_{\max}\times N_{\max}}
```

and embeds it into the full token sequence with zero bias for the lattice token:

```python
sequence_attention_bias = graph_attention_bias.new_zeros(
    (batch_size, max_atoms + 1, max_atoms + 1)
)
sequence_attention_bias[:, 1:, 1:] = graph_attention_bias
```

Since `log(1)=0`, a zero additive bias means unrestricted PET-style connection
strength. Missing graph edges have

```math
\lambda\log\epsilon,
```

which is finite but very negative. With the default `epsilon=10^{-15}` and
`lambda=1`, this is about

```math
\log(10^{-15})\approx -34.54.
```

Thus non-neighbor attention is not represented by `-infty`, but it is suppressed
almost completely, matching PET's clamp-before-log behavior.

## Configuration

The new hyperparameters are exposed in
`src/metatrain/experimental/structure_transformer/documentation.py`:

```yaml
graph_attention: none              # none | binary | smooth_cutoff
graph_attention_cutoff: 4.5
graph_attention_cutoff_width: 0.5
graph_attention_cutoff_function: Bump
graph_attention_bias_strength: 1.0
graph_attention_epsilon: 1.0e-15
```

The v37 config currently enables the PET-style smooth prior:

```yaml
graph_attention: smooth_cutoff
graph_attention_cutoff: 4.5
graph_attention_cutoff_width: 0.5
graph_attention_cutoff_function: Bump
graph_attention_bias_strength: 1.0
graph_attention_epsilon: 1.0e-15
```

Useful run-time overrides are:

```bash
-r architecture.model.graph_attention_bias_strength=0.0
```

for the no-graph-prior baseline with the same config shape, and

```bash
-r architecture.model.graph_attention=binary
```

for a hard graph-connectivity ablation.

## Symmetry Notes

The graph prior preserves the v37 periodic and translation symmetries because it
uses neighbor-list Cartesian edge vectors:

```math
r_{ij}^{\eta}=x_j+\eta L-x_i.
```

A global Cartesian translation `x_i -> x_i+t` leaves

```math
r_{ij}^{\eta}\mapsto (x_j+t)+\eta L-(x_i+t)=r_{ij}^{\eta}.
```

Periodic image shifts are already represented by `eta L`, and the cutoff depends
only on the norm:

```math
c_{ij}^{\eta}=c(\|r_{ij}^{\eta}\|).
```

If atom rows are permuted by `P`, the dense graph factors transform as

```math
C\mapsto PCP^T.
```

The attention logits transform the same way because the atom tokens also permute:

```math
Q\mapsto PQ,
\qquad
K\mapsto PK,
\qquad
QK^T\mapsto P(QK^T)P^T.
```

Therefore

```math
QK^T + \lambda\log C
\mapsto
P(QK^T + \lambda\log C)P^T,
```

and atom outputs remain permutation equivariant while energy and stress reductions
remain permutation invariant.

For now, `graph_attention` is restricted to `atom_ordering: none`. That keeps the
dense graph matrix and token order identical. Supporting sorted/permuted token
orders later would require applying the same permutation to both rows and columns
of `C`.

## Complexity

Let `E` be the number of directed neighbor-list edges and `N=N_max`. Building the
bias costs

```math
T_{\mathrm{graph}}=\Theta(E),
\qquad
S_{\mathrm{graph}}=\Theta(BN^2),
```

because the sparse graph is densified to match the global transformer attention
matrix.

Each transformer block already forms attention logits of size

```math
B\times R\times (N+1)\times (N+1),
```

so adding the graph bias does not change the asymptotic transformer attention
memory. It adds one dense per-batch bias tensor of size

```math
B\times (N+1)\times (N+1),
```

which is much smaller than a persistent edge-token tensor of size

```math
B\times N^2\times D.
```

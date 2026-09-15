# Edge Tokens, Relative Geometry, and Relational Attention

## Scope

The absolute-coordinate PET ablation and the graph-aware Structure Transformer
(ST) both struggle to learn forces, but this does **not** yet prove that a model
must maintain a persistent token for every edge. The narrower conclusion is:

> The current experiments strongly support the need for explicit,
> receiver-specific relative geometry in the message content. They do not yet
> distinguish that requirement from the stronger requirement of persistent edge
> tokens.

This distinction matters because it suggests an intermediate architecture: keep
only node states between layers, as in ST, but construct a pair-conditioned
message from the periodic displacement of every directed neighbor relation at
each layer. This note gives the mathematical motivation for that architecture
and proposes ablations that distinguish the relevant inductive biases.

## 1. What the current experiments show

The two directly comparable nearest-image PET runs are:

| Run | PET geometry input | Last logged train energy RMSE | Last logged train force RMSE | Last logged validation force RMSE |
| --- | --- | ---: | ---: | ---: |
| `outputs/2026-09-07/17-47-34` | relative displacement | 31.96 | 113.45 | 360.97 |
| `outputs/2026-09-08/12-31-33` | absolute coordinates | 116.73 | 797.02 | 573.08 |

The absolute-coordinate run stays close to the force/stress baseline while the
relative-coordinate run learns. The v37 graph-aware ST run at
`outputs/2026-08-31/10-24-18` shows a similar force plateau: its epoch-48 train
and validation force RMSEs are approximately 797.03 and 573.09, respectively.
The two PET configurations differ materially in `geometry_mode`; their
neighbor mode and main architecture/training settings are otherwise matched.

These observations support three claims:

1. Graph connectivity by itself is insufficient.
2. Absolute coordinates are a very poor substitute for an explicit local
   displacement in the tested PET parameterization.
3. Injecting geometry only as a scalar attention preference is weaker than
   putting geometry into the vector-valued message.

They do **not** prove the stronger statement that an edge must be a persistent,
layer-updated token. To make that precise, it helps to separate three objects.

## 2. Three concepts that should not be conflated

Let atom $i$ be in the reference cell, atom $j$ be a neighbor, and
$\mathbf S\in\mathbb Z^3$ denote a periodic cell shift. A directed periodic
relation is

$$
\varepsilon=(i,j,\mathbf S).
$$

The following are different architectural choices:

- **Graph topology:** the set
  $\mathcal E=\{(i,j,\mathbf S):d_{ij\mathbf S}<r_c\}$, or an attention
  mask derived from it.
- **Pair or edge features:** a descriptor
  $\mathbf e_{ij\mathbf S}=\phi(\mathbf r_{ij\mathbf S})$ computed when a
  message is formed.
- **Persistent edge tokens:** a learned state
  $\mathbf g_{ij\mathbf S}^{(\ell)}$ that survives across layers and is
  updated by the network.

PET has the third object. The proposed ST variant has the first two, but not the
third: only node states $\mathbf h_i^{(\ell)}$ persist from one layer to the
next.

The PET `absolute` ablation is therefore better described as replacing the
relative geometric content of PET's edge token with absolute coordinate
content. It does not remove edge tokens. Conversely, v37 ST computes pairwise
quantities, but compresses them to node states and later uses graph geometry
primarily as a scalar attention bias. These experiments do not yet isolate
whether persistent edge state is important.

## 3. Periodic relative geometry

Using row-vector coordinates, let $\mathbf r_i\in\mathbb R^3$ be the
Cartesian position of atom $i$, and let the cell matrix be
$\mathbf L\in\mathbb R^{3\times3}$. The displacement, distance, and unit
direction for relation $\varepsilon=(i,j,\mathbf S)$ are

$$
\begin{aligned}
\mathbf r_\varepsilon
  &= \mathbf r_j + \mathbf S\mathbf L-\mathbf r_i,\\
d_\varepsilon
  &= \lVert\mathbf r_\varepsilon\rVert_2,\\
\widehat{\mathbf r}_\varepsilon
  &= \frac{\mathbf r_\varepsilon}{d_\varepsilon}.
\end{aligned}
$$

Equivalently, for a fractional displacement
$\Delta\mathbf u_\varepsilon$,

$$
\mathbf r_\varepsilon=\Delta\mathbf u_\varepsilon\mathbf L,
\qquad
d_\varepsilon^2
=\Delta\mathbf u_\varepsilon
  \mathbf L\mathbf L^{\mathsf T}
  \Delta\mathbf u_\varepsilon^{\mathsf T}.
$$

The metric tensor $\mathbf G=\mathbf L\mathbf L^{\mathsf T}$ is essential.
Fractional displacement alone does not determine Cartesian distance when cells
vary in size or shape.

Relative displacement has the desired behavior under translation and rotation.
For a global translation $\mathbf t$,

$$
(\mathbf r_j+\mathbf t)+\mathbf S\mathbf L-(\mathbf r_i+\mathbf t)
=\mathbf r_\varepsilon.
$$

For a global orthogonal transformation $\mathbf R$, using column-vector
notation for this line,

$$
\mathbf r_\varepsilon\mapsto\mathbf R\mathbf r_\varepsilon,
\qquad
d_\varepsilon\mapsto d_\varepsilon.
$$

Thus radial functions of $d_\varepsilon$ are invariant, while
$\widehat{\mathbf r}_\varepsilon$ is a covariant direction that can be used
to construct vector outputs.

## 4. Why ordinary self-attention does not easily recover the relation

### 4.1 Geometry in the logit is only a scalar routing signal

One attention head has the form

$$
\begin{aligned}
\mathbf q_i &= \mathbf W_Q\mathbf h_i,&
\mathbf k_j &= \mathbf W_K\mathbf h_j,&
\mathbf v_j &= \mathbf W_V\mathbf h_j,\\
\ell_{ij} &=
\frac{\mathbf q_i^{\mathsf T}\mathbf k_j}{\sqrt{d_h}}+b_{ij},&
\alpha_{ij}&=\frac{\exp\ell_{ij}}
{\sum_{k\in\mathcal N(i)}\exp\ell_{ik}},&
\mathbf m_i&=\sum_{j\in\mathcal N(i)}\alpha_{ij}\mathbf v_j.
\end{aligned}
$$

Suppose geometry enters only through $b_{ij}$. It can change how much of
$\mathbf v_j$ is routed to $i$, but it cannot change the content of
$\mathbf v_j$. In particular, if all neighboring values are identical,
$\mathbf v_j=\mathbf v$, then

$$
\mathbf m_i
=\sum_j\alpha_{ij}\mathbf v
=\mathbf v\sum_j\alpha_{ij}
=\mathbf v,
$$

independently of every distance and every $b_{ij}$. With $H$ heads,
geometry supplies at most $H$ scalar routing channels before the heads are
mixed. A vector-valued pair descriptor cannot generally be reconstructed from
that small set of normalized scalars.

This is not a universal impossibility theorem for transformers. A sufficiently
specialized transformer can encode relative geometry. It is a statement about
the bottleneck in the parameterization being tested: pair information affects
the weights, while the values remain receiver-independent node features.

### 4.2 A linear query/key map of raw coordinates does not naturally give distance

Assume raw absolute coordinates are included in the node input and the
coordinate-dependent part of an attention score is generated by linear query
and key maps. Its most general bilinear-affine form is

$$
s_{ij}
=\mathbf r_i^{\mathsf T}\mathbf A\mathbf r_j
+\mathbf a^{\mathsf T}\mathbf r_i
+\mathbf b^{\mathsf T}\mathbf r_j+c.
$$

Translate both atoms by $\mathbf t$. The change is

$$
\begin{aligned}
\Delta s_{ij}
={}&\mathbf t^{\mathsf T}\mathbf A\mathbf r_j
+\mathbf r_i^{\mathsf T}\mathbf A\mathbf t
+\mathbf t^{\mathsf T}\mathbf A\mathbf t
+(\mathbf a+\mathbf b)^{\mathsf T}\mathbf t.
\end{aligned}
$$

For this to vanish for all $\mathbf r_i,\mathbf r_j,\mathbf t$, one must
have

$$
\mathbf A=\mathbf 0,
\qquad
\mathbf a+\mathbf b=\mathbf 0.
$$

The remaining coordinate term is only
$\mathbf a^{\mathsf T}(\mathbf r_i-\mathbf r_j)$: a single linear directional
projection. It cannot represent
$\lVert\mathbf r_i-\mathbf r_j\rVert^2$, which is quadratic.

Distance *can* be obtained by a dot product if the coordinate embedding is
engineered to contain the required quadratic terms. Define

$$
\begin{aligned}
\boldsymbol\phi_Q(\mathbf r_i)
&=\left[\sqrt 2\,\mathbf r_i,\;-\lVert\mathbf r_i\rVert^2,\;1\right],\\
\boldsymbol\phi_K(\mathbf r_j)
&=\left[\sqrt 2\,\mathbf r_j,\;1,\;-\lVert\mathbf r_j\rVert^2\right].
\end{aligned}
$$

Then

$$
\boldsymbol\phi_Q(\mathbf r_i)^{\mathsf T}
\boldsymbol\phi_K(\mathbf r_j)
=2\mathbf r_i^{\mathsf T}\mathbf r_j
-\lVert\mathbf r_i\rVert^2
-\lVert\mathbf r_j\rVert^2
=-\lVert\mathbf r_i-\mathbf r_j\rVert^2.
$$

This construction is important: it shows that the limitation is not
representational impossibility. It also shows exactly what the network must
discover if distance is not provided—a matched nonlinear embedding of two
absolute positions, with cancellations that remain accurate under arbitrary
translations. Supplying $\mathbf r_{ij\mathbf S}$ and
$d_{ij\mathbf S}$ makes that relation local and exact from the first layer.

### 4.3 What the absolute PET must learn

In the implemented absolute mode, the center receives an embedded
$\mathbf r_i$, while an edge token receives the absolute periodic-image
position $\mathbf r_j+\mathbf S\mathbf L$. Even in a simplified linear
construction, recovering the displacement requires decoder/encoder products
$\mathbf D_n\mathbf A_n$ and $\mathbf D_c\mathbf A_c$ satisfying

$$
\mathbf D_n\mathbf A_n=\mathbf I,
\qquad
\mathbf D_c\mathbf A_c=-\mathbf I.
$$

Indeed, the reconstructed geometric component would be

$$
\mathbf D_n\mathbf A_n(\mathbf r_j+\mathbf S\mathbf L)
+\mathbf D_c\mathbf A_c\mathbf r_i
=\mathbf r_j+\mathbf S\mathbf L-\mathbf r_i.
$$

Translation invariance requires the cancellation

$$
(\mathbf D_n\mathbf A_n+\mathbf D_c\mathbf A_c)\mathbf t=\mathbf 0
$$

for every translation $\mathbf t$. Attention can in principle broadcast the
center representation to every edge token and later layers can learn this
subtraction. But this requires matched coordinate channels, calibrated
attention, and a cancellation that must remain valid across structures. The
relative representation hard-codes all three equalities before learning
begins. The absolute run therefore tests whether the network can *discover* a
good relational coordinate system, not whether absolute coordinates contain
enough information in an information-theoretic sense.

### 4.4 Fourier features can encode differences, but periodic images require care

For any wave vector $\mathbf k$, let

$$
\boldsymbol\phi_{\mathbf k}(\mathbf r)
=\left[\cos(\mathbf k^{\mathsf T}\mathbf r),
        \sin(\mathbf k^{\mathsf T}\mathbf r)\right].
$$

The dot product obeys

$$
\boldsymbol\phi_{\mathbf k}(\mathbf r_i)^{\mathsf T}
\boldsymbol\phi_{\mathbf k}(\mathbf r_j)
=\cos\!\left(\mathbf k^{\mathsf T}(\mathbf r_i-\mathbf r_j)\right).
$$

Thus matched Fourier query/key features can produce translation-invariant
relative scores. However, for integer-periodic fractional features and
$\mathbf S\in\mathbb Z^3$,

$$
\begin{aligned}
\sin\left(2\pi n(\Delta u+S)\right)
  &=\sin(2\pi n\Delta u),\\
\cos\left(2\pi n(\Delta u+S)\right)
  &=\cos(2\pi n\Delta u).
\end{aligned}
$$

Different periodic images of the same atom therefore collapse to the same
phase. This is correct for a function on the torus, but it is not sufficient to
distinguish the distinct Cartesian neighbor vectors required by a cutoff graph.
The image shift and cell metric must enter before those relations are merged.

### 4.5 Softmax suppresses coordination and multiplicity information

Attention is a normalized weighted average. If $m$ identical neighbor
relations are the only inputs, then

$$
\alpha_{ij}=\frac{1}{m},
\qquad
\sum_{j=1}^{m}\alpha_{ij}\mathbf v=\mathbf v,
$$

so the output is independent of $m$. In a mixed neighborhood, duplication can
alter the relative attention mass, but softmax remains an awkward mechanism for
learning extensive quantities such as coordination-dependent energy.

The same issue appears when periodic relations are reduced too early. If a
dense graph bias stores

$$
B_{ij}=\max_{\mathbf S}\log c(d_{ij\mathbf S}),
$$

then a graph containing one nearest image and a graph containing that same
image plus several more distant images have the same $B_{ij}$. This mapping is
many-to-one; no later transformer layer can recover the discarded image
multiplicity or directions.

## 5. Why the force plateau is especially informative

Consider a direct force head made only from rotation-invariant features
$\mathbf h(X)$:

$$
\mathbf F(X)=\mathbf W\mathbf h(X).
$$

If the desired force is equivariant, then for every rotation $\mathbf R$,

$$
\mathbf F(\mathbf R X)=\mathbf R\mathbf F(X).
$$

But invariance of the input features gives

$$
\mathbf F(\mathbf R X)
=\mathbf W\mathbf h(\mathbf R X)
=\mathbf W\mathbf h(X)
=\mathbf F(X).
$$

Consequently $\mathbf F(X)=\mathbf R\mathbf F(X)$ for every rotation. The
only vector satisfying this identity is the zero vector. Similarly, a direct
rank-two tensor built only from invariant features must satisfy

$$
\boldsymbol\sigma
=\mathbf R\boldsymbol\sigma\mathbf R^{\mathsf T}
\quad\text{for every }\mathbf R,
$$

which restricts it to $\lambda\mathbf I$; it cannot represent general
deviatoric stress.

The absolute PET is not literally restricted to invariant hidden features, so
this is not a proof that it must output zero. Rather, it explains the strong
inductive pressure: without a clean local covariant direction, a near-zero
vector is an easy way to fit rotationally diverse training data. A relative
edge supplies that direction directly. For example,

$$
\mathbf F_i
=\sum_{(i,j,\mathbf S)\in\mathcal E}
 a_{ij\mathbf S}\widehat{\mathbf r}_{ij\mathbf S},
$$

is equivariant whenever each $a_{ij\mathbf S}$ is a rotation-invariant
scalar. A stress-like symmetric tensor can similarly use dyads,

$$
\boldsymbol\sigma
\propto
\sum_{(i,j,\mathbf S)\in\mathcal E}
b_{ij\mathbf S}
\mathbf r_{ij\mathbf S}\otimes\mathbf r_{ij\mathbf S},
$$

with the exact sign, volume factor, and symmetrization chosen to match the
dataset convention.

## 6. Where v37 ST loses relational information

The current v37 implementation already computes pairwise fractional-coordinate
features; it is therefore not accurate to describe it as having no pairwise
calculation. The critical operations are instead:

1. Pair messages are summed and normalized into one vector per receiver node.
2. During transformer attention, graph geometry enters the logits as a scalar
   bias, while the value vector remains a function of the sender node alone.
3. Multiple periodic images associated with one atom pair can be collapsed when
   the graph bias is reduced with a maximum.

The corresponding implementation anchors are:

- PET selects absolute neighbor-image coordinates or relative displacement plus
  distance in
  [`src/metatrain/pet/modules/transformer.py`](../../src/metatrain/pet/modules/transformer.py#L485-L503).
- v37 turns pair features into one mean-aggregated receiver feature in
  [`src/structure_transformer_core/transformer.py`](../../src/structure_transformer_core/transformer.py#L280-L308).
- ST adds the graph term to attention logits, then multiplies the resulting
  weights by node-only values in
  [`src/structure_transformer_core/transformer.py`](../../src/structure_transformer_core/transformer.py#L495-L522).
- repeated image factors are reduced with `amax` in
  [`src/structure_transformer_core/graph_attention.py`](../../src/structure_transformer_core/graph_attention.py#L321-L340).

Schematically, the early compression is

$$
\widetilde{\mathbf h}_i
=\frac{1}{Z_i}\sum_j
\psi(\mathbf h_j,\Delta\mathbf u_{ij}),
$$

after which the individual relation $(i,j,\mathbf S)$ is no longer available
to the value path. Moreover, an encoder that sees
$\Delta\mathbf u_{ij}$ but not $\mathbf L$ before this aggregation cannot
in general construct
$d_{ij\mathbf S}^2=\Delta\mathbf u\,\mathbf L\mathbf L^{\mathsf T}
\Delta\mathbf u^{\mathsf T}$ at the pair level.

This gives a more specific hypothesis for the failure than “transformers cannot
learn relative distances”: the current architecture destroys or scalarizes the
pair relation before it can become the content of a receiver-specific message.

## 7. Proposed architecture: node states with relational attention

The goal is to retain ST's node-token backbone while giving each directed
periodic relation an explicit route into both the attention score and the value.
No edge state is carried between layers.

### 7.1 Static/on-the-fly pair descriptor

For every $\varepsilon=(i,j,\mathbf S)$, expand the distance using radial
basis functions, for example

$$
R_n(d_\varepsilon)
=c(d_\varepsilon)
\exp\!\left[-\gamma_n(d_\varepsilon-\mu_n)^2\right],
$$

where $c(d)$ is a smooth cutoff satisfying $c(d)=0$ for $d\ge r_c$.
Construct an invariant scalar descriptor

$$
\mathbf e_\varepsilon
=\operatorname{MLP}_{e}
\left(
  [\mathbf R(d_\varepsilon),
   \mathbf z_i,
   \mathbf z_j]
\right),
$$

where $\mathbf z_i,\mathbf z_j$ are atom-type embeddings. If angular
equivariance is needed inside the backbone, spherical harmonics or tensor
features of $\widehat{\mathbf r}_\varepsilon$ can be added in typed channels;
they should not be mixed into scalar channels without symmetry-aware products.

### 7.2 Pair-conditioned logit and value

For head $h$ in layer $\ell$, compute ordinary node queries, keys, and
values,

$$
\begin{aligned}
\mathbf q_i^h &=\mathbf W_Q^h\mathbf h_i^{(\ell)},\\
\mathbf k_j^h &=\mathbf W_K^h\mathbf h_j^{(\ell)},\\
\mathbf v_j^h &=\mathbf W_V^h\mathbf h_j^{(\ell)}.
\end{aligned}
$$

Let the edge descriptor generate a per-head bias, gate, and additive value:

$$
\begin{aligned}
b_\varepsilon^h
  &=(\mathbf w_b^h)^{\mathsf T}\mathbf e_\varepsilon,\\
\mathbf g_\varepsilon^h
  &=\operatorname{sigmoid}(\mathbf W_g^h\mathbf e_\varepsilon),\\
\mathbf p_\varepsilon^h
  &=\mathbf W_p^h\mathbf e_\varepsilon.
\end{aligned}
$$

Then define

$$
\begin{aligned}
\ell_\varepsilon^h
&=\frac{(\mathbf q_i^h)^{\mathsf T}\mathbf k_j^h}{\sqrt{d_h}}
  +b_\varepsilon^h
  +\log(c(d_\varepsilon)+\epsilon),\\
\alpha_\varepsilon^h
&=\underset{\varepsilon'\,:\,\operatorname{recv}(\varepsilon')=i}
  {\operatorname{segment\_softmax}}
  (\ell_\varepsilon^h),\\
\widetilde{\mathbf v}_\varepsilon^h
&=\mathbf g_\varepsilon^h\odot\mathbf v_j^h
  +\mathbf p_\varepsilon^h,\\
\mathbf m_{i,\mathrm{att}}^h
&=\sum_{\varepsilon\,:\,\operatorname{recv}(\varepsilon)=i}
  \alpha_\varepsilon^h\widetilde{\mathbf v}_\varepsilon^h.
\end{aligned}
$$

The decisive change is

$$
\widetilde{\mathbf v}_{ij\mathbf S}^h
\neq\widetilde{\mathbf v}_{kj\mathbf T}^h
\quad\text{in general},
$$

even when the sender node is the same. The message content now depends on the
receiver, distance, and periodic-image relation. In the invariant scalar
version above, equal-distance images have the same scalar descriptor;
directional dependence enters through an equivariant extension or the output
basis. Geometry is no longer limited to choosing among receiver-independent
values.

The gate and additive term serve different roles. The gate makes the sender's
chemical information distance-dependent; the additive term allows a geometric
message even when neighboring node values happen to be identical. Either one
could be ablated, but using both is the most expressive first test.

### 7.3 Preserve unnormalized density information

Add a parallel unnormalized local sum,

$$
\mathbf m_{i,\mathrm{sum}}
=\frac{1}{z_{\mathrm{ref}}}
\sum_{\varepsilon\,:\,\operatorname{recv}(\varepsilon)=i}
c(d_\varepsilon)
\mathbf U\left(
[\mathbf h_j^{(\ell)},\mathbf e_\varepsilon]
\right).
$$

Unlike softmax attention, this branch changes when identical neighbors are
added. It exposes coordination and periodic-image multiplicity directly. The
constant $z_{\mathrm{ref}}$ controls scale without erasing the count; it can
be a fixed expected coordination number or a dataset statistic. Dividing by the
actual local coordination $z_i$ would recreate the loss of multiplicity.

One layer can then update only the node state:

$$
\begin{aligned}
\mathbf u_i
&=\mathbf h_i^{(\ell)}
 +\mathbf W_O
  \operatorname{concat}_h(\mathbf m_{i,\mathrm{att}}^h)
 +\mathbf W_S\mathbf m_{i,\mathrm{sum}},\\
\mathbf h_i^{(\ell+1)}
&=\mathbf u_i+\operatorname{FFN}(\operatorname{Norm}(\mathbf u_i)),
\end{aligned}
$$

with the project's chosen pre-norm/post-norm convention applied consistently.
There is no $\mathbf g_{ij\mathbf S}^{(\ell+1)}$:
$\mathbf e_{ij\mathbf S}$ is static or recomputed from the current structure,
and only $\mathbf h_i$ persists. This is the cleanest test of whether
pair-conditioned message content is sufficient without PET-style edge tokens.

### 7.4 Local and global paths can coexist

The relational block need not replace global ST attention. A practical hybrid
is

$$
\mathbf h_i^{(\ell+1)}
=\mathbf h_i^{(\ell)}
+\mathcal A_{\mathrm{global}}(\mathbf h)_i
+\mathcal M_{\mathrm{local}}(\mathbf h,\mathcal E)_i
+\operatorname{FFN}(\cdot),
$$

where the global branch handles long-range/contextual interactions and the
local branch provides explicit metric geometry. A cell token can remain in the
global branch, while the local branch uses the cell matrix directly to construct
each Cartesian displacement.

## 8. Handling periodic images correctly

For the `nearest` experiment, retain one selected relation per physical atom
pair after the existing nearest-image canonicalization. This is the simplest
first comparison because it avoids changing both the geometric representation
and image multiplicity simultaneously.

For the `all` experiment, every $(i,j,\mathbf S)$ must remain a separate
entry through the segment softmax and the unnormalized sum:

$$
\mathbf m_i
=\sum_{(j,\mathbf S)\in\mathcal N(i)}
\mathcal M(\mathbf h_i,\mathbf h_j,
           \mathbf r_{ij\mathbf S},d_{ij\mathbf S}).
$$

Do not first reduce all shifts to one scalar $(i,j)$ bias using `max`. If a
dense representation is unavoidable, a less destructive scalar/vector
preaggregation is

$$
\mathbf E_{ij}
=\sum_{\mathbf S}
c(d_{ij\mathbf S})\phi(
\mathbf r_{ij\mathbf S},d_{ij\mathbf S}),
$$

but even this can cancel directional components and entangle distinct images.
A sparse relation list is the cleaner implementation.

Self-interactions require the usual distinction:

- remove $(i,i,\mathbf 0)$, the true zero-displacement self-loop;
- retain $(i,i,\mathbf S)$ for $\mathbf S\ne\mathbf 0$ when that periodic
  image is inside the cutoff.

## 9. Output heads

For energy-only training, use an invariant atomic readout and sum:

$$
E=\sum_i f_E(\mathbf h_i^{(L)}).
$$

The safest physically constrained force is conservative differentiation,

$$
\mathbf F_i=-\frac{\partial E}{\partial\mathbf r_i}.
$$

This also gives force equivariance automatically when the learned energy is
rotation invariant. With stacked column-vector coordinates
$\mathbf Y=\mathbf R\mathbf X$ and
$E(\mathbf R\mathbf X)=E(\mathbf X)$,

$$
\nabla_{\mathbf Y}E(\mathbf Y)
=\mathbf R\nabla_{\mathbf X}E(\mathbf X),
$$

by the chain rule applied to
$E(\mathbf Y)=E(\mathbf R^{\mathsf T}\mathbf Y)$. Therefore

$$
\mathbf F(\mathbf R\mathbf X)
=-\nabla_{\mathbf Y}E(\mathbf Y)
=\mathbf R\mathbf F(\mathbf X).
$$

If a direct force head is required, it should receive a covariant geometric
basis rather than map scalar node states directly to Cartesian components. One
simple form is

$$
\mathbf F_i
=\sum_{(i,j,\mathbf S)\in\mathcal E}
a_\theta(
\mathbf h_i^{(L)},\mathbf h_j^{(L)},\mathbf e_{ij\mathbf S})
\widehat{\mathbf r}_{ij\mathbf S}.
$$

If exact zero total internal force is desired, pair contributions can be made
antisymmetric between $(i,j,\mathbf S)$ and
$(j,i,-\mathbf S)$. That constraint should be introduced carefully when
external fields or other non-pairwise effects are in scope.

## 10. Decisive ablation sequence

Hold the data split, seed, optimizer, parameter scale, neighbor construction,
and output heads fixed. Begin with `nearest`, then repeat the winning variant
with `all` while keeping each periodic image separate.

| Variant | Pair information in logits | Pair information in values | Unnormalized local sum | Persistent edge state |
| --- | --- | --- | --- | --- |
| A: current graph ST | fixed cutoff bias | no | no | no |
| B: learned relational bias | learned RBF bias | no | no | no |
| C: relational values | learned RBF bias | yes | no | no |
| D: relational values + sum | learned RBF bias | yes | yes | no |
| E: PET reference | yes | yes | effectively retained by edge processing/readout | yes |

The comparisons answer different questions:

- **B succeeds over A:** a fixed monotone cutoff bias was too weak.
- **B fails but C succeeds:** the scalar-routing bottleneck, rather than graph
  topology, was the main failure.
- **C fails but D succeeds:** normalized attention was erasing important
  coordination or multiplicity information.
- **D approaches PET:** persistent edge tokens are not necessary; explicit
  pair-conditioned messages are sufficient.
- **D remains far behind PET:** repeated edge-state refinement or PET's
  edge-to-edge attention is a plausible remaining inductive bias to test.

The following diagnostics should accompany loss curves:

1. **Tiny-set overfit:** verify that each variant can nearly memorize a small
   subset. Failure indicates an architectural or implementation bottleneck,
   not a generalization problem.
2. **Translation test:** translate every position by the same vector; invariant
   outputs should be unchanged and forces should translate trivially.
3. **Rotation test:** rotate coordinates and cell together; energy should be
   invariant, forces equivariant, and stress transform as a rank-two tensor.
4. **Cell-wrapping test:** replace one atom by an equivalent periodic image;
   predictions should not change.
5. **Multiplicity test:** construct examples with the same nearest relation but
   different additional periodic images and verify that the `all` path can
   distinguish them.
6. **Gradient test:** log gradient norms for pair-bias, pair-value, and force
   direction paths to detect a disconnected or numerically suppressed branch.

Translation augmentation or centering absolute coordinates can be tested
separately, but neither substitutes for the decisive C-versus-B comparison.
Centering removes a global origin but still leaves the model to construct every
receiver-specific pair relation; relational values supply that structure
directly.

## 11. Conclusion

There is no theorem saying that a transformer must use literal edge tokens.
With suitable nonlinear positional features and enough capacity, self-attention
can represent relative distance. The important practical issue is the path by
which that information reaches a message.

The current evidence is most consistent with the following hierarchy:

$$
\text{topology only}
\;<\;
\text{geometry in scalar attention logits}
\;<\;
\text{geometry in pair-conditioned message values}
\;\leq\;
\text{persistent edge-token processing}.
$$

The first two inequalities are well motivated by the observed failures and the
bottleneck derivations above. The final inequality is an empirical question.
Implementing relational attention with node-only persistent state—especially
variant C followed by D—is the clean experiment that can answer it.

# Structure Transformer v37 Invariant Backbone Integration

This note documents the v37-style invariant backbone added to
`src/metatrain/experimental/structure_transformer/modules/transformer.py`. The
metatrain task is now direct prediction: the model receives atom types `A`,
Cartesian positions `X`, and cell matrices `L`, then predicts energy, forces, and
stress instead of flow-matching denoising targets.

## Configuration

The new knobs are exposed in
`src/metatrain/experimental/structure_transformer/documentation.py` through
`ModelHypers`, so they are available to normal metatrain YAML config loading:

```yaml
model:
  position_representation: fractional
  coordinate_encoding: v37_torus_relative
  coord_num_harmonics: 4
  use_rotary_embeddings: false
  use_periodic_rope: true
  atom_ordering: none
  center_positions: false
  fractional_wrap_eps: 1.0e-6
  atom_embedding_type: scalar
  atomic_number_scale: 118.0
  atom_scalar_embedding_scale: 1000.0
```

The defaults preserve the pre-existing backbone behavior:

```math
\texttt{coordinate\_encoding}=\texttt{absolute\_mlp},\qquad
\texttt{use\_periodic\_rope}=\texttt{false},\qquad
\texttt{atom\_embedding\_type}=\texttt{embedding}.
```

For the v37 invariant path, `StructureTransformer.__init__` rejects
`atom_ordering != "none"`, because sorted/index-based atom order injects an
extra coordinate-origin or sequence-index convention. It also rejects using
ordinary sequence RoPE and periodic coordinate RoPE at the same time.

## Inputs And Outputs

`StructureTransformerModel._systems_to_transformer_data` already converts a list
of `System` objects into

```math
A = (a_1,\ldots,a_N),\qquad X=(x_1,\ldots,x_N),\qquad
L\in\mathbb{R}^{3\times 3}.
```

The backbone converts Cartesian coordinates to fractional coordinates in
`StructureTransformer._dense_inputs` whenever the v37 path needs them:

```python
needs_fractional = (
    self.position_representation == "fractional"
    or self.atom_ordering != "none"
    or self.coordinate_encoding == "v37_torus_relative"
    or self.use_periodic_rope
)
frac_pos_dense = cartesian_to_fractional_dense(pos_dense, cell)
```

Mathematically, for row-vector coordinates, this is

```math
u_i = x_i L^{-1} \pmod 1,\qquad u_i\in[0,1)^3.
```

The heads remain prediction heads:

```math
E = \frac{1}{\bar N}\sum_i \phi_E(h_i) + \phi_E^{cell}(h_{cell}),
```

```math
F_i = \phi_F(h_i),\qquad
\sigma = \frac{1}{N}\sum_i \phi_\sigma(h_i) + \phi_\sigma^{cell}(h_{cell}).
```

These are implemented in `StructureTransformer.forward` as
`atom_energy_head`, `force_head`, `atom_stress_head`, `cell_energy_head`, and
`cell_stress_head`.

## v37 Coordinate Encoder

The v37 coordinate path is implemented by `TorusRelativeCoordEncoder.forward`.
First, fractional coordinates are canonicalized by `_canonical_frac_coords`:

```python
frac_coords = torch.remainder(frac_coords, 1.0)
```

This preserves the torus equivalence

```math
u_i \sim u_i + b_i,\qquad b_i\in\mathbb{Z}^3.
```

For each ordered atom pair, the encoder forms

```python
delta = u.unsqueeze(2) - u.unsqueeze(1)
```

that is,

```math
\Delta_{ij}=u_i-u_j.
```

It then builds integer-harmonic Fourier features

```math
\phi(\Delta_{ij}) =
\left[\cos(2\pi m\Delta_{ij,d}),\ \sin(2\pi m\Delta_{ij,d})\right]_
{d\in\{1,2,3\},\ m=1,\ldots,M}.
```

The learned pair message is

```math
m_{ij}=\operatorname{MLP}\left(\left[\phi(\Delta_{ij}), h_j^A\right]\right),
```

where `h_j^A` is the atom-type feature from `_encode_atoms`. The atom coordinate
feature is the masked neighbor average

```math
h_i^X = \frac{1}{|\mathcal{N}_i|}\sum_{j\in\mathcal{N}_i} m_{ij},\qquad
\mathcal{N}_i=\{j: j\neq i\}.
```

In code, this is the `pair_mlp`, `valid_pair` mask, and
`messages.sum(dim=2) / denom` block inside `TorusRelativeCoordEncoder.forward`.

## Periodic Coordinate RoPE

The v37 attention path is implemented by
`_precompute_periodic_3d_rope_harmonics`, `_apply_periodic_3d_rotary_emb`, and
`MultiHeadAttention.forward`. For each two-dimensional head channel pair, v37
assigns a coordinate axis and an integer harmonic:

```python
pair_axis = pair_index % 3
harmonic = pair_index // 3 + 1
```

For atom `i`, channel pair `p`, assigned axis `d(p)`, and harmonic `m_p`, the
rotation angle is

```math
\theta_{i,p}=2\pi m_p u_{i,d(p)}.
```

A query/key pair is rotated by

```math
R(\theta_{i,p})
\begin{bmatrix}z_{2p}\\ z_{2p+1}\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_{i,p} & -\sin\theta_{i,p}\\
\sin\theta_{i,p} & \cos\theta_{i,p}
\end{bmatrix}
\begin{bmatrix}z_{2p}\\ z_{2p+1}\end{bmatrix}.
```

Atom-atom attention scores therefore contain relative phases:

```math
\left(R(\theta_i)q_i\right)^T\left(R(\theta_j)k_j\right)
= q_i^T R(\theta_j-\theta_i) k_j.
```

`MultiHeadAttention.forward` applies these rotations only to atom tokens
`[:, :, 1:, :]`. It then overwrites every attention score involving the
lattice/CLS token with the unrotated dot product:

```python
attn[:, :, 0, :] = unrotated_attn[:, :, 0, :]
attn[:, :, :, 0] = unrotated_attn[:, :, :, 0]
```

The CLS token has no physical fractional coordinate, so this avoids choosing an
arbitrary origin for it.

## Symmetry Bookkeeping

Let `B=(b_1, ..., b_N)` with `b_i in Z^3`, let

```math
(T_c U)_i = (u_i+c)\bmod 1,\qquad c\in\mathbb{R}^3,
```

and let `P` permute atom rows of both `A` and `U`. The v37 path is designed to
satisfy

```math
E(A,U+B,L)=E(A,U,L),\qquad
\sigma(A,U+B,L)=\sigma(A,U,L),\qquad
F(A,U+B,L)=F(A,U,L),
```

```math
E(A,T_cU,L)=E(A,U,L),\qquad
\sigma(A,T_cU,L)=\sigma(A,U,L),\qquad
F(A,T_cU,L)=F(A,U,L),
```

```math
E(PA,PU,L)=E(A,U,L),\qquad
\sigma(PA,PU,L)=\sigma(A,U,L),\qquad
F(PA,PU,L)=P F(A,U,L).
```

The important operations preserve these symmetries as follows:

| Operation | Periodic wrapping | Global translation | Permutation | Code location |
| --- | --- | --- | --- | --- |
| Fractional conversion and `torch.remainder` | Enforces `u_i = u_i+b_i` | Preserves once relative features are used | Row-wise equivariant | `_dense_inputs`, `_canonical_frac_coords` |
| Pair differences `u_i-u_j` | Needs Fourier periodicity for wrap jumps | Removes common `c` | Pair-tensor equivariant | `TorusRelativeCoordEncoder.forward` |
| Integer Fourier features | Enforces unit periodicity | Makes translated wrap jumps invisible | Pair-tensor equivariant | `angles.cos()`, `angles.sin()` |
| Shared pair MLP and neighbor average | Preserves | Preserves | Atom-wise equivariant | `pair_mlp`, `valid_pair`, `messages.sum(dim=2)` |
| No sequence-index PE in v37 mode | Neutral | Neutral | Avoids index leakage | `use_rotary_embeddings=False` |
| Integer-harmonic coordinate RoPE | Enforces periodic relative phases | Attention scores depend on `u_j-u_i` | Atom-wise equivariant | `_apply_periodic_3d_rotary_emb` |
| Unrotated CLS interactions | Preserves | Preserves | Lets global token stay invariant | `MultiHeadAttention.forward` |
| Energy/stress reductions | Preserves | Preserves | Invariant system outputs | `StructureTransformer.forward` |
| Force head applied per atom | Preserves | Preserves | Equivariant atom output | `StructureTransformer.forward` |

## Differences From The Notebook

The reference notebook `181_fm_llama_mp20_vsun_xb_v37.ipynb` was a generation
model. It had flow state `x_t`, atom state `a_t`, lattice state `l_t`, and a
time embedding `p_t`. The metatrain version removes the time input because the
task is supervised prediction. The existing cell token and per-layer
`cell_condition_encoder` remain, so lattice information still conditions every
transformer block.

The notebook used scalar Fourier atom encoding by default:

```math
h_i^A = \operatorname{MLP}(\operatorname{scalar\_embedding}(a_i)).
```

The metatrain implementation supports this with `atom_embedding_type: scalar`,
but keeps the old learned embedding-table path as the default for backwards
compatibility.

The optional edge-vector head was left unchanged. It uses neighbor vectors in
Cartesian space and is not part of the v37 invariant backbone tests.


## Training Config

I added `options-structure-transformer-mptrj-salex-direct-160k-v37.yaml` as a
v37-style variant of `options-structure-transformer-mptrj-salex-direct-160k.yaml`.
It keeps the same MPtrj/sAlex data, optimization schedule, target definitions,
and model size, but changes the backbone settings that determine the symmetry
behavior. It also sets `architecture.training.use_data_augmentation: false`, so
training does not apply the PET random rotation/inversion augmenter:

```yaml
architecture:
  name: experimental.structure_transformer
  training:
    use_data_augmentation: false
  model:
    position_representation: fractional
    coordinate_encoding: v37_torus_relative
    coord_num_harmonics: 4
    coord_encoder_chunk_size: 32
    coord_encoder_use_checkpoint: true
    use_rotary_embeddings: false
    use_periodic_rope: true
    atom_ordering: none
    center_positions: false
    fractional_wrap_eps: 1.0e-6
    atom_embedding_type: scalar
    atomic_number_scale: 118.0
    atom_scalar_embedding_scale: 1000.0
```

The important differences from the starting config are:

- `coordinate_encoding: v37_torus_relative` replaces the absolute coordinate MLP
  with pairwise periodic Fourier features of fractional coordinate differences.
- `use_periodic_rope: true` enables integer-harmonic coordinate RoPE in
  atom-atom attention.
- `use_rotary_embeddings: false` disables sequence-index RoPE, which would inject
  token-index information.
- `atom_ordering: none` avoids sorting or rank-based position IDs, so atom order
  is handled by the transformer as a set order rather than as positional metadata.
- `atom_embedding_type: scalar` matches the notebook's scalar Fourier encoding
  for atom numbers; the embedding-table path remains available through config.
- `coord_encoder_chunk_size: 32` and `coord_encoder_use_checkpoint: true`
  reduce the training memory of the v37 all-pairs coordinate encoder without
  removing large structures from the dataset.
- `architecture.training.use_data_augmentation: false` removes
  the PET `RotationalAugmenter` from the training path, disabling both random
  rotations and random inversions.

### Memory and Large MPtrj Structures

The MPtrj EDA notebook at
`/home/ryoji/equiformer_v3/experimental/datasets/mptrj_metadata_eda.ipynb`
shows that the 160k split has a long atom-count tail: maximum 444 atoms,
99th percentile 144 atoms, and 99.9th percentile 216 atoms. A hard cutoff such
as `max_atoms_per_batch=128` would therefore remove meaningful large-cell
training examples.

The v37 memory issue comes from `TorusRelativeCoordEncoder.forward`, not from the
parameter count. For a padded batch with `B` structures and `N_max` atoms, the
full receiver-neighbor coordinate encoder forms all-pairs features

```math
\Delta u_{b i j}=\operatorname{wrap}(u_{b i}-u_{b j})\in\mathbb{T}^3,
```

then evaluates

```math
 h_{b i j}=\operatorname{MLP}\left([
    \phi(\Delta u_{b i j}), z_{b j}
 ]\right)\in\mathbb{R}^{D},
```

where the hidden layer inside that MLP has width `H = encoder_hidden_dim`. The
large saved activation has scale

```math
\Theta(B N_{\max}^{2} H).
```

This is why v37 can OOM even when a larger parameter-count model fits: v37 adds a
wide all-pairs pair MLP before the transformer blocks.

To keep the large structures, `TorusRelativeCoordEncoder.forward` now chunks over
receiver atoms:

```python
for start in range(0, num_atoms, chunk_size):
    end = min(start + chunk_size, num_atoms)
    chunk = self._forward_receiver_chunk(u, atom_features, atom_mask, start, end)
```

For chunk size `C`, the largest per-chunk pair hidden tensor is

```math
H_{\mathrm{chunk}}\in\mathbb{R}^{B\times C\times N_{\max}\times H},
```

so the pair-MLP peak memory changes from approximately

```math
B N_{\max}^{2} H \quad\text{to}\quad B C N_{\max} H.
```

The arithmetic is still all-pairs, `\Theta(BN_{\max}^{2})`, but the peak memory
is smaller by roughly

```math
\frac{C}{N_{\max}}.
```

With `coord_encoder_chunk_size: 32` and a 400-atom structure, that pair-hidden
peak is about `32 / 400 = 0.08` of the unchunked receiver dimension. The config
also sets `coord_encoder_use_checkpoint: true`, which routes each receiver chunk
through

```python
activation_checkpoint(chunk_fn, u, atom_features, atom_mask, use_reentrant=False)
```

During training, checkpointing avoids storing the chunk's internal pair-MLP
activations and recomputes them during backward. This trades extra compute for a
much lower backward memory footprint, which is the right tradeoff before cutting
large MPtrj structures.

If this still OOMs, the next knobs to try are lower chunk sizes before lowering
atom coverage:

```bash
-r architecture.model.coord_encoder_chunk_size=16 \
-r architecture.model.coord_encoder_use_checkpoint=true
```

PyTorch allocator fragmentation can also be reduced with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### What Changed for the OOM Fix

The fix is an exact refactor of the v37 coordinate encoder computation, plus two
new hyperparameters that expose the memory behavior in config:

```python
coord_encoder_chunk_size: Optional[int] = None
coord_encoder_use_checkpoint: bool = False
```

These are accepted by `StructureTransformer.__init__`, validated as positive
when present, stored on the model, and passed into `TorusRelativeCoordEncoder`:

```python
self.position_encoder = TorusRelativeCoordEncoder(
    embed_dim=embed_dim,
    hidden_dim=encoder_hidden_dim,
    num_harmonics=coord_num_harmonics,
    wrap_eps=self.fractional_wrap_eps,
    chunk_size=coord_encoder_chunk_size,
    use_checkpoint=coord_encoder_use_checkpoint,
)
```

Inside `TorusRelativeCoordEncoder`, I split the original all-receiver computation
into `_forward_receiver_chunk`. For a receiver slice

```math
I_s=\{s,s+1,\ldots,e-1\},\quad |I_s|=C_s,
```

that function computes only the pair tensor whose receiver index is in `I_s`:

```python
receiver_coords = u[:, start:end, :]
delta = receiver_coords.unsqueeze(2) - u.unsqueeze(1)
```

Mathematically, this is the restricted pair-difference tensor

```math
\Delta u^{(s)}_{b i j}=u_{b i}-u_{b j},
\quad i\in I_s,
\quad j\in\{1,\ldots,N_{\max}\}.
```

The rest of the per-pair computation is unchanged. The same Fourier features are
built, the same pair MLP is applied, masked self-pairs and padding are removed,
and neighbor messages are averaged:

```python
pair_features = torch.cat((angles.cos(), angles.sin()), dim=-1)
pair_input = torch.cat((pair_features, neighbor_features), dim=-1)
messages = self.pair_mlp(pair_input)
coord_features = messages.sum(dim=2) / denom.to(dtype=messages.dtype)
```

So the chunked output is exactly the concatenation of the same per-receiver
features the unchunked encoder would have produced:

```math
Z_{\mathrm{coord}}
=\operatorname{concat}_{s}
  \left(
    \frac{1}{|\mathcal{N}_i|}
    \sum_{j\in\mathcal{N}_i}
    \operatorname{MLP}([
      \phi(u_i-u_j), z_j
    ])
  \right)_{i\in I_s}.
```

The main `forward` method now loops over receiver chunks:

```python
chunk_size = num_atoms if self.chunk_size is None else self.chunk_size
chunks = []
for start in range(0, num_atoms, chunk_size):
    end = min(start + chunk_size, num_atoms)
    chunk = self._forward_receiver_chunk(u, atom_features, atom_mask, start, end)
    chunks.append(chunk)
return torch.cat(chunks, dim=1)
```

When `coord_encoder_use_checkpoint: true`, training calls the chunk through
`torch.utils.checkpoint`. The defaults `chunk_start=start` and `chunk_end=end`
are intentionally bound in the nested function so backward replays the same
slice that was used in forward:

```python
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
```

The v37 YAML enables this path with:

```yaml
coord_encoder_chunk_size: 32
coord_encoder_use_checkpoint: true
```

Let

```math
B=\text{number of structures in the dense batch},\quad
N=N_{\max},\quad
C=\text{receiver chunk size},
```

```math
D=\texttt{embed\_dim},\quad
H=\texttt{encoder\_hidden\_dim},\quad
M=\texttt{coord\_num\_harmonics}.
```

The per-pair Fourier feature width is

```math
F_{\phi}=3\times 2M=6M,
```

because each of the three fractional coordinates gets `cos` and `sin` features
for `M` integer harmonics. The pair MLP receives width `F_phi + D`, expands to
`H`, and returns width `D`:

```math
\mathbb{R}^{6M+D}\rightarrow\mathbb{R}^{H}\rightarrow\mathbb{R}^{D}.
```

The total coordinate-encoder arithmetic is still all-pairs:

```math
T_{\mathrm{coord}}
=\Theta\left(
  B N^2\left[M+(6M+D)H+HD+D\right]
\right).
```

Chunking does not change this asymptotic compute; it changes the largest tensor
that must exist at once. Without chunking, the hidden pair activation is shaped
approximately

```math
B\times N\times N\times H.
```

With receiver chunks, the largest hidden pair activation is only

```math
B\times C\times N\times H.
```

So the coordinate-encoder peak activation memory drops from roughly

```math
S_{\mathrm{full}}=\Theta(BN^2(H+D+6M))
```

to the per-chunk peak

```math
S_{\mathrm{chunk}}=\Theta(BCN(H+D+6M))+\Theta(BND).
```

The `\Theta(BND)` term is the final per-atom coordinate feature output that must
still be kept for the transformer. Without checkpointing, autograd may still
retain many chunk internals for backward, so chunking alone mostly reduces
transient peak allocation. With checkpointing, those internals are not stored;
they are recomputed during backward. The training compute becomes approximately

```math
T_{\mathrm{train,ckpt}}
\approx T_{\mathrm{train,no\ ckpt}}+T_{\mathrm{coord\ forward}},
```

which is a deliberate compute-for-memory tradeoff.

This refactor preserves the v37 symmetries because it does not introduce any
receiver-index feature or order-dependent reduction. It only partitions the same
set of receiver atoms into temporary slices:

```math
\operatorname{concat}_{s} f_{I_s}(A,U,L)=f(A,U,L).
```

The regression tests now check both sides of that claim: chunked and unchunked
models with identical weights produce the same outputs, and checkpointed chunks
can backpropagate through `position_encoder.pair_mlp`.

Run a single-process CUDA training job with:

```bash
/home/ryoji/miniconda3/envs/metatrain-pet/bin/python -m metatrain train \
  options-structure-transformer-mptrj-salex-direct-160k-v37.yaml \
  --output outputs/structure-transformer-metatrain-160k-v37/model.pt
```

Your current DDP command can be run as-is with this config because the YAML already
sets `use_data_augmentation: false`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  -m metatrain train options-structure-transformer-mptrj-salex-direct-160k-v37.yaml \
  -o structure-transformer-mptrj-salex-ddp-160k-v37.pt \
  -r device=cuda \
  -r architecture.training.distributed=true \
  -r architecture.training.distributed_port=39591 \
  -r training_set.indices=indices/mptrj_160k_seed0.txt
```

For any config that still has augmentation enabled, add this override:

```bash
-r architecture.training.use_data_augmentation=false
```

Or, if `mtt` is on your active environment's `PATH`, the equivalent command is:

```bash
mtt train options-structure-transformer-mptrj-salex-direct-160k-v37.yaml \
  --output outputs/structure-transformer-metatrain-160k-v37/model.pt
```

For a quick CPU plumbing check, override the device and shrink the run at the
command line:

```bash
/home/ryoji/miniconda3/envs/metatrain-pet/bin/python -m metatrain train \
  options-structure-transformer-mptrj-salex-direct-160k-v37.yaml \
  --output outputs/structure-transformer-metatrain-160k-v37-smoke/model.pt \
  -r device=cpu \
  -r architecture.training.num_epochs=1 \
  -r architecture.training.batch_size=1 \
  -r architecture.training.max_atoms_per_batch=64 \
  -r wandb.mode=disabled
```

## Verification

The v37 config was checked with OmegaConf and direct `StructureTransformer`
instantiation from `architecture.model`, after removing the wrapper-only
`symmetrize_stress` key. The documented `python -m metatrain train` command was
also checked against the local CLI help in the `metatrain-pet` environment.

The new test file
`src/metatrain/experimental/structure_transformer/tests/test_v37_invariant_backbone.py`
checks:

1. integer fractional shifts leave energy, stress, and forces unchanged;
2. a common real fractional translation leaves energy, stress, and forces
   unchanged;
3. atom permutation leaves energy and stress unchanged and permutes forces;
4. chunked v37 coordinate encoding matches the unchunked encoder output;
5. checkpointed coordinate chunks support backpropagation;
6. the new hyperparameters appear in `get_default_hypers` and pass through
   `StructureTransformerModel`;
7. invalid v37 combinations, such as index ordering, double RoPE, or nonpositive
   coordinate chunk sizes, are rejected.

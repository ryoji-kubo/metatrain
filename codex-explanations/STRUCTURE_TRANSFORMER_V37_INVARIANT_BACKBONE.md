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
- `architecture.training.use_data_augmentation: false` removes
  the PET `RotationalAugmenter` from the training path, disabling both random
  rotations and random inversions.

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
4. the new hyperparameters appear in `get_default_hypers` and pass through
   `StructureTransformerModel`;
5. invalid v37 combinations, such as index ordering or double RoPE, are rejected.

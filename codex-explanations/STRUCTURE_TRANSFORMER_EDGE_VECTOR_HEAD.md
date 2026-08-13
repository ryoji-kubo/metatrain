# Structure Transformer Edge-Vector Head

This note explains the optional edge-vector prediction head added to
`experimental.structure_transformer`.

The goal of this change is diagnostic. The original StructureTransformer remains a
global sequence model over atom and cell tokens. When the new flag is disabled, the
model does not request neighbor lists and uses the same direct atom/cell readouts as
before. When the flag is enabled, the transformer backbone is unchanged, but force
and stress predictions receive an explicit local Cartesian edge-vector readout at
the prediction head.

## Why This Was Added

The failed StructureTransformer runs looked like they were learning energy but not
learning useful force or stress directions. PET, although not equivariant, has a
strong Cartesian local-geometry bias: it builds pair edge vectors and uses them in
its transformer/readout path.

This head tests a narrower hypothesis:

> Is the StructureTransformer backbone producing useful scalar/context features, but
> missing an easy way to turn local Cartesian geometry into vector and tensor outputs?

So this implementation does not add graph message passing inside the transformer.
It only adds graph geometry at the prediction head.

## Flags

The new model hyperparameters live under `architecture.model`:

```yaml
edge_vector_head: false
edge_vector_head_cutoff: 10.0
edge_vector_head_hidden_dim: 256
edge_vector_head_num_radial_basis: 16
edge_vector_head_replace_direct: false
```

`edge_vector_head` is the main switch. The default is `false`, which preserves the
original implementation.

`edge_vector_head_cutoff` controls the neighbor-list cutoff used only by this head.

`edge_vector_head_hidden_dim` controls the hidden size of the small edge MLPs.

`edge_vector_head_num_radial_basis` controls how many Gaussian radial basis values
are used to encode edge distance.

`edge_vector_head_replace_direct` controls whether the edge head is added to the
original direct force/stress heads or replaces them:

```text
false: output = original_direct_head + edge_vector_head
true:  output = edge_vector_head only
```

For the cleanest diagnostic, use `edge_vector_head_replace_direct: true`. That asks
whether the local edge-vector readout alone can escape the zero-force-like behavior.

## Architecture

The transformer backbone still receives the same dense sequence:

```text
[cell token, atom token 1, atom token 2, ...]
```

Each atom token is built from an atom embedding plus an encoded position. The cell
token and per-layer cell conditioning are unchanged.

When `edge_vector_head: true`, the metatrain wrapper requests a full neighbor list:

```python
NeighborListOptions(
    cutoff=edge_vector_head_cutoff,
    full_list=True,
    strict=True,
)
```

The wrapper extracts, for all systems in the batch:

```text
edge_vectors:  r_j - r_i + periodic cell shift
edge_centers:  i
edge_neighbors: j
```

These edge vectors come from the metatomic neighbor-list block. No extra periodic
geometry reconstruction is needed in the StructureTransformer wrapper.

## Edge Features

After the transformer finishes, the final atom features are flattened back to the
original atom order. For each edge `i -> j`, the head builds:

```text
[projected h_i, projected h_j, radial_basis(|r_ij|)]
```

where:

- `h_i` is the final transformer feature for the center atom,
- `h_j` is the final transformer feature for the neighbor atom,
- `radial_basis(|r_ij|)` is a Gaussian expansion of the edge length.

This feature vector goes through a small MLP that predicts one scalar per edge.

## Force Readout

For forces, the edge MLP predicts a scalar coefficient `a_ij`. The vector direction
comes directly from the unit edge vector:

```text
u_ij = r_ij / |r_ij|
f_i += a_ij * u_ij
```

The contributions are scatter-added onto center atoms. Because the neighbor list is
full, both directions of a pair can contribute separately if present in the list.

This is intentionally simple. It gives the model a direct path from learned scalar
features plus local Cartesian directions to force vectors.

## Stress Readout

For stress, the edge MLP predicts a scalar coefficient `b_ij`. The tensor direction
comes from the edge-direction dyad:

```text
S_system += b_ij * outer(u_ij, u_ij)
```

The edge tensor contributions are scatter-added per system and normalized by the
number of edges in that system.

This is a diagnostic stress head, not a full virial-style derivation. Its purpose is
to test whether local orientation information at the readout helps the model learn a
rank-2 Cartesian target.

## Default Behavior

With:

```yaml
edge_vector_head: false
```

the model:

- does not request neighbor lists,
- does not build edge features,
- does not create edge force/stress heads,
- uses the original direct `force_head`, `atom_stress_head`, and `cell_stress_head`.

The edge-head sizing/cutoff knobs are validated only when `edge_vector_head: true`,
so disabled edge options are inert.

## Minimal Diagnostic Configuration

Start from the Cartesian 200M config:

```text
options-structure-transformer-mptrj-salex-direct-200m-cartesian.yaml
```

Add this under `architecture.model`:

```yaml
    edge_vector_head: true
    edge_vector_head_cutoff: 10.0
    edge_vector_head_hidden_dim: 256
    edge_vector_head_num_radial_basis: 16
    edge_vector_head_replace_direct: true
```

For example, the relevant model section should contain:

```yaml
architecture:
  name: experimental.structure_transformer

  model:
    position_representation: cartesian
    regress_forces: true
    regress_stress: true
    direct_prediction: true

    edge_vector_head: true
    edge_vector_head_cutoff: 10.0
    edge_vector_head_hidden_dim: 256
    edge_vector_head_num_radial_basis: 16
    edge_vector_head_replace_direct: true
```

Use `edge_vector_head_replace_direct: true` first. If that learns force directions
substantially better than the original model, then the issue is likely not just the
transformer backbone capacity; it is very likely the missing local Cartesian
readout bias.

After that, a second run with:

```yaml
edge_vector_head_replace_direct: false
```

tests whether the edge-vector readout is best used as a residual correction on top
of the original direct heads.

## How To Run

Create a copied options file so the baseline config stays untouched:

```bash
cp options-structure-transformer-mptrj-salex-direct-200m-cartesian.yaml \
  options-structure-transformer-mptrj-salex-direct-200m-cartesian-edge-head.yaml
```

Edit the copied YAML and add the `edge_vector_head` block under
`architecture.model`.

Then run training from the repository root:

```bash
/home/ryoji/miniconda3/envs/metatrain-pet/bin/python -m metatrain train \
  options-structure-transformer-mptrj-salex-direct-200m-cartesian-edge-head.yaml \
  -o structure-transformer-mptrj-salex-ddp-200m-cartesian-edge-head.pt
```

The shorter CLI form should also work if `mtt` is on `PATH` in the active
environment:

```bash
mtt train options-structure-transformer-mptrj-salex-direct-200m-cartesian-edge-head.yaml \
  -o structure-transformer-mptrj-salex-ddp-200m-cartesian-edge-head.pt
```

For a smaller smoke run, copy the same edge-head flags into the 160k config and
lower `num_epochs` in the copied YAML:

```bash
cp options-structure-transformer-mptrj-salex-direct-160k.yaml \
  options-structure-transformer-mptrj-salex-direct-160k-edge-head-smoke.yaml
```

Then run:

```bash
/home/ryoji/miniconda3/envs/metatrain-pet/bin/python -m metatrain train \
  options-structure-transformer-mptrj-salex-direct-160k-edge-head-smoke.yaml \
  -o structure-transformer-mptrj-salex-160k-edge-head-smoke.pt
```

## What To Look For

The most important diagnostic is validation force error. The failed
StructureTransformer runs stayed close to the zero-force baseline. A useful edge
head should quickly move validation force RMSE/MAE below that baseline.

Check:

- `outputs/<date>/<time>/train.csv`
- validation `non_conservative_force` RMSE/MAE columns,
- validation stress RMSE/MAE,
- whether force cosine similarity moves away from zero if logged.

Interpretation:

- If `edge_vector_head_replace_direct: true` improves forces, the missing local
  edge-vector readout was probably a major issue.
- If it does not improve forces, the problem is likely earlier: representation,
  optimization, target scaling, augmentation, or insufficient useful geometric
  information in the transformer features.
- If `replace_direct: false` works better than `true`, the original direct head may
  still carry useful global/context information, while the edge head supplies the
  missing local direction bias.

## Files

Main implementation files:

- `src/metatrain/experimental/structure_transformer/documentation.py`
- `src/metatrain/experimental/structure_transformer/model.py`
- `src/metatrain/experimental/structure_transformer/modules/transformer.py`

Focused tests:

- `src/metatrain/experimental/structure_transformer/tests/test_edge_vector_head.py`


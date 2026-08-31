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

## Appendix: How PET Uses Edge Vectors In Its Readout Path

PET does not use edge vectors only at the final prediction head. The edge vectors
enter the local transformer backbone first, then the target-specific readout consumes
the resulting edge features.

The path is:

```text
System neighbor list
  -> Cartesian edge vectors and distances
  -> adaptive/cutoff filtering
  -> NEF edge tensor [n_atoms, max_neighbors, 3]
  -> CartesianTransformer edge tokens
  -> target-specific edge heads
  -> cutoff-weighted sum over neighbors
  -> node contribution + edge contribution
```

### 1. PET Builds Cartesian Edge Vectors

In `src/metatrain/pet/modules/structures.py`, `systems_to_batch(...)` first
concatenates the structures and reads the neighbor-list samples. It then reconstructs
periodic displacements:

```python
cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
edge_vectors = positions[neighbors] - positions[centers] + cell_contributions
edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15
```

For multiple systems, PET uses the cell belonging to the center atom's system:

```python
cell_contributions = torch.einsum(
    "ab, abc -> ac",
    cell_shifts.to(cells.dtype),
    cells[system_indices[centers]],
)
```

So PET's geometric edge input is:

```text
r_ij = x_j - x_i + cell_shift @ cell
```

with both the vector `r_ij` and scalar distance `|r_ij|`.

### 2. PET Reduces The Raw Neighbor Set

If `num_neighbors_adaptive` is set, PET does not simply keep every edge within the
global cutoff. It estimates an atom-wise adaptive cutoff, symmetrizes it across the
pair, and drops edges outside that pair cutoff:

```python
pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[neighbors]) / 2.0
keep = torch.nonzero(edge_distances <= pair_cutoffs).squeeze(-1)
edge_vectors = edge_vectors.index_select(0, keep)
edge_distances = edge_distances.index_select(0, keep)
```

Then PET computes a smooth cutoff factor:

```python
cutoff_factors = cutoff_func_bump(edge_distances, pair_cutoffs, cutoff_width)
```

or, for fixed-cutoff configs:

```python
cutoff_factors = cutoff_func_cosine(edge_distances, pair_cutoffs, cutoff_width)
```

This matters for the comparison with the StructureTransformer diagnostic head. PET's
successful MPtrj/sAlex config has:

```yaml
cutoff: 10.0
num_neighbors_adaptive: 40
cutoff_function: Bump
cutoff_width: 0.5
```

So although the nominal cutoff is 10 A, PET is not using an unweighted sum over every
10 A edge. It adapts the effective neighborhood and applies a smooth envelope.

### 3. PET Converts Edges To NEF Format

PET converts the flat edge array into a padded node-edge-feature layout:

```python
edge_vectors = edge_array_to_nef(edge_vectors, nef_indices)
edge_distances = torch.sqrt(torch.sum(edge_vectors**2, dim=2) + 1e-15)
cutoff_factors = edge_array_to_nef(cutoff_factors, nef_indices, nef_mask, 0.0)
```

After this conversion:

```text
edge_vectors:    [n_atoms, max_neighbors, 3]
edge_distances:  [n_atoms, max_neighbors]
cutoff_factors:  [n_atoms, max_neighbors]
padding_mask:    [n_atoms, max_neighbors]
```

PET also computes a reverse-neighbor index. Between GNN layers, this lets PET replace
or combine an `i -> j` message with the corresponding `j -> i` message.

### 4. Edge Vectors Become Edge Tokens Inside CartesianTransformer

In `src/metatrain/pet/modules/transformer.py`, `CartesianTransformer.forward(...)`
embeds the Cartesian edge vector and the edge distance together:

```python
edge_embeddings = [edge_vectors, edge_distances[:, :, None]]
edge_embeddings = torch.cat(edge_embeddings, dim=2).to(edge_vectors.dtype)
edge_embeddings = self.edge_embedder(edge_embeddings)
```

The `edge_embedder` is:

```python
self.edge_embedder = nn.Linear(4, d_model)
```

So each edge token initially receives:

```text
[dx, dy, dz, distance] -> Linear(4, d_model)
```

On the first PET GNN layer, this geometric edge embedding is concatenated with the
current edge/message embedding:

```python
edge_tokens = torch.cat([edge_embeddings, input_messages], dim=2)
```

On later GNN layers, PET also concatenates neighbor element embeddings:

```python
neighbor_elements_embeddings = self.neighbor_embedder(element_indices_neighbors)
edge_tokens = torch.cat(
    [edge_embeddings, neighbor_elements_embeddings, input_messages],
    dim=2,
)
```

Then PET compresses these into transformer edge tokens:

```python
edge_tokens = self.compress(edge_tokens)
```

The local transformer attends over one center-node token plus that atom's edge tokens.
The attention mask uses the cutoff factors:

```python
cutoff_factors = torch.cat([central_token_factor, cutoff_factors], dim=1)
cutoff_factors[~total_padding_mask] = 0.0
```

Inside attention, PET adds `log(cutoff_factors)` to the attention weights. Thus edge
distance/cutoff affects the transformer attention, not just the final readout.

### 5. PET Reuses Edge Vectors At Every GNN Layer

In `src/metatrain/pet/model.py`, `_calculate_features(...)` passes the same
`edge_vectors`, `edge_distances`, `padding_mask`, and `cutoff_factors` into each
`CartesianTransformer` layer:

```python
output_node_embeddings, output_edge_embeddings = gnn_layer(
    input_node_embeddings,
    input_edge_embeddings,
    inputs["element_indices_neighbors"],
    inputs["edge_vectors"],
    inputs["padding_mask"],
    inputs["edge_distances"],
    inputs["cutoff_factors"],
    use_manual_attention,
)
```

After each GNN layer, PET uses the reverse-neighbor index to feed reversed edge
messages into the next layer:

```python
new_input_edge_embeddings = output_edge_embeddings.reshape(...)[
    inputs["reverse_neighbor_index"]
].reshape(...)
```

For the `feedforward` featurizer used in the successful PET run, PET combines:

```text
previous edge message
+ current output edge embedding
+ MLP([output_edge_embedding, reversed_output_edge_embedding])
```

This is much richer than a final one-shot edge readout.

### 6. PET Has Target-Specific Node Heads And Edge Heads

For every target, PET creates both node and edge heads in `_add_output(...)`:

```python
self.node_heads[target_name] = ModuleList([... Linear(d_node, d_head) ...])
self.edge_heads[target_name] = ModuleList([... Linear(d_pet, d_head) ...])
```

It also creates target-specific final linear layers for both paths:

```python
self.node_last_layers[target_name] = ModuleList([... Linear(d_head, prod(shape)) ...])
self.edge_last_layers[target_name] = ModuleList([... Linear(d_head, prod(shape)) ...])
```

For `non_conservative_force`, `shape` is the Cartesian rank-1 target shape:

```text
[3, 1]
```

For `non_conservative_stress`, `shape` is the Cartesian rank-2 target shape:

```text
[3, 3, 1]
```

This means PET's edge branch does not predict just a scalar coefficient that is later
multiplied by a unit vector. The edge branch can directly predict target components
from edge features that already encode `[dx, dy, dz, distance]`.

### 7. PET Sums Edge Contributions With Cutoff Weighting

PET first applies the target-specific edge head:

```python
edge_last_layer_features = edge_head(edge_features_list[i])
```

Then the target-specific final edge layer predicts per-edge target-shaped values:

```python
edge_atomic_predictions = edge_last_layer_by_block(edge_last_layer_features)
```

PET masks padded edges and sums real edge contributions onto the center atom with the
smooth cutoff factors:

```python
edge_atomic_predictions = torch.where(
    ~expanded_padding_mask, 0.0, edge_atomic_predictions
)
edge_atomic_predictions = (
    edge_atomic_predictions * cutoff_factors[:, :, None]
).sum(dim=1)
```

The node branch makes a per-atom prediction too. PET adds both:

```python
atomic_prediction = node_atomic_prediction + edge_atomic_prediction
```

Finally:

- atom-level targets such as `non_conservative_force` remain per atom;
- system-level targets such as `non_conservative_stress` are summed over atoms;
- `non_conservative_stress` also goes through PET's stress post-processing path.

### 8. Why This Is Different From The StructureTransformer Diagnostic Head

The diagnostic StructureTransformer edge head currently does this:

```text
final atom features + edge radial basis
  -> scalar edge coefficient
  -> coefficient * unit edge vector
  -> scatter-add to center atom
```

PET does this instead:

```text
[dx, dy, dz, distance]
  -> edge token inside every CartesianTransformer layer
  -> attention with cutoff-factor masking
  -> reverse-edge message exchange between layers
  -> target-specific edge head
  -> target-shaped edge prediction
  -> cutoff-weighted neighbor sum
  -> add node contribution
```

So PET's advantage is not merely "it has edge vectors in the readout." PET uses edge
vectors throughout local message construction, attention, edge-feature evolution, and
target-specific edge readout. The StructureTransformer diagnostic head tested only
the last part, and in an especially simple form.


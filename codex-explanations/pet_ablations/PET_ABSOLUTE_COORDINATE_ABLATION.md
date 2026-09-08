# PET Absolute-Coordinate Token Ablation

## Motivation

Standard PET gives each local-attention sequence one central atom token and one
token for every directed neighbor-list edge. Geometry is encoded explicitly on
each edge as

```text
delta_r_ij^S = r_j + S @ cell - r_i
d_ij^S = norm(delta_r_ij^S)
```

and the geometric edge encoder receives `[delta_r_ij^S, d_ij^S]`. This gives PET
the relative displacement and its scalar norm directly.

The absolute-coordinate ablation asks whether PET still performs well when those
relative edge features are removed, while leaving its graph, local attention,
periodic images, reverse-edge message passing, and cutoff weighting unchanged.
The model instead receives enough information to reconstruct relative geometry:

```text
central token i:       atomic type Z_i + Cartesian position r_i
neighbor token j -> i: atomic type Z_j + periodic-image position r_j + S @ cell
```

This is an inductive-bias ablation, not an information-removal ablation. In
principle, the model can obtain the relative vector by subtracting the two input
positions and can then infer the distance. In practice, PET must learn these
operations rather than receiving them from preprocessing.

The neighbor token intentionally retains the type of atom `j`. Replacing it with
the type of the center `i` would also remove the neighbors' chemical identities
and would confound the geometric comparison.

## Configuration

PET accepts the following model option:

```yaml
architecture:
  model:
    geometry_mode: absolute
```

The values are:

| Mode | Central geometry | Neighbor geometry | Purpose |
| --- | --- | --- | --- |
| `relative` | None | Relative vector and distance | Standard PET and default |
| `absolute` | Base-cell Cartesian position | Cartesian position of the selected periodic image | This ablation |

`relative` remains the default so existing configurations and checkpoints retain
the standard PET behavior. Checkpoints predating the option are migrated to
`geometry_mode: relative`.

## Implementation

### Periodic-image coordinates

`systems_to_batch` already constructs every edge vector from the neighbor-list
cell shift. After periodic-image selection and adaptive-cutoff filtering, the new
neighbor coordinate is constructed as

```text
neighbor_image_position = center_position + edge_vector
                        = r_j + S @ cell
```

Constructing it after edge filtering guarantees that it has exactly the same edge
ordering as PET's retained neighbor list. It is then converted to PET's padded
node-edge-feature (NEF) layout. Padded coordinates are zero and remain excluded by
the existing padding mask.

It is important not to use the base-cell `r_j` alone: multiple periodic images of
the same atom have the same base-cell coordinate and would become indistinguishable.

### Token encoding

In `geometry_mode: relative`, `CartesianTransformer` is unchanged:

```text
edge_geometry = Linear(4, d_pet)([edge_vector, edge_distance])
node_token = node_species_or_message_embedding
```

In `geometry_mode: absolute`:

```text
edge_geometry = Linear(3, d_pet)(neighbor_image_position)
node_token = node_species_or_message_embedding
           + Linear(3, d_node)(node_position)
```

The existing neighbor-species embeddings and edge messages are merged with the
geometric embedding exactly as in standard PET. Central and neighbor coordinates
are injected in every PET GNN layer, matching the existing repeated injection of
edge geometry.

### What remains distance dependent

This implementation does **not** remove distance from the entire PET computation.
The following are deliberately held fixed:

- neighbor-list membership is determined by the distance cutoff;
- optional adaptive-cutoff selection uses edge distances;
- PET adds the logarithm of the smooth cutoff factor to attention logits;
- edge feature and prediction readouts are multiplied by cutoff factors.

Consequently, this experiment tests removal of explicit relative edge geometry
from the token features. It does not test a completely distance-blind model.

### Files changed

- `src/metatrain/pet/documentation.py`: declares and documents `geometry_mode`;
- `src/metatrain/pet/modules/structures.py`: constructs central and periodic-image
  position tensors;
- `src/metatrain/pet/modules/transformer.py`: implements the absolute token encoders;
- `src/metatrain/pet/model.py`: validates the mode and wires coordinates through both
  featurization paths;
- `src/metatrain/pet/modules/diagnostic.py`: exposes the coordinate inputs as
  `mtt::feature::node_positions` and
  `mtt::feature::neighbor_image_positions`;
- `src/metatrain/pet/checkpoints.py`: migrates older checkpoints to `relative`;
- `src/metatrain/pet/tests/test_absolute_coordinates.py`: checks periodic-image
  coordinates, forward/position gradients, defaults, and validation.

## Symmetry caveat

Raw absolute Cartesian coordinates are not translation invariant. Under a rigid
translation `t`, both `r_i` and `r_j + S @ cell` change even though the physical
structure and all relative distances are unchanged. The representation can also
be sensitive to the choice of cell origin and to an equivalent rewrapping of atoms
across periodic boundaries.

This is intentional for the ablation, but it affects interpretation. If the
absolute model performs poorly, the result supports the usefulness of PET's
relative geometric inductive bias; it does not prove that the scalar distance
alone is responsible. The degradation could come from having to learn subtraction,
the norm, translation invariance, or periodic gauge invariance.

PET's normal rotation/inversion augmentation is retained, but it does not add
random translations. Useful diagnostics are therefore:

- evaluate the same structures after several rigid translations;
- report the change in predicted energy under translation;
- report the norm of the summed predicted forces, which should vanish for a
  translation-invariant energy;
- evaluate equivalent periodic rewrappings when possible.

## Running the experiment

Use `options-pet-oam-l-modern-mptrj-salex-direct.yaml` and override only the
geometry mode. Keep `neighbor_cell_shift_mode: all` so this run is not mixed with
the periodic-image ablations.

### Absolute-coordinate PET

From the repository root:

```bash
PYTHONPATH=src /home/ryoji/miniconda3/envs/metatrain-pet/bin/python \
  -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-mptrj160k-absolute-coordinates.pt \
  -r training_set.indices=indices/mptrj_160k_seed0.txt \
  -r architecture.model.geometry_mode=absolute \
  -r architecture.model.neighbor_cell_shift_mode=all \
  -r wandb.name=pet-oam-l-mptrj160k-absolute-coordinates
```

Compare it with the existing standard PET run, or launch a matched baseline with:

```bash
PYTHONPATH=src /home/ryoji/miniconda3/envs/metatrain-pet/bin/python \
  -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-mptrj160k-relative-coordinates.pt \
  -r training_set.indices=indices/mptrj_160k_seed0.txt \
  -r architecture.model.geometry_mode=relative \
  -r architecture.model.neighbor_cell_shift_mode=all \
  -r wandb.name=pet-oam-l-mptrj160k-relative-coordinates
```

Use the same data indices, random seed, validation set, loss weights, scaling,
cutoff configuration, and stopping rule for both runs.

For a one-epoch pipeline check, append:

```bash
-r architecture.training.num_epochs=1 \
-r wandb.mode=offline
```

Before a full run, also overfit both modes on the same small subset. This checks
that the absolute model has a working position/force gradient path and separates
a basic optimization failure from a generalization difference.

## Validation performed during implementation

The focused checks cover:

1. `geometry_mode` defaults to `relative` and rejects unknown values.
2. For every retained periodic edge, the batched absolute coordinate equals
   `r_j + S @ cell`.
3. Absolute mode completes a PET energy forward pass and supports the position
   double backward required for conservative-force training.
4. Absolute mode can be compiled and evaluated with TorchScript.
5. A version-14 PET checkpoint upgrades to version 15 as `geometry_mode: relative`
   and loads with its original four-component relative edge encoder.

The repository PET environment did not contain `pytest`, so the focused test module
was also exercised through equivalent direct Python probes.

## Interpreting the result

If absolute-coordinate PET matches standard PET, explicit relative edge geometry
is not essential when the local graph, all periodic images, and cutoff weighting
are retained.

If it performs substantially worse, PET's explicit relative representation is an
important inductive bias. Translation and periodic-rewrapping diagnostics should
then be examined before attributing the gap specifically to the explicit scalar
distance.

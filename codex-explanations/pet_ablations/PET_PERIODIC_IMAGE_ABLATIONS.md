# PET Periodic-Image Ablations

## Motivation

The Structure Transformer graph-attention experiment represents every atom only
once, even when the neighbor list contains several periodic images of that atom.
Its dense atom-pair attention bias collapses all edges `(i, j, eta)` onto one
entry `A_ij`. PET instead exposes every periodic edge as a separate neighbor
token containing the neighbor species, Cartesian displacement, distance, and
cutoff factor.

This ablation reverses the original proposal: rather than adding PET-style image
tokens to Structure Transformer, it removes some or all periodic images from PET.
That is a smaller and more controlled way to test whether PET's explicit periodic
edge instances are an important inductive bias.

The ablation does not make PET equivalent to Structure Transformer. PET still has
local environment attention, explicit edge vectors, reverse-edge message passing,
and its existing readout. It isolates the periodic-image representation within
PET while holding the rest of PET fixed.

## Configuration

PET now accepts:

```yaml
architecture:
  model:
    neighbor_cell_shift_mode: all
```

The supported modes are:

| Mode | Retained neighbor-list edges | Question tested |
| --- | --- | --- |
| `all` | Every periodic edge | Standard PET baseline |
| `nearest` | The shortest reverse-closed periodic interaction for each unordered base-atom pair | Does the multiplicity of periodic images matter? |
| `zero` | Only edges with `eta = (0, 0, 0)` | Do cross-boundary and repeated-cell neighbors matter? |

For distinct atoms, `nearest` retains one directed edge in each direction. For a
periodic self-image, PET needs both reverse directions for message passing, so
the nearest `+eta` and `-eta` pair is retained.

The default is `all`, preserving existing PET behavior and old checkpoints.

## Implementation

The selection happens in
`src/metatrain/pet/modules/structures.py` after the full periodic edge vectors and
distances have been constructed, but before adaptive cutoffs are calculated.
Therefore, neither discarded edge geometry nor discarded neighbor counts enter
the adaptive-cutoff calculation.

The processing order is:

1. Request PET's usual full periodic neighbor list up to `cutoff`.
2. Construct `r_ij^eta = x_j - x_i + eta L` and its distance.
3. Apply `neighbor_cell_shift_mode`.
4. Compute adaptive per-atom cutoffs from the retained edges.
5. Apply cutoff pruning and cutoff factors.
6. Convert the retained, reverse-closed edges to PET's padded NEF representation.

### `all`

No filtering is performed.

### `zero`

An edge is retained exactly when all three integer cell-shift components are zero.
The neighbor-list request is unchanged; filtering happens inside PET so the same
data and collate pipeline can be used for every mode.

### `nearest`

PET exchanges edge messages with the reverse edge between GNN layers. Selecting
the nearest directed edge independently could remove its reverse when distances
are tied. The implementation consequently:

1. pairs every edge with its reverse;
2. chooses one representative per reverse pair;
3. groups representatives by unordered base-atom pair;
4. chooses the shortest representative in each group;
5. restores the selected representative's reverse edge.

Original neighbor-list order is used as the deterministic tie-break for exactly
equal distances.

## Interpretation and limitations

The three-way comparison separates two effects:

- `all` versus `nearest` measures the value of multiple periodic images for
  the same base-cell atom pair;
- `nearest` versus `zero` measures the value of retaining the closest
  cross-boundary image;
- `all` versus `zero` measures the combined effect.

The `zero` model is intentionally not a valid periodic potential. Its result
depends on which equivalent periodic image is stored in the input unit cell. A
physically close pair on opposite sides of a boundary can have no retained edge,
and remapping one atom by a lattice vector can change the prediction.

Adaptive neighbor counts also need careful interpretation. The reference config
targets approximately 40 neighbors. A zero-shift environment in a cell with fewer
than 41 atoms cannot reach that target. The adaptive solver/grid still returns a
cutoff, but the actual retained neighbor count is lower. This loss of coordination
is part of the ablation, but it should be measured rather than silently assumed.

For each dataset split, record at least:

- the fraction of original edges with nonzero cell shift;
- mean and quantiles of retained neighbors per atom;
- the fraction of atoms with zero retained neighbors;
- metrics stratified by unit-cell atom count and, if possible, cell volume.

PET already exposes retained neighbor counts as property 1 of
`mtt::aux::cutoff_stats`.

## Running the experiments

The reference configuration is
`options-pet-oam-l-modern-mptrj-salex-direct.yaml`. It explicitly sets `all`,
and the other runs should use command-line overrides so every other setting stays
identical.

From the repository root, using the project environment:

### Standard PET baseline

```bash
PYTHONPATH=src /home/ryoji/miniconda3/envs/metatrain-pet/bin/python \
  -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-mptrj160k-periodic-all.pt \
  -r training_set.indices=indices/mptrj_160k_seed0.txt \
  -r architecture.model.neighbor_cell_shift_mode=all \
  -r wandb.name=pet-oam-l-mptrj160k-periodic-all
```

### Nearest-image PET

```bash
PYTHONPATH=src /home/ryoji/miniconda3/envs/metatrain-pet/bin/python \
  -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-mptrj160k-periodic-nearest.pt \
  -r training_set.indices=indices/mptrj_160k_seed0.txt \
  -r architecture.model.neighbor_cell_shift_mode=nearest \
  -r wandb.name=pet-oam-l-mptrj160k-periodic-nearest
```

### Unit-cell-only PET

```bash
PYTHONPATH=src /home/ryoji/miniconda3/envs/metatrain-pet/bin/python \
  -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-mptrj160k-periodic-zero.pt \
  -r training_set.indices=indices/mptrj_160k_seed0.txt \
  -r architecture.model.neighbor_cell_shift_mode=zero \
  -r wandb.name=pet-oam-l-mptrj160k-periodic-zero
```

Use the same seed, 160k index file, validation set, target scaling, loss weights,
and stopping rule for all three runs.

Before launching the full comparison, a one-epoch pipeline check can be run by
adding:

```bash
-r architecture.training.num_epochs=1 \
-r wandb.mode=offline
```

This verifies neighbor construction, training, checkpointing, and evaluation.
A small-data overfit is also recommended before the full run; use a dedicated
small indices file rather than changing the data independently for each mode.

## Expected conclusions

A large `all` to `nearest` degradation would show that repeated images of the
same base atom carry important information. A large `nearest` to `zero`
degradation would show that correct cross-boundary connectivity is important even
when only one image per pair is retained.

If both reduced modes remain competitive with `all`, the missing-image
hypothesis is unlikely to explain Structure Transformer's failure by itself. The
remaining differences to investigate would include explicit Cartesian geometry,
local versus global attention, reverse-edge message passing, and the force/stress
readouts.

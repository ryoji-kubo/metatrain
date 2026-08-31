# PET Tutorial Training Flow

This note documents what happens during the beginner PET tutorial training run:

```bash
cd examples/0-beginner
python -m metatrain train options-scratch.yaml
```

It focuses on the PET architecture, the forward operations, and optimizations
that are worth knowing before moving on to larger datasets.

## Tutorial Configuration

The tutorial config is `examples/0-beginner/options-scratch.yaml`.

The important PET settings are:

```yaml
architecture:
  name: pet
  model:
    cutoff: 4.5
  training:
    num_epochs: 10
    batch_size: 10
    log_interval: 1
    checkpoint_interval: 10
```

The dataset section is:

```yaml
training_set:
  systems:
    read_from: ./ethanol_reduced_100.xyz
    length_unit: Angstrom
  targets:
    energy:
      key: energy
      unit: eV
```

The ethanol file is an extended XYZ with:

```text
Properties=species:S:1:pos:R:3:forces:R:3 energy=...
```

So the tutorial learns:

- a system-level scalar energy target
- conservative forces, represented internally as gradients of that energy target

The config expansion in `src/metatrain/utils/omegaconf.py` automatically enables
`forces` lookup for energy targets. The ASE reader then finds the `forces` array
in the XYZ file.

## PET Architecture In This Repo

The PET implementation lives mainly in:

```text
src/metatrain/pet/model.py
src/metatrain/pet/trainer.py
src/metatrain/pet/modules/transformer.py
src/metatrain/pet/modules/structures.py
```

The architecture entry point is `src/metatrain/pet/__init__.py`:

```python
__model__ = PET
__trainer__ = Trainer
```

When `architecture.name: pet` is read from YAML, the generic metatrain CLI
imports this package and uses these two classes.

At a high level, PET is an attention-based atomistic graph neural network. It
builds a graph from neighbor lists, embeds atomic species and geometric edge
features, applies Cartesian transformer/message-passing layers, and then uses
target-specific heads to produce atomic contributions. For energy targets,
atomic contributions are summed over atoms to get the total structure energy.

The tutorial uses the default PET model width unless explicitly overridden:

- `d_pet: 128`
- `d_node: 256`
- `d_head: 128`
- `num_gnn_layers: 2`
- `num_attention_layers: 2`
- `num_heads: 8`
- `featurizer_type: feedforward`
- `cutoff_function: Bump`
- `activation: SwiGLU`
- `normalization: RMSNorm`

These defaults are documented in `src/metatrain/pet/documentation.py`.

## Model Setup

During `train_model(...)` in `src/metatrain/cli/train.py`, metatrain:

1. Loads and validates `options-scratch.yaml`.
2. Merges the user config with PET defaults.
3. Reads the dataset and target metadata.
4. Infers atomic types from the train and validation datasets.
5. Builds a `DatasetInfo` object.
6. Creates:

```python
model = PET(hypers["model"], dataset_info)
trainer = Trainer(hypers["training"])
```

Inside `PET.__init__`, the model creates:

- a requested neighbor list with cutoff `4.5`
- species-to-index mapping for the atomic types in the dataset
- PET transformer layers
- node and edge embedding modules
- output-specific node and edge heads
- final linear layers for each target
- a composition model for atomic energy baselines
- a target scaler
- optional ZBL and long-range modules, if enabled

For the tutorial, ZBL and long-range features are off unless the YAML is changed.

## Data Preparation Before Forward

Training batches are prepared in `src/metatrain/pet/trainer.py`.

Before the epoch loop, the trainer:

1. Moves the model to the selected device and dtype.
2. Trains the composition model, which learns per-species baseline
   contributions for compatible targets such as energy.
3. Trains the scaler, which normalizes targets for easier optimization.
4. Builds data loaders.
5. Builds collate functions that compute neighbor lists and remove baseline and
   scaling contributions from the training targets.

The important collate-time transforms are:

- `get_system_with_neighbor_lists_transform(...)`
- `get_remove_additive_transform(...)`
- `get_remove_scale_transform(...)`

This means the neural network is trained on a normalized residual target:

```text
reference target
  - composition baseline
  - optional additive terms
  - scale normalization
```

At evaluation time, PET adds these pieces back.

## One Training Batch

For each batch in `Trainer.train(...)`, the loop does roughly:

```python
systems, targets, extra_data = unpack_batch(batch)
systems, targets, extra_data = batch_to(..., dtype=dtype, device=device)

predictions = evaluate_model(
    model,
    systems,
    {key: train_targets[key] for key in targets.keys()},
    is_training=True,
)

predictions = average_by_num_atoms(...)
targets = average_by_num_atoms(...)
predictions = model.scaler(..., use_per_property_scales=True)

loss = loss_fn(predictions, targets, extra_data)
loss.backward()
clip_grad_norm_(...)
optimizer.step()
scheduler.step()
```

The default optimizer is Adam, unless `weight_decay` is set, in which case AdamW
is used. Gradients are clipped with `grad_clip_norm`, which defaults to `1.0`.

## How Forces Are Trained

The tutorial does not train a separate direct force head. Instead:

1. PET predicts energy.
2. `src/metatrain/utils/evaluate_model.py` checks that the energy target has
   position gradients.
3. It prepares systems with `positions.requires_grad_(True)`.
4. It runs the PET forward pass for energy.
5. It calls `torch.autograd.grad(...)` to compute:

```text
d energy / d positions
```

6. The ASE reader stored reference forces as negative position gradients, so the
   loss compares gradients in a consistent sign convention.

This is why force training needs higher-order derivatives. During training,
the loss depends on gradients of the model output with respect to positions, so
backpropagation through the force loss requires differentiating through those
gradients.

## PET Forward Pass

The main forward method is `PET.forward(...)` in `src/metatrain/pet/model.py`.

The stages below are the ones used by the tutorial.

### Stage 0: Systems To Batch

`systems_to_batch(...)` in `src/metatrain/pet/modules/structures.py` converts a
list of `System` objects into tensors PET can process.

It constructs:

- center atom indices
- neighbor atom indices
- atomic species
- periodic cell shifts
- edge vectors
- edge distances
- cutoff factors
- padding mask
- reverse-edge index
- sample labels for metatensor outputs

The tutorial uses fixed cutoff mode with `cutoff: 4.5`. If adaptive cutoff is
enabled through `num_neighbors_adaptive`, the same function can compute per-atom
cutoffs before masking edges.

### Stage 1: NEF Batching

PET converts edge arrays into NEF format: node-edge-feature.

Instead of storing a ragged list of neighbors for each atom, it creates tensors
with shape like:

```text
n_atoms x max_neighbors x feature_dim
```

and a padding mask that distinguishes real neighbors from padded slots.

This shape is convenient for attention because each atom gets a local token
sequence consisting of:

```text
central atom token + neighbor edge tokens
```

### Stage 2: Species And Edge Embeddings

PET embeds:

- central atomic species into node features
- neighbor atomic species into edge/message features
- geometric edge inputs, including edge vectors and edge distances

The edge geometry is processed in `CartesianTransformer.forward(...)` in
`src/metatrain/pet/modules/transformer.py`.

### Stage 3: Transformer Message Passing

PET applies `num_gnn_layers` Cartesian transformer layers. In the tutorial
defaults, this means two GNN layers.

Within each layer:

1. A central atom token is combined with neighbor edge tokens.
2. Multi-head attention updates the token sequence.
3. Feed-forward blocks update features.
4. Cutoff factors are included in the attention mask.

The default transformer type is `PreLN`, so layer normalization is applied
before attention/feed-forward blocks.

### Stage 4: Reversed-Edge Message Flow

PET tracks corresponding reverse edges, so the message for `i -> j` can be
reused as the reversed message for `j -> i`.

In feedforward featurization, the model:

1. runs one GNN layer
2. reshapes edge embeddings into a flat edge array
3. reorders them using `reverse_neighbor_index`
4. combines forward and reversed messages with an MLP
5. feeds that result into the next GNN layer

This gives PET bidirectional edge/message flow while still keeping a dense
NEF tensor layout for attention.

### Stage 5: Output-Specific Heads

After the shared PET feature extractor, each target gets its own heads:

- node heads
- edge heads
- node final linear layers
- edge final linear layers

These are created in `PET._add_output(...)`.

For the tutorial, the main output is `energy`. PET first predicts atomic
contributions, then sums over atoms because the requested energy output is a
system-level quantity.

### Stage 6: Energy Output

For each atom, PET combines:

```text
node contribution + cutoff-weighted sum of edge contributions
```

Then, for system-level outputs such as energy, metatrain sums atomic
contributions over atoms.

The result is a metatensor `TensorMap` with an `energy` block.

### Stage 7: Gradient Outputs For Forces

PET itself returns energy from `forward`. Then `evaluate_model(...)` computes
position gradients of that energy and attaches them as a gradient block.

That gradient block is what the loss function sees as the model's force-related
prediction.

## Loss Construction

Loss setup happens in two places:

- `expand_loss_config(...)` in `src/metatrain/utils/omegaconf.py`
- `LossAggregator` in `src/metatrain/utils/loss.py`

For the tutorial, no custom loss is specified, so metatrain builds default MSE
losses for:

- `energy`
- `energy` position gradients, because forces are present

The `LossAggregator` creates one loss term for the target values and one loss
term for each gradient listed in the target metadata.

## Metrics And Logging

During training, PET logs:

- training loss
- validation loss
- energy RMSE/MAE
- forces RMSE/MAE
- learning rate

Metrics are written to:

```text
outputs/.../train.log
outputs/.../train.csv
```

Validation metrics are also used to track the best model. The default
selection metric is `mae_prod`, defined in PET trainer hyperparameters.

## Checkpoints And Export

The trainer saves intermediate checkpoints every `checkpoint_interval` epochs.
For the tutorial, `checkpoint_interval: 10`.

The output directory contains:

```text
model_0.ckpt
model.ckpt
model.pt
```

The distinction matters:

- `.ckpt` files include training state and can be used for restart/fine-tuning.
- `.pt` is an exported metatomic model for evaluation and simulation.

At the end of training, `src/metatrain/cli/train.py` reloads the final
checkpoint, exports the model, and saves the `.pt` model.

## Optimizations Worth Highlighting

### Editable CLI Path

`mtt` and `python -m metatrain` hit the same code path. For debugging, using
`python -m metatrain train options-scratch.yaml` avoids treating `mtt` as a
black box.

### NEF Tensor Layout

PET converts ragged neighbor lists into padded `node x neighbor x feature`
tensors. This makes local attention easier to express with batched tensor
operations.

### Cutoff Factors Inside Attention

Cutoff factors are not only used at final edge summation. They are also folded
into the attention mask. This lets distance/cutoff information modulate message
passing throughout the transformer.

### Bump Cutoff Default

The PET default cutoff function is `Bump`, with `cutoff_width: 0.5`. This gives
a smooth taper near the cutoff instead of a hard discontinuity.

### Manual Attention During Force Training

When positions require gradients and the model is in training mode, PET sets:

```python
use_manual_attention = edge_vectors.requires_grad and self.training
```

The code then uses `manual_attention(...)` instead of
`torch.nn.functional.scaled_dot_product_attention`.

The reason is practical: force training requires double backward through the
attention computation, and the built-in scaled-dot-product attention path does
not support that requirement in this code.

At inference/evaluation without training-mode force loss, PET can use the
built-in attention path.

### Unique Padded Reverse Indices

In `systems_to_batch(...)`, padded reverse-edge indices are replaced by unique
indices. The code comments note that too many repeated padded indices can make
backward very slow in PyTorch.

This is a small but important low-level optimization for message passing with
reverse-edge indexing.

### Single-Cell Fast Path

When converting systems to edge vectors, `systems_to_batch(...)` has a special
path for `len(cells) == 1` to avoid a slower `einsum` path. This helps small
or single-structure evaluation cases.

### Composition Baseline

PET trains a composition model before the neural network loop. For energy-like
targets, this subtracts fitted per-species baseline contributions so the neural
network learns the residual.

This often makes energy training easier because the network does not have to
rediscover large atomic baseline terms from scratch.

### Target Scaling

PET trains a scaler before the neural network loop. During training, targets are
scaled; during evaluation, predictions are unscaled. This improves optimizer
conditioning while preserving physical units in reported predictions.

### Float64 For Additive Models And Scaler

The trainer explicitly keeps additive models and scaler quantities in float64,
even when the neural network itself uses float32. The code comments point out
that composition weights can be large, so this avoids numerical issues.

### Automatic DataLoader Workers

If `num_workers` is not specified, PET chooses a worker count automatically via
shared data-loading utilities. This is convenient for small tutorial runs and
usually worth overriding only when profiling larger jobs.

### Optional Atom-Count Batch Packing

PET supports `max_atoms_per_batch` for `MemmapDataset`. This is not used in the
ethanol tutorial, but it matters for larger materials datasets with variable
atom counts. Instead of fixed `batch_size`, batches are packed up to an atom
budget.

This is likely useful later for MPTrj-scale training.

### Optional Adaptive Cutoff

PET can use `num_neighbors_adaptive` to choose per-atom cutoffs targeting a
rough number of neighbors. The default tutorial does not enable this.

If enabled, the default method is `solver`, which uses Newton-bisection root
finding with a cheaper implicit-function-theorem backward step. The older
`grid` method is kept for checkpoint compatibility.

### Optional Distributed Training

PET has trainer support for distributed training through the `distributed`
hyperparameter. This is not part of the beginner tutorial, but the code path is
present in `src/metatrain/pet/trainer.py`.

## What Is Active In The Tutorial

Active by default in `options-scratch.yaml` plus PET defaults:

- fixed cutoff neighbor graph
- Bump cutoff function
- feedforward PET featurization
- Cartesian transformer layers
- energy output head
- force loss through energy gradients
- composition baseline
- target scaling
- Adam optimizer
- learning-rate scheduler
- gradient clipping
- checkpointing
- validation-based best model tracking

Not active unless the YAML is changed:

- adaptive cutoff
- ZBL repulsion
- long-range electrostatics
- distributed training
- atom-count batch packing
- fine-tuning
- direct non-conservative force or stress targets

## Suggested Debugging Path

To understand the tutorial training flow, set breakpoints in this order:

```text
src/metatrain/cli/train.py
  train_model(...)

src/metatrain/pet/trainer.py
  Trainer.train(...)

src/metatrain/utils/evaluate_model.py
  evaluate_model(...)
  compute_gradient(...)

src/metatrain/pet/model.py
  PET.forward(...)
  PET._calculate_features(...)
  PET._calculate_atomic_predictions(...)

src/metatrain/pet/modules/structures.py
  systems_to_batch(...)

src/metatrain/pet/modules/transformer.py
  CartesianTransformer.forward(...)
  AttentionBlock.forward(...)
```

The most revealing first inspection is usually one batch inside
`Trainer.train(...)`: inspect `systems`, `targets`, `predictions`, and the
gradient block attached to the `energy` prediction after `evaluate_model(...)`
returns.


# PET MPtrj/sAlex Direct Training Pipeline

This note explains the training path for:

```text
options-pet-oam-l-modern-mptrj-salex-direct.yaml
```

That config trains PET on prepared MPtrj memmap data in `data/` and validates on the
prepared 30k sAlex memmap split in `data_salex_val_30k/`.

The config is direct-only:

- `energy` is a scalar system target from key `e`.
- `non_conservative_force` is an atom-level Cartesian vector target from key `f`.
- `non_conservative_stress` is a system-level Cartesian rank-2 target from key `s`.
- `energy.forces: false` and `energy.stress: false`, so forces/stress are not produced
  by energy gradients in this run.

The nearby config `options-pet-oam-l-modern-mptrj-salex.yaml` is different: it asks for
energy gradients for forces and stress, while also keeping direct
`non_conservative_force` and `non_conservative_stress` heads.

## Debugger Setup

The existing `.vscode/launch.json` has a smoke debugger for
`options-pet-oam-l-modern-mptrj-salex.yaml`. For this direct-only config, use the same
shape but swap the options file and disable W&B for debugging:

```json
{
  "name": "Debug PET-OAM-L direct MPtrj-sAlex smoke",
  "type": "debugpy",
  "request": "launch",
  "python": "/home/ryoji/miniconda3/envs/metatrain-pet/bin/python",
  "module": "metatrain",
  "args": [
    "--debug",
    "train",
    "options-pet-oam-l-modern-mptrj-salex-direct.yaml",
    "-o",
    "pet-oam-l-modern-mptrj-salex-direct-debug.pt",
    "-r",
    "device=cpu",
    "-r",
    "wandb.mode=disabled",
    "-r",
    "training_set.indices=[0,1,2,3,4,5,6,7]",
    "-r",
    "validation_set.indices=[0,1,2,3]",
    "-r",
    "architecture.training.num_epochs=1",
    "-r",
    "architecture.training.max_atoms_per_batch=128",
    "-r",
    "architecture.training.num_workers=0"
  ],
  "cwd": "/home/ryoji/equivarient/metatrain",
  "console": "integratedTerminal",
  "justMyCode": false
}
```

Use `device=cpu` for a debugger-friendly smoke run. Remove that override when you want
to debug CUDA behavior.

Good breakpoints:

- `src/metatrain/__main__.py`, `main()`
- `src/metatrain/cli/train.py`, `train_model(...)`
- `src/metatrain/utils/data/dataset.py`, `MemmapDataset.__getitem__`
- `src/metatrain/pet/trainer.py`, `Trainer.train(...)`
- `src/metatrain/utils/scaler/scaler.py`, `Scaler.train_model(...)`
- `src/metatrain/pet/model.py`, `PET.forward(...)`
- `src/metatrain/utils/evaluate_model.py`, `evaluate_model(...)`

## High-Level Flow

The launch command enters `python -m metatrain`, which calls:

```text
src/metatrain/__main__.py::main
src/metatrain/cli/train.py::train_model
src/metatrain/pet/trainer.py::Trainer.train
src/metatrain/pet/model.py::PET.forward
```

The main phases are:

1. Parse and merge options.
2. Load training/validation datasets.
3. Infer target metadata and atomic types.
4. Build the PET model.
5. Fit PET's energy composition baseline.
6. Fit PET's target scalers.
7. Build collate transforms and atom-count batch samplers.
8. Run the training loop.
9. Validate, log metrics, checkpoint, and export.

## Dataset Loading

The direct config points to:

```yaml
training_set:
  systems:
    read_from: data
  targets:
    energy:
      key: e
      forces: false
      stress: false
    non_conservative_force:
      key: f
      sample_kind: atom
    non_conservative_stress:
      key: s
      sample_kind: system

validation_set:
  systems:
    read_from: data_salex_val_30k
```

`train_model(...)` expands these dataset configs, then calls `get_dataset(...)`. For
these folders, metatrain uses `MemmapDataset`.

`MemmapDataset` expects:

```text
ns.npy    number of structures
na.npy    cumulative atom counts
x.bin     positions
a.bin     atomic numbers/types
c.bin     cells
e.bin     energies
f.bin     forces
s.bin     stresses
```

In `MemmapDataset.__getitem__`, each sample is converted into:

- a `metatomic.torch.System` with positions, atomic types, cell, and periodic flags
- one `TensorMap` per target

For this direct-only config, `energy` has no attached gradients. If you use the mixed
config, the same dataset class attaches force targets as the `positions` gradient of
energy and stress targets as the `strain` gradient of energy.

## CLI Setup

`src/metatrain/cli/train.py::train_model` does the non-PET-specific setup:

- validates base options
- imports the `pet` architecture dynamically
- merges PET defaults with the YAML
- picks the device and dtype
- seeds Torch, NumPy, and Python
- expands train/validation/test datasets
- applies `training_set.indices` and `validation_set.indices` overrides
- infers atomic types from train plus validation
- creates `DatasetInfo`
- saves `options_restart.yaml`
- creates `PET(...)` and `Trainer(...)`

The `test_set.indices: []` in this config means no random test split is drawn.

## PET Model Construction

`src/metatrain/pet/model.py::PET.__init__` builds the model from the architecture
hypers:

- cutoff `10.0`
- adaptive neighbor target `40`
- `Bump` cutoff with width `0.5`
- `d_pet = 512`, `d_head = 512`, `d_node = 2048`
- `3` GNN layers
- `2` attention layers per PET block
- RMSNorm, SwiGLU, PreLN

The model requests one full neighbor list through `requested_neighbor_lists()`.
The neighbor list cutoff and adaptive cutoff are part of the PET input pipeline.

The constructor also creates:

- output-specific node and edge heads for `energy`, `non_conservative_force`, and
  `non_conservative_stress`
- an energy composition model in `model.additive_models`
- a `Scaler` in `model.scaler`

The composition model only supports scalar or spherical targets. For this direct
config, the important composition baseline is therefore the scalar `energy` baseline,
not the Cartesian force/stress targets.

## Pre-Training Fitting

Before the first optimizer step, `Trainer.train(...)` does two fitted preprocessing
passes.

First, it fits composition weights:

```text
model.additive_models[0].train_model(...)
```

This learns per-species linear baseline contributions, mainly for energy in this run.
For the exact fitted linear model and where it enters training/evaluation, see
[Appendix: CompositionModel Deep Dive](#appendix-compositionmodel-deep-dive).

Second, it fits target scales:

```text
model.scaler.train_model(...)
```

Scaler fitting removes additive contributions first, then averages system targets by
atom count unless they are listed in `per_structure_targets`.
For the detailed scaler mechanics, see
[Appendix: Scaler Deep Dive](#appendix-scaler-deep-dive).

In this config:

```yaml
per_structure_targets:
  - non_conservative_stress
```

So:

- energy is treated per atom for loss/metrics
- force is already atom-level and is not divided by atom count
- stress stays per structure and is not divided by atom count

This fitting step matters. PET is not merely using a hard-coded normalizer; it fits
the residual target scale on the current training set or subset.

## Data Preprocessing Intuition

The important high-level idea is that PET is not trained directly on the raw energy,
force, and stress numbers. Before the loss sees them, the targets are turned into
residual, normalized quantities.

For energy, PET first removes a simple composition baseline. The fitted composition
model learns one energy contribution per atom type, then estimates each structure's
composition energy as:

```text
composition_energy =
  count_H  * weight_H
+ count_O  * weight_O
+ count_Si * weight_Si
+ ...
```

The energy target that PET learns is therefore not raw total energy. It is the part
left over after subtracting this stoichiometric baseline:

```text
energy_residual = raw_total_energy - composition_energy
```

Because total energy grows with system size, the residual energy is then treated per
atom for training and logged metrics:

```text
energy_residual_per_atom = energy_residual / number_of_atoms
```

For forces, there is no composition baseline in this direct config. The force target
is already atom-level, so it is not divided by the number of atoms:

```text
force_preprocessed = raw_direct_force
```

For stress, there is also no composition baseline. The direct stress target is listed
as a per-structure target, so it is not divided by the number of atoms:

```text
stress_preprocessed = raw_direct_stress
```

After this, each target is normalized by a fitted RMS-like scale from the training set:

```text
s_energy = RMS(energy_residual_per_atom)
s_force  = RMS(force components)
s_stress = RMS(stress components)
```

For per-atom targets such as force, the implementation can keep separate force scales
per atom type. The high-level meaning is still the same: force values are divided by a
typical force magnitude learned from the training set.

The training targets are then:

```text
energy_for_loss = energy_residual_per_atom / s_energy
force_for_loss  = raw_direct_force / s_force
stress_for_loss = raw_direct_stress / s_stress
```

So PET learns normalized residual targets. The scale fitting makes energy, force, and
stress numerically easier to optimize together, while the composition baseline removes
a simple extensive energy trend that the neural network should not have to relearn
from scratch.

There is a post-processing step before logged metrics are computed. The trainer
multiplies predictions and targets by the fitted scales again, so MAE/RMSE are reported
in physical units:

```text
energy error -> meV/atom
force error  -> meV/A
stress error -> meV/A^3
```

For energy metrics during training/validation, the composition baseline is not added
back in the trainer metric path. This does not change MAE/RMSE, because adding the
same composition baseline to both prediction and target cancels in the error:

```text
(predicted_residual + composition_energy)
  - (target_residual + composition_energy)
= predicted_residual - target_residual
```

For exported or explicit eval-mode predictions, PET does restore both pieces: it
multiplies by the fitted scales and adds the composition baseline back to produce full
physical predictions.

## Collate-Time Transforms

The training dataloader uses this ordered collate transform stack:

1. `atomic_basis_transform`
2. random rotational/inversion augmentation
3. neighbor-list construction
4. remove additive baseline contributions
5. remove target scales

Neighbor-list construction is detailed in
[Appendix: Neighbor-List Deep Dive](#appendix-neighbor-list-deep-dive).

The validation dataloader uses the same path except it does not apply random
rotational augmentation.

For this direct config, the random rotation/inversion transform rotates:

- atomic positions/cell through the `System`
- Cartesian vector force targets
- Cartesian rank-2 stress targets

That is a major PET training-pipeline difference from the Transformer run we compared.

## Batching

The config uses:

```yaml
batch_size: 16
max_atoms_per_batch: 512
```

When `max_atoms_per_batch` is set, PET uses `MaxAtomDistributedBatchSampler` rather
than a normal fixed-structure-count batch.

The sampler:

- reads atom counts quickly from `MemmapDataset.get_all_atom_counts()`
- greedily packs structures so each batch has at most `512` atoms
- packs batches once with seed `0`
- reshuffles batch order each epoch
- splits batches across distributed ranks when distributed training is enabled

`batch_size` is still used for composition/scaler fitting. The actual neural-network
training batches are atom-count packed.

## One Training Step

Inside the training loop:

1. Get a batch from the dataloader.
2. Move systems and targets to the selected device/dtype.
3. Call `evaluate_model(...)`.
4. `evaluate_model(...)` asks PET for the requested direct outputs.
5. Because this config has no energy gradients, no force/stress autograd gradients are
   computed from energy.
6. `average_by_num_atoms(...)` converts system energy to energy per atom and leaves
   direct force/stress as intended.
7. PET applies per-property prediction scaling before the loss.
8. `LossAggregator` computes Huber losses.
9. Backpropagation runs.
10. Gradients are clipped to `1.0`.
11. Adam updates parameters.
12. The warmup/cosine scheduler steps.

The configured losses are:

```yaml
energy:
  type: huber
  weight: 1.0
  delta: 0.015
non_conservative_force:
  type: huber
  weight: 1.0
  delta: 0.04
non_conservative_stress:
  type: huber
  weight: 0.01
  delta: 0.004
```

The optimizer is Adam unless `weight_decay` is set, in which case the trainer uses
AdamW. This config has `weight_decay: null`, so it uses Adam.

## PET Forward Pass

`PET.forward(...)` has four main stages.

Stage 0: systems to PET batch

- Converts `System` objects plus neighbor lists into padded neighbor tensors.
- Builds central atom species, neighbor species, edge vectors, distances, cutoff
  weights, reverse-edge indices, and system/sample labels.
- Applies the fixed/adaptive cutoff logic.

Stage 1: feature computation

- Embeds atomic species.
- Runs Cartesian transformer/GNN layers.
- Uses edge vectors and neighbor information.
- Uses reverse neighbor indices for bidirectional message passing.
- Optionally adds long-range features if configured. In this config, long range is
  disabled.

Stage 2: optional intermediate features

- Only used if feature outputs are requested.

Stage 3: output-specific last-layer features

- Builds node and edge features specialized for each requested output.

Stage 4: atomic predictions

- Applies output-specific final linear layers.
- Sums node and edge contributions.
- Produces TensorMaps for `energy`, `non_conservative_force`, and
  `non_conservative_stress`.
- Symmetrizes and volume-normalizes rank-2 Cartesian tensor outputs such as stress.

During training, PET returns scaled-residual-space predictions. It does not add the
composition baseline or full target scales inside `PET.forward(...)`. Those are handled
by the trainer/collate path.

During evaluation/exported inference, `PET.forward(...)` applies the scaler and adds
the additive composition contribution back to the outputs.

## Validation And Metrics

Validation uses the same model/evaluate path but:

- no random rotational augmentation
- no optimizer step
- no gradient clipping
- no scheduler step

Validation predictions and targets are averaged by atom count using the same
`per_structure_targets` policy. Metrics are accumulated after restoring the learned
target scales, so logged metric errors are in physical units:

- energy: meV/atom
- force: meV/A
- stress: meV/A^3

The trainer metric path restores scales but does not add the energy composition
baseline back. This is fine for MAE/RMSE because the same baseline would be added to
both prediction and target and therefore cancels in the error. For the exact code path
and terminology, see
[Appendix: Validation Metrics, Scaling, And Baselines](#appendix-validation-metrics-scaling-and-baselines).

For the run at `outputs/2026-06-30/10-17-34`, the final validation row was:

```text
energy MAE: 53.29 meV/atom
force MAE:  69.81 meV/A
stress MAE:  3.1197 meV/A^3
```

## What PET Does That The Transformer Run Did Not

This section refers to the Transformer run:

```text
/home/ryoji/equivarient/equiformer_v3/logs/omat24/transformer/logs/wandb/2026-06-06-07-30-08-transformer_mptrj_direct_NoDeNS_June5
```

Important differences outside the model architecture:

1. PET applies random rotational/inversion augmentation during training.

   The Transformer run used direct Cartesian outputs, but the inspected config/log did
   not show equivalent random rotation augmentation. This is probably the most
   important pipeline-level difference.

2. PET fits target scales on the active training set.

   PET fits scales after subtracting additive baselines. The Transformer run used fixed
   normalizers, with the same RMSD value `0.8124567` for energy, force, and stress.

3. PET fits an energy composition baseline in the metatrain pipeline.

   The Transformer run used an energy element-reference file. That is related in
   spirit, but in the inspected run it was not fitted on the active run in the same
   way PET fits composition weights before training.

4. PET uses Huber losses with small deltas.

   The Transformer run used `per_atom_mae`, `l2mae`, and `mae`. FairChem's default
   registered losses in this repo did not appear to include a Huber/SmoothL1 loss.

5. PET clips gradients much more tightly.

   PET direct config uses `grad_clip_norm: 1.0`. The Transformer run used
   `clip_grad_norm: 100`.

6. PET uses local neighbor-list construction with cutoff/adaptive-neighbor behavior.

   This is partly architecture and partly preprocessing. The Transformer model densifies
   atoms into sequence tokens and feeds absolute/cartesian or fractional coordinates
   plus cell information. PET explicitly constructs local neighbor tensors and edge
   vectors before the neural layers.

7. PET trains in residual target space.

   PET's training targets have additive baselines removed and scales divided out before
   loss computation. The model learns residuals, and physical-scale predictions are
   restored for metrics/evaluation.

## Things The Transformer Also Has

The Transformer/FairChem training path is not barebones. The inspected run has:

- element references for energy
- target normalizers, although fixed in that run
- AdamW
- cosine scheduler with warmup
- gradient clipping
- EMA
- AMP
- distributed training
- atom-load-balanced batching

So the point is not that PET is the only optimized pipeline. The most suspicious
differences are the rotation augmentation, fitted residual scaling, loss shape, and
local-neighbor representation.

## Direct Versus Conservative Forces

For `options-pet-oam-l-modern-mptrj-salex-direct.yaml`, PET directly predicts force
and stress through:

```text
non_conservative_force
non_conservative_stress
```

The force/stress values in this direct run are not energy gradients.

The mixed config `options-pet-oam-l-modern-mptrj-salex.yaml` asks for both:

- conservative force/stress via energy gradients
- direct non-conservative force/stress heads

When debugging, make sure you are using the direct YAML if the goal is to compare
against the direct Transformer run.

## Suggested Debugging Walkthrough

Start with the smoke launch config above and step in this order:

1. `train_model(...)`
   - confirm overrides were applied
   - confirm `training_set.indices` and `validation_set.indices`
   - inspect `target_info_dict`

2. `MemmapDataset.__getitem__`
   - inspect one MPtrj sample
   - inspect one sAlex sample
   - verify target TensorMap shapes

3. `PET.__init__`
   - inspect `self.target_names`
   - inspect `self.outputs`
   - inspect `self.additive_models`
   - inspect `self.scaler`

4. `Trainer.train(...)`, before the dataloaders
   - step through composition fitting
   - step through scaler fitting
   - inspect learned scales

5. `collate_fn_train`
   - watch the order of transforms
   - confirm random augmentation happens only for training
   - confirm neighbor lists are attached
   - confirm additive/scale removal has changed target values

6. `evaluate_model(...)`
   - confirm no energy gradients are requested in the direct config
   - confirm requested outputs are direct targets

7. `PET.forward(...)`
   - inspect `systems_to_batch(...)` output tensors
   - inspect neighbor counts/cutoffs
   - step through `_calculate_features`
   - inspect final TensorMap output keys

8. Back in `Trainer.train(...)`
   - inspect `average_by_num_atoms(...)`
   - inspect loss inputs
   - inspect gradient norm before/after clipping

## Mental Model

For this direct PET run, think of the model as learning:

```text
scaled residual targets = neural PET(local neighbor graph, edge vectors, atom types)
```

where the residual targets are produced by:

```text
raw target
  -> rotate/invert for training augmentation
  -> subtract fitted energy composition baseline where applicable
  -> divide by fitted target scales
  -> average system energy by atom count
```

For logged train/validation metrics, metatrain restores target scales before computing
errors. The additive energy baseline is not added back in the trainer metric path, but
this does not change energy MAE/RMSE because the same baseline would be added to both
predictions and targets. For exported/eval-mode PET predictions, `PET.forward(...)`
restores both scales and additive baselines.

## Appendix: CompositionModel Deep Dive

The `CompositionModel` in `src/metatrain/utils/additive/composition.py` is the PET
additive baseline model. It is not another neural network. It is a small fitted linear
model that predicts target contributions from chemical composition only.

For the direct MPtrj/sAlex PET config, this mainly means energy. In `PET.__init__`,
the composition model is created from only the targets that pass
`CompositionModel.is_valid_target(...)`. That check accepts scalar targets and
selected invariant spherical targets. It rejects ordinary Cartesian vector/tensor
targets, so `non_conservative_force` and `non_conservative_stress` are not fitted by
the composition model in `options-pet-oam-l-modern-mptrj-salex-direct.yaml`.

### Where It Sits In PET

PET stores the composition model as the first item in:

```text
model.additive_models[0]
```

During `Trainer.train(...)`, it is fitted before the scaler:

```text
model.additive_models[0].train_model(...)
model.scaler.train_model(...)
```

Then the training and validation collate functions subtract the additive baseline
before scale removal:

```text
raw targets
  -> remove additive composition contribution
  -> remove target scale
```

During evaluation/exported inference, PET reverses this direction. `PET.forward(...)`
first applies the learned scaler to the neural residual predictions, then loops over
`self.additive_models` and adds the composition contribution back to any output the
additive model supports.

### The Fitted Model

The lower-level implementation lives in
`src/metatrain/utils/additive/_base_composition.py` as `BaseCompositionModel`.

For a scalar per-structure target such as total energy, the model builds one feature
vector per structure:

```text
X_s = [count of type_1 in system s,
       count of type_2 in system s,
       ...,
       count of type_N in system s]
```

The fitted prediction is:

```text
E_base(s) = sum_Z count_Z(s) * w_Z
```

where `w_Z` is one fitted contribution for each atomic type `Z`. In matrix form, the
fit solves:

```text
XTX * W = XTY
```

where:

- `XTX` is accumulated as `X.T @ X`
- `XTY` is accumulated as `X.T @ Y`
- `W` contains the per-species composition weights

The code stores these normal-equation tensors as metatensor `TensorMap`s, with one
block per target block. For scalar energy, this is the simple stoichiometric
least-squares problem above. For supported spherical targets, only invariant blocks
are included. For supported rank-2 spherical targets, the fitting path uses the trace
as the invariant part.

### Accumulation Details

`CompositionModel.train_model(...)` builds a dedicated dataloader just for fitting the
baseline. A few implementation details matter:

- it does not shuffle
- it does not drop the final batch
- it requires float64 training data
- in distributed runs, each rank accumulates its shard and then all-reduces `XTX` and
  `XTY`
- it applies `atomic_basis_transform` before fitting
- it can request neighbor lists if another additive model needs them

In this direct config, `zbl: false`, so there is no second additive model to remove
before fitting the composition baseline. If ZBL were enabled, the composition fit would
first subtract the other additive model from the targets and then fit the residual.

The actual accumulation happens in `BaseCompositionModel.accumulate(...)`:

- for per-structure targets, `_compute_X_per_structure(...)` creates count vectors
- for per-atom targets, `_compute_X_per_atom(...)` creates one-hot atom-type rows
- `XTX` receives `X.T @ X`
- `XTY` receives `X.T @ Y`

For per-atom targets, the code uses type-wise scatter/add logic instead of a dense
matrix multiply. That avoids leaking `NaN` values between atom types.

### Fitting And Fixed Weights

The training config has:

```yaml
atomic_baseline: {}
```

That empty dict means there are no fixed composition weights, so the model fits all
valid new composition targets from the training data.

If fixed weights are supplied, `BaseCompositionModel._sanitize_fixed_weights(...)`
normalizes them into a dict from atomic type to weight. A single float means the same
fixed value for every atomic type. A per-type dict must contain every atomic type used
by the model.

For fitted targets, `BaseCompositionModel.fit(...)` solves the normal equations. For
per-structure targets such as energy, it calls `_solve_linear_system(...)`, which adds
a tiny diagonal regularizer:

```text
regularizer = 1e-14 * mean(abs(diag(XTX)))
```

and then uses `torch.linalg.solve(...)`. For per-atom targets, `XTX` is diagonal, so
the code divides `XTY` by the atom-type counts instead of solving a dense system.

After fitting, `CompositionModel.train_model(...)` serializes the fitted TensorMaps
into buffers named like:

```text
energy_composition_buffer
```

Those buffers keep the learned composition weights inside the model state so they can
be restored, moved between devices/dtypes, exported, and used during evaluation.

### Forward Pass

`BaseCompositionModel.forward(...)` always starts by building atom-level sample labels
and one-hot atom-type features. For a system-level output, it computes atom
contributions and then sums over the atom sample dimension if the requested
`ModelOutput.sample_kind` is `system`.

For energy, the operational meaning is:

```text
each atom gets the fitted weight for its element
system baseline energy = sum of those atom weights
```

`remove_additive(...)` then aligns the additive TensorMap samples and blocks with the
actual target TensorMap and subtracts the additive contribution block by block. The
subtracted additive values are detached, so the training loss does not backpropagate
through the baseline subtraction.

### What It Does And Does Not Explain

The composition model can have a large effect on energy learning because it removes a
simple extensive stoichiometric trend before PET learns the residual. This is
especially relevant for mixed-composition datasets such as MPtrj/sAlex.

It does not directly predict force or stress in the direct config. The direct
`non_conservative_force` and `non_conservative_stress` targets are learned by PET's
neural heads, then scaled/unscaled by the scaler. The composition baseline only enters
those targets if a target is compatible with `CompositionModel.is_valid_target(...)`,
which the Cartesian direct force/stress targets are not.

This is related to, but not identical to, using fixed element reference energies in
another codebase. Here the baseline is fitted at run start from the active metatrain
training data or subsplit, is stored with the PET model, is subtracted in the collate
pipeline, and is added back inside PET evaluation/export behavior.

### Debugger Checkpoints

Useful breakpoints for understanding this path:

- `src/metatrain/pet/model.py`: `composition_model = CompositionModel(...)`
- `src/metatrain/pet/trainer.py`: `model.additive_models[0].train_model(...)`
- `src/metatrain/utils/additive/composition.py`: `CompositionModel.train_model(...)`
- `src/metatrain/utils/additive/_base_composition.py`: `BaseCompositionModel.accumulate(...)`
- `src/metatrain/utils/additive/_base_composition.py`: `BaseCompositionModel.fit(...)`
- `src/metatrain/utils/additive/remove.py`: `remove_additive(...)`
- `src/metatrain/pet/model.py`: the evaluation-only post-processing block in
  `PET.forward(...)`

Variables worth inspecting:

- `self.atomic_types`
- `self.target_infos`
- `self._new_outputs`
- `self.model.XTX["energy"]`
- `self.model.XTY["energy"]`
- `self.model.weights["energy"]`
- target TensorMap values before and after `remove_additive(...)`

The quickest sanity check is to confirm that `self.target_infos` in the composition
model contains `energy` and does not contain `non_conservative_force` or
`non_conservative_stress` for the direct MPtrj/sAlex config.

## Appendix: Scaler Deep Dive

The `Scaler` in `src/metatrain/utils/scaler/scaler.py` is the second fitted
preprocessing model in the PET training pipeline. Its job is to learn target
magnitudes from the training set so that PET is trained on normalized residual
targets rather than raw physical-unit values.

The high-level order is:

```text
raw target
  -> subtract additive baseline where applicable
  -> divide by atom count unless this target is per-structure
  -> compute RMS-like target scale
  -> divide targets by that scale during training
```

This is not mean/std standardization. The scaler does not subtract a learned mean.
For each fitted scale it accumulates a count `N` and a sum of squared values `Y2`,
then computes:

```text
scale = sqrt(Y2 / N)
```

So it is closer to a root-mean-square magnitude than a centered standard deviation.

### Where It Sits In PET

`PET.__init__` creates:

```text
model.scaler = Scaler(hypers={}, dataset_info=train_dataset_info)
```

During `Trainer.train(...)`, the scaler is fitted after the composition model:

```text
model.additive_models[0].train_model(...)
model.scaler.train_model(...)
```

The scaler fit receives the already-created additive models. In the direct MPtrj/sAlex
config, the important additive model is the energy composition baseline. Therefore the
scaler sees energy after subtracting the fitted composition contribution.

After fitting, the training and validation collate functions call:

```text
get_remove_scale_transform(scaler)
```

That transform divides targets by the learned per-target scale before the optimizer
ever sees them.

### What Data The Scaler Fits On

`Scaler.train_model(...)` builds a dedicated dataloader for scale fitting. Like the
composition fitting dataloader, it is meant for statistics, not stochastic training:

- it does not shuffle
- it does not drop the final batch
- it requires float64 data
- it applies the provided `initial_transforms`, including `atomic_basis_transform`
- in distributed training, each rank accumulates local statistics and then all-reduces
  them

For each batch, the scaler does:

```text
systems, targets, extra_data = unpack_batch(batch)
targets = remove_additive(systems, targets, additive_model, ...)
targets = average_by_num_atoms(targets, systems, per_structure_targets)
self.model.accumulate(systems, targets, extra_data)
```

This means the scales are fitted on the same residual/averaged target space that the
neural network will train against.

### Energy, Force, And Stress In This Direct Config

The direct MPtrj/sAlex config has:

```yaml
per_structure_targets:
  - non_conservative_stress
```

So the scaler sees:

- `energy`: composition baseline removed, then divided by number of atoms
- `non_conservative_force`: not composition-corrected and not divided by atom count,
  because force samples contain an `atom` dimension already
- `non_conservative_stress`: not composition-corrected and not divided by atom count,
  because it is explicitly listed as per-structure

The atom-count logic comes from `average_by_num_atoms(...)` in
`src/metatrain/utils/per_atom.py`. That helper divides blocks whose samples do not
contain `"atom"`, unless the target name is listed in `per_structure_targets`.

### Accumulated Statistics

The lower-level implementation lives in
`src/metatrain/utils/scaler/_base_scaler.py` as `BaseScaler`.

For every target block, `BaseScaler.accumulate(...)` collects:

```text
N  = number of valid target values
Y2 = sum of squared target values
```

If a target has an accompanying mask in `extra_data`, the mask controls which values
count. NaNs are also masked out and set to zero before summation so they do not
contribute to `Y2`.

For per-structure targets, there is one scale row with sample label:

```text
atomic_type = -1
```

For per-atom targets, the scaler computes separate rows by atomic type. In other
words, atom-level quantities can get different scales for different elements.

After all batches are accumulated, `BaseScaler.fit(...)` computes:

```text
scale = sqrt(Y2 / N)
```

without Bessel's correction and without subtracting a mean. If a scale becomes NaN
because no samples were seen, it is replaced with `1.0`.

### Per-Target Versus Per-Property Scales

There are two related scale concepts in this implementation:

- per-target scales
- per-property scales

The per-target scale is the main one. It gives one overall magnitude for a target, or
one magnitude per atomic type for per-atom targets.

If a target has multiple blocks or multiple properties, the scaler can also compute
per-property scales. It does this in a second accumulation pass:

```text
target
  -> remove per-target scale
  -> accumulate per-block/per-property Y2 and N
  -> compute per-property RMS correction
```

The full stored scale is:

```text
full scale = per-target scale * per-property scale
```

One subtle training detail: `get_remove_scale_transform(...)` removes only the
per-target scale from the dataloader targets. Later, inside the trainer, per-property
scales are applied to the predictions before loss computation. For single-property
targets, this extra step is effectively a no-op because the per-property scale is `1`.

### How Scaling Is Applied

`Scaler.forward(...)` delegates to `BaseScaler.forward(...)`.

With `remove=True`, it divides by the selected scale:

```text
scaled target = target / scale
```

With `remove=False`, it multiplies:

```text
physical-unit value = scaled value * scale
```

For per-structure targets, the same scale row is broadcast across the whole block. For
per-atom targets, the code looks up the atomic type of each atom and applies that
element-specific scale row.

During training:

- collate subtracts additive baselines
- collate removes per-target scale from targets
- trainer applies per-property scale to predictions before loss if needed

During metric logging and evaluation/exported inference, metatrain multiplies scales
back so predictions and metrics are reported in physical units.

### Fixed Scaling Weights

`fixed_scaling_weights` can override fitted scales for selected targets. A single
float applies the same scale everywhere. For per-atom targets, a dict can specify a
different scale for each atomic type. If all new targets have fixed weights and no
per-property scale is needed, `Scaler.train_model(...)` can skip the accumulation
pass.

In the reconstructed direct MPtrj/sAlex config, these scales are fitted from the
training subset unless `fixed_scaling_weights` is changed.

### What It Does And Does Not Explain

The scaler can strongly affect optimization because energy, force, and stress have
different natural units and magnitudes. Scaling makes the neural residual targets
numerically comparable before the weighted Huber losses are applied.

The scaler is not a physical model and does not remove chemical trends by itself. The
composition model removes the simple stoichiometric energy baseline; the scaler then
normalizes the size of the remaining targets.

### Debugger Checkpoints

Useful breakpoints for this path:

- `src/metatrain/pet/trainer.py`: `model.scaler.train_model(...)`
- `src/metatrain/utils/scaler/scaler.py`: `Scaler.train_model(...)`
- `src/metatrain/utils/additive/remove.py`: `remove_additive(...)`
- `src/metatrain/utils/per_atom.py`: `average_by_num_atoms(...)`
- `src/metatrain/utils/scaler/_base_scaler.py`: `BaseScaler.accumulate(...)`
- `src/metatrain/utils/scaler/_base_scaler.py`: `BaseScaler.fit(...)`
- `src/metatrain/utils/scaler/remove.py`: `remove_scale(...)`
- `src/metatrain/pet/trainer.py`: the per-property scaling call before loss
- `src/metatrain/pet/model.py`: the evaluation-only scaler call in `PET.forward(...)`

Variables worth inspecting:

- `self.model.N["energy"]`
- `self.model.Y2["energy"]`
- `self.model.scales["energy"]`
- `self.model.per_target_scales["energy"]`
- `self.model.per_property_scales`
- target values before and after `remove_additive(...)`
- target values before and after `average_by_num_atoms(...)`
- target values before and after `get_remove_scale_transform(...)`

The quickest sanity check is to compare a residual target before and after scale
removal. After `get_remove_scale_transform(...)`, the target should have the same
structure and units conceptually, but its numerical magnitude should be divided by the
fitted RMS scale.

## Appendix: Neighbor-List Deep Dive

The neighbor-list path answers the question: for each atom, which other atoms are
within the PET cutoff, including periodic images? PET uses that list to build the edge
features consumed by the Cartesian transformer layers.

The high-level order in the PET training dataloader is:

```text
individual System objects
  -> optional random rotation/inversion
  -> construct neighbor lists on each System
  -> subtract additive baselines
  -> remove fitted target scales
  -> PET forward pass converts neighbor lists into edge tensors
```

For validation, the same neighbor-list construction happens, but without random
rotation/inversion before it.

### What PET Requests

PET declares its neighbor-list requirement in `src/metatrain/pet/model.py`:

```python
self.requested_nl = NeighborListOptions(
    cutoff=self.cutoff,
    full_list=True,
    strict=True,
)
```

and exposes it through:

```python
def requested_neighbor_lists(self):
    return [self.requested_nl]
```

The trainer collects neighbor-list requests from the model with
`get_requested_neighbor_lists(model)`. That helper walks the module tree, asks every
module that has `requested_neighbor_lists()` for its options, and merges duplicate
requests.

For the direct MPtrj/sAlex PET config, the important option is:

```text
full_list=True
```

This means PET wants directed/full pair information. If atom `i` sees atom `j`, the
reverse direction is also expected to be present as its own edge, subject to periodic
image bookkeeping.

### Where Construction Happens

The construction is inserted into the collate function in `Trainer.train(...)`:

```text
get_system_with_neighbor_lists_transform(requested_neighbor_lists)
```

This transform lives in `src/metatrain/utils/neighbor_lists.py`. It loops over each
`System` in the collated batch and calls:

```text
get_system_with_neighbor_lists(system, requested_neighbor_lists)
```

That function first converts the metatomic `System` to an ASE `Atoms` object:

```text
atoms = system_to_ase(system)
```

Then it computes every requested neighbor list that is not already attached to the
system.

### Backend: Vesin Or ASE

The low-level computation is `_compute_single_neighbor_list(...)`.

If the structure is either fully periodic or fully non-periodic, metatrain uses
`vesin`:

```python
vesin.ase_neighbor_list("ijSD", atoms, cutoff=options.cutoff)
```

If the structure has mixed periodic boundary conditions, it falls back to ASE:

```python
ase.neighborlist.neighbor_list("ijSD", atoms, cutoff=options.cutoff)
```

The `"ijSD"` request means:

```text
i = first atom index
j = second atom index
S = integer cell shift vector
D = Cartesian displacement vector
```

So the backend finds every pair within the cutoff and returns both the atom indices
and the periodic image needed to describe the pair.

### Full List Versus Half List

`neighbor_lists.py` contains logic for half lists when `options.full_list` is false.
That path drops duplicate pairs and keeps a canonical representative for periodic
self-image cases.

PET asks for:

```text
full_list=True
```

so that half-list filtering is skipped for PET. The full directed list is important
because PET later builds reverse-edge indexing for message passing.

### What Gets Attached To The System

The computed neighbor list is wrapped as a metatensor `TensorBlock`.

Its samples are:

```text
first_atom
second_atom
cell_shift_a
cell_shift_b
cell_shift_c
```

Its values have shape:

```text
(n_edges, 3, 1)
```

where the `3` component stores the Cartesian displacement vector returned by the
backend.

Then metatrain attaches the block to the `System`:

```text
register_autograd_neighbors(system, neighbors)
system.add_neighbor_list(options, neighbors)
```

The returned `System` now carries the neighbor list under the exact
`NeighborListOptions` object PET requested.

### What PET Uses From The Neighbor List

Inside PET, `systems_to_batch(...)` calls `concatenate_structures(...)` in
`src/metatrain/pet/modules/structures.py`.

For each system, it reads:

```python
neighbor_list = system.get_neighbor_list(neighbor_list_options)
nl_values = neighbor_list.samples.values
```

The important subtlety is that PET primarily reads the neighbor-list sample labels:

```text
first atom
second atom
cell shift
```

Then PET recomputes differentiable edge vectors from the current positions and cell:

```python
edge_vectors = positions[neighbors] - positions[centers] + cell_contributions
```

where:

```text
cell_contributions = cell_shift_a * cell[0]
                   + cell_shift_b * cell[1]
                   + cell_shift_c * cell[2]
```

This means the neighbor list decides which pairs exist, but PET's actual edge vectors
come from the `System` positions/cell at forward time.

### Adaptive Cutoff And PET Edge Format

After the base cutoff neighbor list is read, PET may apply its optional adaptive
cutoff logic:

```text
num_neighbors_adaptive is not None
```

If enabled, PET computes per-atom adaptive cutoffs and masks the base neighbor list
down to approximately the requested number of neighbors. If disabled, every edge uses
the fixed model cutoff.

PET then computes cutoff factors with the configured cutoff function, currently
`cosine` or `bump`.

Finally, PET converts the flat edge array into NEF format:

```text
node x edge-slot x feature
```

It also builds reverse-edge indices so message passing can efficiently find the
corresponding `j -> i` edge for an `i -> j` edge.

### Why Construction Happens After Rotation

In the training collate stack, random rotation/inversion is applied before neighbor
list construction. A pure rotation or inversion preserves distances, so the set of
pairs within the cutoff should be unchanged. Building after augmentation still keeps
the attached neighbor-list metadata consistent with the transformed `System`.

The actual edge vectors used by PET are recomputed from the transformed positions and
cell, so they are in the augmented coordinate frame.

### Debugger Checkpoints

Useful breakpoints for this path:

- `src/metatrain/pet/model.py`: `self.requested_nl = NeighborListOptions(...)`
- `src/metatrain/pet/model.py`: `requested_neighbor_lists(...)`
- `src/metatrain/pet/trainer.py`: `get_requested_neighbor_lists(model)`
- `src/metatrain/pet/trainer.py`: `get_system_with_neighbor_lists_transform(...)`
- `src/metatrain/utils/neighbor_lists.py`: `get_system_with_neighbor_lists_transform(...)`
- `src/metatrain/utils/neighbor_lists.py`: `get_system_with_neighbor_lists(...)`
- `src/metatrain/utils/neighbor_lists.py`: `_compute_single_neighbor_list(...)`
- `src/metatrain/pet/modules/structures.py`: `concatenate_structures(...)`
- `src/metatrain/pet/modules/structures.py`: `systems_to_batch(...)`

Variables worth inspecting:

- `requested_neighbor_lists`
- `options.cutoff`
- `options.full_list`
- `system.positions`
- `system.cell`
- `system.pbc`
- `atoms.pbc`
- `nl_i`, `nl_j`, `nl_S`, `nl_D`
- `neighbors.samples.values`
- `neighbors.values`
- `centers`, `neighbors`, `cell_shifts`
- `edge_vectors`
- `edge_distances`
- `reverse_neighbor_index`

The quickest sanity check is to inspect `neighbors.samples.values` right after
`_compute_single_neighbor_list(...)`. Each row should identify one edge as:

```text
first_atom, second_atom, cell_shift_a, cell_shift_b, cell_shift_c
```

Then inspect `edge_vectors` in `systems_to_batch(...)` to confirm PET has converted
those labels into the actual Cartesian edge vectors used by the model.

## Appendix: Validation Metrics, Scaling, And Baselines

There are three closely related, but distinct, quantities:

```text
scale value      = the fitted RMS-like number, sqrt(Y2 / N)
scaled target    = target divided by that scale
restored target  = scaled target multiplied by that scale
```

The confusing part is terminology. In code, the `Scaler` object can either remove a
scale or apply a scale. Both operations are methods of the "scaler", so both can sound
like "scaling" in conversation.

The code meaning is:

```text
remove=True   -> divide by scale      -> normalize for training
remove=False  -> multiply by scale    -> restore physical magnitude
```

So if the fitted value is:

```text
s = sqrt(Y2 / N)
```

then target preprocessing does:

```text
y_scaled = y / s
```

and metric/eval restoration does:

```text
y_restored = y_scaled * s
```

### Where Targets Are Scaled Down

During dataloader collation, the PET trainer applies:

```text
get_remove_scale_transform(scaler)
```

That calls `remove_scale(...)` in `src/metatrain/utils/scaler/remove.py`, which calls:

```python
scaler(
    systems,
    targets,
    remove=True,
    use_per_target_scales=True,
    use_per_property_scales=False,
)
```

This is the normalization direction. It divides by the fitted per-target scale. In
other words, it removes the physical target magnitude before the model/loss sees the
target.

For energy in this direct PET config, the target path is effectively:

```text
raw total energy
  -> subtract fitted composition baseline
  -> divide by fitted energy scale
  -> divide by number of atoms in the training loop
```

For force:

```text
raw direct force
  -> divide by fitted force scale
```

For stress:

```text
raw direct stress
  -> divide by fitted stress scale
```

Stress is listed in `per_structure_targets`, so it is not divided by atom count.

### Validation Loss Space

The validation loss is computed in the same normalized residual space as training.

In `src/metatrain/pet/trainer.py`, the validation loop calls:

```python
predictions = evaluate_model(..., is_training=False)
```

Then it applies `average_by_num_atoms(...)` to predictions and targets. The dataloader
targets have already had additive baselines and per-target scales removed.

Before the loss, the trainer applies only per-property scaling to predictions:

```python
predictions = model.scaler(
    systems,
    predictions,
    remove=False,
    use_per_target_scales=False,
    use_per_property_scales=True,
)
```

For single-property targets, this is effectively a no-op because the per-property
scale is `1`. The important point is that validation loss is still a training-space
quantity, not the final physical-unit MAE/RMSE.

### Validation Metric Space

For logged train/validation metrics, the trainer restores the per-target scale before
calling the metric accumulators.

The validation path in `src/metatrain/pet/trainer.py` does:

```python
scaled_predictions = model.scaler(
    systems,
    predictions,
    remove=False,
    use_per_target_scales=True,
    use_per_property_scales=False,
)

scaled_targets = model.scaler(
    systems,
    targets,
    remove=False,
    use_per_target_scales=True,
    use_per_property_scales=False,
)
```

Despite the variable name `scaled_predictions`, this is the restoration direction:
`remove=False` means multiply by the fitted scale. A less confusing mental name would
be:

```text
scale_restored_predictions
scale_restored_targets
```

Then the trainer computes:

```python
val_rmse_calculator.update(scaled_predictions, scaled_targets, extra_data)
val_mae_calculator.update(scaled_predictions, scaled_targets, extra_data)
```

The metric accumulators themselves just compare the two TensorMaps they receive. They
do not know about composition baselines or scalers.

### What Happens To The Composition Baseline In Metrics

The trainer metric path restores scale, but it does not add the composition baseline
back.

For energy, the metric comparison is approximately:

```text
prediction_residual_per_atom
target_residual_per_atom
```

not:

```text
prediction_absolute_energy_per_atom
target_absolute_energy_per_atom
```

This does not change MAE or RMSE because the same fitted baseline would be added to
both sides:

```text
(prediction_residual + baseline) - (target_residual + baseline)
  = prediction_residual - target_residual
```

After division by atom count, the same cancellation holds per atom:

```text
(prediction_residual + baseline) / N_atoms
  - (target_residual + baseline) / N_atoms
= (prediction_residual - target_residual) / N_atoms
```

So the logged energy MAE/RMSE is still a physical-unit error, e.g. meV/atom, even
though the absolute baseline has not been added back in the metric path.

### Exported/Eval-Mode Prediction Path

There is a separate code path for actual eval-mode PET predictions.

Inside `PET.forward(...)` in `src/metatrain/pet/model.py`, the post-processing block
only runs when:

```python
if not self.training:
```

In that block, PET first restores the full scale:

```python
return_dict = self.scaler(
    systems,
    return_dict,
    selected_atoms=selected_atoms,
    use_per_target_scales=True,
    use_per_property_scales=True,
)
```

Since `remove` defaults to `False`, this multiplies predictions by the fitted scales.
Then PET loops over additive models:

```python
for additive_model in self.additive_models:
    additive_contributions = additive_model(...)
    return_dict[name] = return_dict[name] + additive_contributions[name]
```

This is the path that returns full physical predictions with both:

```text
scale restored
composition baseline added back
```

In the inspected PET trainer validation loop, `evaluate_model(..., is_training=False)`
does not itself switch the module into `eval()` mode. The `is_training` argument
controls gradient handling in `evaluate_model(...)`; it is not the same thing as
`model.eval()`. Therefore the ordinary training-loop validation metrics use the
trainer restoration path described above, not the `if not self.training` postprocessing
inside `PET.forward(...)`.

### Debugger Checkpoints

Useful breakpoints for this distinction:

- `src/metatrain/utils/scaler/remove.py`: `remove_scale(...)`
- `src/metatrain/utils/scaler/scaler.py`: `Scaler.forward(...)`
- `src/metatrain/utils/scaler/_base_scaler.py`: `BaseScaler.forward(...)`
- `src/metatrain/pet/trainer.py`: validation call to `evaluate_model(...)`
- `src/metatrain/pet/trainer.py`: `scaled_predictions = model.scaler(...)`
- `src/metatrain/pet/trainer.py`: `val_rmse_calculator.update(...)`
- `src/metatrain/pet/model.py`: `if not self.training:`

Variables worth inspecting:

- `targets["energy"].block().values` after collate
- `predictions["energy"].block().values` before metric restoration
- `scaled_targets["energy"].block().values`
- `scaled_predictions["energy"].block().values`
- `model.scaler.model.per_target_scales["energy"]`
- `model.additive_models[0].model.weights["energy"]`
- `model.training`

The quick check is:

```text
scaled_targets ~= targets * fitted_per_target_scale
```

for the relevant target block. The additive baseline will not appear in
`scaled_targets` inside the trainer metric path.


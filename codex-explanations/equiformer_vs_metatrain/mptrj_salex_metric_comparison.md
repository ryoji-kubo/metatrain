# MPtrj/sAlex Validation Metric Comparison

This note explains how to compare validation metrics between:

- metatrain PET run:
  `outputs/2026-06-29/07-47-01`
- Equiformer/FairChem run:
  `/home/ryoji/equivarient/equiformer_v3/logs/omat24/equiformer_v3/logs/wandb/2026-05-27-14-24-00-mptrj_direct_160k_NoDeNS`

The main question is whether the validation metrics are computed in compatible
ways. The short answer is: we can make a reasonably apples-to-apples comparison
for validation energy-per-atom MAE, force MAE, and stress MAE, but not for loss.

## Comparable Metrics

Use these metric mappings:

| Quantity | metatrain metric | Equiformer/FairChem metric |
| --- | --- | --- |
| energy | `validation energy MAE (per atom)` | `val/energy_per_atom_mae` |
| force | `validation non_conservative_force MAE (per atom)` | `val/forces_mae` |
| stress | `validation non_conservative_stress MAE` | `val/stress_mae` |

FairChem logs these metrics in eV-like units:

- `val/energy_per_atom_mae`: eV/atom
- `val/forces_mae`: eV/A
- `val/stress_mae`: eV/A^3

metatrain logs the corresponding values in meV-like units:

- `validation energy MAE (per atom)`: meV/atom
- `validation non_conservative_force MAE (per atom)`: meV/A
- `validation non_conservative_stress MAE`: meV/A^3

Therefore, multiply Equiformer/FairChem values by `1000` before comparing to
metatrain's printed values.

## Current Comparison

Using:

```bash
python scripts/compare_mptrj_salex_metrics.py
```

the current comparison is:

```text
Comparable sAlex validation MAEs
metric      metatrain   equiformer       unit
energy       114.7500      60.4426   meV/atom
force        117.3800      64.8488      meV/A
stress         4.7848       3.1460    meV/A^3
```

For metatrain, this table uses the final epoch row in:

```text
outputs/2026-06-29/07-47-01/train.csv
```

For Equiformer, this table uses:

```text
wandb-summary.json
```

from the referenced run directory.

## Why These Metrics Match

### Equiformer/FairChem

The Equiformer run uses FairChem's evaluator. The relevant implementation is:

```text
/home/ryoji/equivarient/equiformer_v3/src/fairchem/core/modules/evaluator.py
```

The evaluator computes metric names from the configured target and function names,
then accumulates `total / numel`.

Relevant code:

- `Evaluator.eval(...)` loops through requested metrics:
  `evaluator.py:88-107`
- `Evaluator.update(...)` accumulates `total` and `numel`:
  `evaluator.py:109-130`
- `mae(...)` is elementwise absolute error:
  `evaluator.py:164-170`
- `metrics_dict(...)` returns `sum(error)` and `error.numel()`:
  `evaluator.py:133-150`

The Equiformer trainer prepares prediction/target tensors here:

```text
/home/ryoji/equivarient/equiformer_v3/src/fairchem/core/trainers/ocp_trainer.py
```

Relevant code:

- `_compute_metrics(...)` clones predictions:
  `ocp_trainer.py:373-378`
- atom-level force targets can be masked to free atoms:
  `ocp_trainer.py:383-405`
- atom targets are reshaped as `(num_atoms, -1)` and system targets as
  `(batch_size, -1)`:
  `ocp_trainer.py:407-411`
- predictions are denormalized before metrics:
  `ocp_trainer.py:413`
- metrics are sent to the evaluator:
  `ocp_trainer.py:428`

The referenced Equiformer run config says:

```yaml
evaluation_metrics:
  metrics:
    energy:
      - mae
      - per_atom_mae
    forces:
      - mae
      - cosine_similarity
    stress:
      - mae
  primary_metric: forces_mae
```

It also says:

```yaml
model:
  direct_prediction: true
  regress_forces: true
  regress_stress: true
```

So `val/forces_mae` and `val/stress_mae` are direct prediction metrics.

### metatrain

The metatrain PET trainer computes validation metrics in:

```text
src/metatrain/pet/trainer.py
```

Relevant code:

- validation predictions are evaluated at `trainer.py:566-571`
- predictions and targets are passed through `average_by_num_atoms(...)` at
  `trainer.py:573-579`
- scales are restored before metric accumulation at `trainer.py:604-619`
- validation RMSE/MAE accumulators are updated at `trainer.py:634-640`
- final validation metrics are created at `trainer.py:642-655`

The MAE accumulator is in:

```text
src/metatrain/utils/metrics.py
```

Relevant code:

- elementwise absolute error is accumulated at `metrics.py:277-304`
- final MAE is `sum_abs_error / num_elements` at `metrics.py:387-393`

The atom-normalization helper is:

```text
src/metatrain/utils/per_atom.py
```

Relevant code:

- `average_by_num_atoms(...)` skips keys listed in `per_structure_targets`:
  `per_atom.py:8-37`
- system-level values are divided by atom count when not skipped:
  `per_atom.py:40-90`

In the metatrain run, the expanded options contain:

```yaml
per_structure_targets:
  - non_conservative_stress
```

and the energy target has:

```yaml
forces: false
stress: false
```

Therefore:

- energy is compared per atom
- force is already atom-level and is not divided by atom count
- `non_conservative_stress` is kept as a per-structure tensor, not divided by atom
  count, in the training-loop validation metrics

## Important Caveats

### Compare Epoch Validation, Not Final Export Evaluation For Stress

Use metatrain's epoch validation rows in:

```text
outputs/2026-06-29/07-47-01/train.csv
```

Do not use the final exported-model evaluation stress line in `train.log` for this
specific comparison.

Reason: the epoch validation loop respects `per_structure_targets` and logs:

```text
validation non_conservative_stress MAE
```

The final exported-model evaluation later logs:

```text
non_conservative_stress MAE (per atom)
```

That final stress number is not the same normalization as FairChem's
`val/stress_mae`.

### Do Not Compare Loss Directly

The loss values are not apples-to-apples.

Reasons include:

- Equiformer uses its own loss functions and target coefficients
- metatrain uses Huber loss settings from the PET config
- target normalization/scaling differs
- the model architectures and prediction heads are different

Use validation MAEs for comparison, not `loss`.

### Use Equiformer Energy Per Atom, Not Raw Energy MAE

Equiformer logs both:

```text
val/energy_mae
val/energy_per_atom_mae
```

Compare metatrain energy to:

```text
val/energy_per_atom_mae
```

not to:

```text
val/energy_mae
```

because metatrain's PET training-loop metric divides system-level energy by atom
count.

### Force Masking

FairChem can evaluate forces only on free atoms when `eval_on_free_atoms: true`.
The relevant code is in `ocp_trainer.py:383-405`.

For the local `sAlex val_30k` ASE-LMDB, I sampled 200 structures and found no ASE
constraints. That suggests the free-atom mask is effectively all atoms for this
validation set. If a future validation set contains constraints, this becomes a
real difference: FairChem may mask fixed atoms while metatrain currently evaluates
all atoms in the memmap target.

## Best-Checkpoint Policy

The referenced Equiformer run uses:

```yaml
primary_metric: forces_mae
```

For that run, the last validation epoch and best force-validation epoch are both
epoch 70.

For the referenced metatrain run:

- last epoch force MAE: `117.38 meV/A`
- best force MAE among the five logged epochs: `115.77 meV/A` at epoch 1

So decide whether comparison should be "last epoch" or "best validation force MAE",
and apply that policy consistently to both runs.

## Reproducible Parser

The comparison helper is:

```text
scripts/compare_mptrj_salex_metrics.py
```

It reads:

```text
outputs/2026-06-29/07-47-01/train.csv
```

and:

```text
/home/ryoji/equivarient/equiformer_v3/logs/omat24/equiformer_v3/logs/wandb/2026-05-27-14-24-00-mptrj_direct_160k_NoDeNS/wandb/run-20260527_142306-2026-05-27-14-24-00-mptrj_direct_160k_NoDeNS/files/wandb-summary.json
```

and prints the comparable validation MAEs after converting Equiformer values from
eV units to meV units.

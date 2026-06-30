# PET Tutorial Repository Guide

This note maps the beginner PET tutorial onto the source tree. It is meant to
answer: when you run the tutorial command, which files are involved, and what
does each of them do?

The tutorial entry point is:

```bash
cd examples/0-beginner
python -m metatrain train options-scratch.yaml
```

This is equivalent to:

```bash
mtt train options-scratch.yaml
```

The `mtt` command is only a console-script wrapper around
`metatrain.__main__:main`, defined in `pyproject.toml`.

## High-Level Layout

The repo has four important layers for this tutorial:

```text
examples/0-beginner/          Tutorial inputs and narrative examples
src/metatrain/__main__.py     CLI entry point used by mtt and python -m metatrain
src/metatrain/cli/            train/eval/export command implementations
src/metatrain/pet/            PET model, PET trainer, and PET hyperparameter docs
src/metatrain/utils/          Shared config, data, loss, metrics, IO, logging helpers
```

For the PET tutorial, the important execution path is:

```text
options-scratch.yaml
  -> src/metatrain/__main__.py
  -> src/metatrain/cli/train.py
  -> src/metatrain/utils/omegaconf.py
  -> src/metatrain/utils/data/get_dataset.py
  -> src/metatrain/utils/data/readers/ase.py
  -> src/metatrain/pet/__init__.py
  -> src/metatrain/pet/model.py
  -> src/metatrain/pet/trainer.py
  -> outputs/.../model.ckpt, model.pt, train.log, train.csv
```

## Tutorial Files

### `examples/0-beginner/03-train_from_scratch.py`

This is the source file for the rendered online tutorial. It is mostly prose
and code snippets for documentation generation, not the main training logic.

It explains:

- using `ethanol_reduced_100.xyz` as an example dataset
- running `mtt train options-scratch.yaml`
- expected output files under `outputs/<date>/<time>/`
- running evaluation with `mtt eval model.pt eval-scratch.yaml`

### `examples/0-beginner/options-scratch.yaml`

This is the main tutorial training configuration. It selects PET:

```yaml
architecture:
  name: pet
```

It overrides a small number of PET defaults:

```yaml
model:
  cutoff: 4.5
training:
  num_epochs: 10
  batch_size: 10
```

It points metatrain to the local extended XYZ dataset:

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

A subtle but important point: this YAML does not explicitly list `forces`.
For energy targets, metatrain expands the config to look for `forces` by
default. Since the dataset stores a `forces` array, PET trains on energy and
forces.

### `examples/0-beginner/ethanol_reduced_100.xyz`

This is the native dataset used by the tutorial. It is an extended XYZ file
that ASE can read. The header looks like:

```text
Properties=species:S:1:pos:R:3:forces:R:3 energy=...
```

That means each frame contains:

- atomic species
- atomic positions
- per-atom force vectors under the key `forces`
- per-structure energy under the key `energy`

### `examples/0-beginner/eval-scratch.yaml`

This is the evaluation dataset config used after training. It points to the
same ethanol file and target key:

```yaml
systems:
  read_from: ./ethanol_reduced_100.xyz
  length_unit: Angstrom

targets:
  energy:
    key: energy
    unit: eV
```

## CLI Entry Point

### `pyproject.toml`

This file defines the package metadata and the `mtt` executable:

```toml
[project.scripts]
mtt = "metatrain.__main__:main"
```

So `mtt train ...` and `python -m metatrain train ...` run the same Python
entry point.

### `src/metatrain/__main__.py`

This builds the top-level command-line parser. It registers subcommands from:

- `src/metatrain/cli/train.py`
- `src/metatrain/cli/eval.py`
- `src/metatrain/cli/export.py`

For `train`, it also creates the timestamped output directory:

```text
outputs/YYYY-MM-DD/HH-MM-SS/
```

Then it calls `train_model(...)` in `src/metatrain/cli/train.py`.

This is a good first VSCode breakpoint if you want to inspect how command-line
arguments are parsed.

## Training Flow

### `src/metatrain/cli/train.py`

This is the central orchestration file for the tutorial.

For `options-scratch.yaml`, it does the following:

1. Validates base options.
2. Reads `architecture.name: pet`.
3. Imports the PET architecture dynamically.
4. Merges user YAML with base defaults and PET defaults.
5. Selects device and dtype.
6. Expands the dataset config.
7. Loads the dataset.
8. Splits train/validation/test sets.
9. Expands the loss config.
10. Creates `DatasetInfo`.
11. Instantiates the PET model and PET trainer.
12. Calls `trainer.train(...)`.
13. Saves the final checkpoint and exported model.

Key functions/classes to inspect:

- `train_model(...)`: the main orchestration function
- `expand_dataset_config(...)`: expands shorthand YAML
- `get_dataset(...)`: loads the configured dataset
- `DatasetInfo`: passes atomic types, units, and target metadata to PET

The tutorial's `test_set: 0.1` and `validation_set: 0.1` are handled here. The
actual split indices are saved under the output directory in:

```text
outputs/.../indices/
```

The fully expanded configuration is saved as:

```text
outputs/.../options_restart.yaml
```

This file is extremely useful when learning the codebase because it shows the
defaults metatrain inferred from the short tutorial YAML.

## Config Expansion

### `src/metatrain/utils/omegaconf.py`

This file contains the configuration expansion logic.

For the PET tutorial, the key function is:

```python
expand_dataset_config(...)
```

It turns the compact YAML into a full dataset config. For energy targets, it
adds default gradient sections for:

- `forces`
- `stress`

That is why the tutorial can omit `forces:` from `options-scratch.yaml` while
still training on forces.

In this ethanol example:

- `forces` is found in the extended XYZ file
- `stress` is not found

The expected log message is therefore:

```text
Forces found in section 'energy', we will use this gradient to train the model
No stress found in section 'energy'.
```

## Data Loading

### `src/metatrain/utils/data/get_dataset.py`

This chooses which dataset backend to use.

For the tutorial, `./ethanol_reduced_100.xyz` is neither a directory nor a zip
file, so metatrain uses the regular in-memory path:

```text
read_systems(...)
read_targets(...)
Dataset.from_dict(...)
```

Other paths exist for larger datasets:

- `.zip` files use `DiskDataset`
- directories use `MemmapDataset`

Those are relevant later for MPTrj-scale data, but not needed for the beginner
PET tutorial.

### `src/metatrain/utils/data/readers/readers.py`

This file selects the reader implementation. For `.xyz` and `.extxyz`, the
default reader is ASE:

```text
.xyz    -> ase
.extxyz -> ase
```

### `src/metatrain/utils/data/readers/ase.py`

This is the actual ASE-backed reader used by the tutorial.

Important pieces:

- `read_systems(...)` reads structures and turns ASE `Atoms` into metatomic
  `System` objects.
- `_read_energy_ase(...)` reads `Atoms.info["energy"]`.
- `_read_forces_ase(...)` reads `Atoms.arrays["forces"]`.
- `read_energy(...)` combines the energy target with force gradients.

Metatrain stores forces as gradients of the energy with respect to positions,
so `_read_forces_ase(...)` flips the sign internally.

This is a useful breakpoint location if you want to verify that the tutorial
dataset is being read correctly.

## PET Architecture Files

### `src/metatrain/pet/__init__.py`

This file exposes PET to the generic metatrain training machinery:

```python
__model__ = PET
__trainer__ = Trainer
```

When `architecture.name: pet` is read from YAML, metatrain imports this package
and retrieves these two classes.

### `src/metatrain/pet/documentation.py`

This file defines typed hyperparameter documentation and defaults for PET.

Important tutorial defaults include:

- `cutoff: 4.5`
- `batch_size: 16`
- `num_epochs: 1000`
- `learning_rate: 1e-4`
- `loss: "mse"`

The tutorial overrides some of these in `options-scratch.yaml`, for example
`num_epochs: 10` and `batch_size: 10`.

This file is also used to generate the online PET architecture documentation.

### `src/metatrain/pet/model.py`

This contains the PET neural network implementation.

The main class is:

```python
class PET(ModelInterface)
```

For the tutorial, the model constructor receives:

- PET model hyperparameters from `architecture.model`
- `DatasetInfo`, including atomic types and target metadata

The model sets up:

- cutoff and neighbor-list requirements
- PET transformer layers
- target output heads
- additive composition model support
- target scaling support
- export support for metatomic `AtomisticModel`

This is a good breakpoint location if you want to inspect the architecture
created from `options-scratch.yaml`.

### `src/metatrain/pet/trainer.py`

This contains the PET-specific training loop.

The main class is:

```python
class Trainer(TrainerInterface)
```

For the tutorial, `Trainer.train(...)`:

1. Selects CPU/GPU and dtype.
2. Moves the PET model to the selected device.
3. Computes composition weights for the energy baseline.
4. Computes scaling weights if `scale_targets` is enabled.
5. Builds data loaders.
6. Requests neighbor lists needed by PET.
7. Builds the loss function.
8. Runs the epoch loop.
9. Logs metrics.
10. Saves intermediate checkpoints.
11. Tracks the best model.

This is the main place to debug training-time behavior such as losses,
gradients, dataloader batches, and optimizer steps.

## Evaluation Flow

After training, the tutorial runs:

```bash
python -m metatrain eval model.pt eval-scratch.yaml
```

The main evaluation files are:

```text
src/metatrain/__main__.py
src/metatrain/cli/eval.py
examples/0-beginner/eval-scratch.yaml
```

`src/metatrain/cli/eval.py` loads the exported `.pt` model, reads the evaluation
dataset, computes requested outputs, accumulates metrics, and writes prediction
output such as:

```text
output.xyz
```

## Output Files

A normal tutorial run creates:

```text
outputs/YYYY-MM-DD/HH-MM-SS/
  indices/
    training.txt
    validation.txt
    test.txt
  model_0.ckpt
  model.ckpt
  model.pt
  options_restart.yaml
  train.csv
  train.log
```

The important distinction is:

- `model.ckpt`: training checkpoint, useful for restart/fine-tuning
- `model.pt`: exported metatomic model, useful for evaluation and MD engines
- `options_restart.yaml`: fully expanded training config
- `train.log`: human-readable log
- `train.csv`: structured metrics log

## Suggested Debugging Breakpoints

For understanding the tutorial end to end, start with these:

```text
src/metatrain/__main__.py
  main()

src/metatrain/cli/train.py
  train_model(...)

src/metatrain/utils/omegaconf.py
  expand_dataset_config(...)
  expand_loss_config(...)

src/metatrain/utils/data/get_dataset.py
  get_dataset(...)

src/metatrain/utils/data/readers/ase.py
  read_systems(...)
  read_energy(...)
  _read_forces_ase(...)

src/metatrain/pet/model.py
  PET.__init__(...)

src/metatrain/pet/trainer.py
  Trainer.train(...)
```

Use this VSCode style launch config:

```json
{
  "name": "Debug PET tutorial training",
  "type": "debugpy",
  "request": "launch",
  "module": "metatrain",
  "args": [
    "--debug",
    "train",
    "options-scratch.yaml"
  ],
  "cwd": "/home/ryoji/equivarient/metatrain/examples/0-beginner",
  "console": "integratedTerminal",
  "justMyCode": false
}
```

## How This Helps With MPTrj Later

The ethanol tutorial teaches the native contract metatrain expects:

- structures become metatomic `System` objects
- energy is a system-level scalar target
- forces are gradients of the energy target
- PET receives target metadata through `DatasetInfo`
- PET training is architecture-specific, but data loading and config parsing are
  shared infrastructure

For MPTrj, the main extra work will be replacing the small ASE-readable XYZ
dataset with a converted dataset that follows the same target contract. The
PET-side training path should remain the same once the dataset is readable.


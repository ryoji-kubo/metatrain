# Structure Transformer in Metatrain

This note documents the experimental `experimental.structure_transformer`
architecture that was added to this repository to run the local
`equiformer_v3` direct transformer through the PET/metatrain training pipeline.

The short version is:

- The PET trainer is reused almost as-is.
- The neural network architecture is swapped from PET to a copied/adapted
  structure transformer.
- The model predicts energy, force, and stress directly.
- The PET preprocessing/postprocessing stack is still used: composition baseline,
  target scaling, random rotational augmentation, losses, metrics, checkpointing,
  and export.

## Files Added

### Architecture Registration

`src/metatrain/experimental/structure_transformer/__init__.py`

This makes the architecture importable as:

```yaml
architecture:
  name: experimental.structure_transformer
```

It registers:

```python
__model__ = StructureTransformerModel
__trainer__ = Trainer
```

### Trainer Alias

`src/metatrain/experimental/structure_transformer/trainer.py`

This file intentionally does not implement a new trainer. It aliases the PET trainer:

```python
from metatrain.pet.trainer import Trainer
```

That is the main design choice. It means this transformer uses PET's existing
training path, including:

- composition baseline fitting,
- target scaling,
- rotational augmentation,
- target residualization/scaling in the collate function,
- PET loss aggregation,
- train/validation metric calculation,
- checkpointing,
- optional distributed training.

### Hyperparameter Schema

`src/metatrain/experimental/structure_transformer/documentation.py`

This defines the model hyperparameters used by the copied transformer:

- `embed_dim`
- `num_heads`
- `num_layers`
- dropout settings
- `position_representation`
- `use_rotary_embeddings`
- `atom_ordering`
- `include_cell_energy`
- `include_cell_stress`
- `regress_forces`
- `regress_stress`
- `direct_prediction`
- `symmetrize_stress`

The training hyperparameters are imported from PET:

```python
from metatrain.pet.documentation import TrainerHypers
```

So the structure transformer accepts the same training section as PET.

### Metatrain Wrapper

`src/metatrain/experimental/structure_transformer/model.py`

`StructureTransformerModel` is the adapter between metatrain's atomistic model
interface and the copied transformer.

It does four main jobs.

First, it creates the transformer:

```python
self.transformer = StructureTransformer(**transformer_hypers)
```

Second, it creates PET-style additive/scaling models:

```python
self.additive_models = torch.nn.ModuleList([composition_model])
self.scaler = Scaler(hypers={}, dataset_info=dataset_info)
```

These are required because the reused PET trainer expects the model to own
`additive_models` and `scaler`.

Third, it maps metatrain target names to raw transformer outputs:

| Metatrain target | Raw transformer output |
| --- | --- |
| `energy` | `energy` |
| `non_conservative_force` | `forces` |
| `non_conservative_stress` | `stress` |

Fourth, it converts between data formats:

- input: list of metatomic `System` objects,
- internal: `TransformerData(atomic_numbers, pos, batch, cell)`,
- output: metatensor `TensorMap`s with the same layouts as the configured targets.

The wrapper intentionally returns:

```python
def requested_neighbor_lists(self):
    return []
```

The copied transformer is a dense global sequence model, not a local-neighborhood
model, so it does not consume neighbor lists. The PET trainer still calls the normal
neighbor-list transform, but the requested list is empty.

### Transformer Core

`src/metatrain/experimental/structure_transformer/modules/transformer.py`

This is the copied/adapted architecture from:

```text
/home/ryoji/equivarient/equiformer_v3/experimental/models/transformer/transformer.py
```

The main class is:

```python
class StructureTransformer(nn.Module)
```

The implementation was made self-contained for metatrain:

- removed FairChem registry usage,
- removed PyG `to_dense_batch`,
- added local `_to_dense_batch`,
- added `TransformerData`,
- made a few TorchScript/export-friendly changes.

## Forward Path

### 1. Systems Become Dense Tokens

`StructureTransformerModel.forward(...)` receives:

```python
systems: list[System]
outputs: dict[str, ModelOutput]
```

It concatenates atom positions and atom types across the batch:

```python
positions = torch.cat([system.positions for system in systems], dim=0)
atomic_numbers = torch.cat([system.types for system in systems], dim=0)
batch = ...
cells = torch.stack([system.cell for system in systems], dim=0)
```

This becomes:

```python
TransformerData(
    atomic_numbers=atomic_numbers,
    pos=positions,
    batch=batch,
    cell=cells,
)
```

Inside `StructureTransformer`, `_to_dense_batch` turns the ragged atom list into
dense tensors:

```text
pos_dense:            (batch, max_atoms, 3)
atomic_numbers_dense: (batch, max_atoms)
atom_mask:            (batch, max_atoms)
```

### 2. Positions and Cell Are Encoded

The config currently uses:

```yaml
position_representation: fractional
center_positions: false
```

So Cartesian positions are converted to fractional coordinates using the cell.

The model encodes:

- atom number through an embedding table,
- atom position through an MLP,
- cell matrix through an MLP cell token,
- cell matrix again as a per-layer conditioning vector.

The token sequence is:

```text
[cell_token, atom_1, atom_2, ..., atom_N]
```

### 3. Transformer Blocks

Each block is a standard dense-attention transformer block:

- RMSNorm,
- multi-head self-attention,
- residual connection,
- RMSNorm,
- SwiGLU feed-forward network,
- residual connection.

This is global attention over all atoms in the structure, not local message passing.
That means the memory/time cost grows roughly like `max_atoms^2` inside each batch.

### 4. Direct Output Heads

After the final normalization:

```python
cell_features = tokens[:, 0, :]
atom_features = tokens[:, 1 : max_atoms + 1, :]
```

The model predicts:

```text
energy: atom_energy_head(atom_features).sum(...) + optional cell_energy_head
forces: force_head(atom_features)
stress: mean(atom_stress_head(atom_features)) + optional cell_stress_head
```

Important: force and stress are direct predictions. They are not obtained as
gradients of the energy.

The wrapper converts these raw tensors into metatensor layouts:

```text
energy:                  (n_systems, 1)
non_conservative_force:  (n_atoms, 3, 1)
non_conservative_stress: (n_systems, 3, 3, 1)
```

If `symmetrize_stress: true`, the wrapper symmetrizes the stress matrix before
returning it. The current config keeps:

```yaml
symmetrize_stress: false
```

to stay closer to the original direct transformer behavior.

## What Is Reused From PET

Because `trainer.py` aliases `metatrain.pet.trainer.Trainer`, the training loop is
the PET training loop.

The trainer does the following before optimization starts:

1. Builds the dataset and `DatasetInfo`.
2. Fits the `CompositionModel` on the training set.
3. Fits the `Scaler` on residualized targets.
4. Builds train/validation `DataLoader`s.
5. Builds collate functions that transform every batch.

For training batches, the collate function applies:

```text
atomic_basis_transform
random rotational augmentation
neighbor-list transform, no-op here because requested_neighbor_lists() is empty
remove additive composition baseline
remove target scales
```

For validation batches, it applies the same transforms except random augmentation.

So the transformer is trained on the same kind of preprocessed targets as PET:

- energy is composition-residualized and scaled,
- force is scaled,
- stress is scaled,
- per-structure targets such as stress are handled according to
  `per_structure_targets`.

During training, `StructureTransformerModel.forward(...)` returns predictions in the
training target space. During evaluation/export-style inference, the wrapper applies
the model scaler and additive composition contributions back to return physical-unit
outputs.

Inside the PET trainer's metric path, the trainer also explicitly reapplies scales
before accumulating train/validation metrics. This is why logged metrics are intended
to be in comparable physical units, even though the loss is computed in the scaled
training space.

## Main Config

`options-structure-transformer-mptrj-salex-direct.yaml`

This config trains on:

```yaml
training_set:
  systems:
    read_from: data
```

and validates on:

```yaml
validation_set:
  systems:
    read_from: data_salex_val_30k
```

It uses direct targets:

```yaml
energy:
  key: e
non_conservative_force:
  key: f
non_conservative_stress:
  key: s
```

The architecture section starts with an explicit `atomic_types` list. This is
important for MPtrj because otherwise metatrain infers atomic types by iterating over
the training and validation datasets. On the full MPtrj memmap this is slow. The
explicit list was computed from `data/a.bin` with `np.unique`, and it covers the
30k sAlex validation memmap as well.

The model size in the config is the larger Equiformer transformer-like setting:

```yaml
embed_dim: 768
num_heads: 12
num_layers: 12
```

For debugging, the VSCode launch config overrides these to a tiny model:

```yaml
embed_dim: 64
num_heads: 4
num_layers: 1
```

## TorchScript and Export Adjustments

Metatrain exports the final model through `AtomisticModel.save(...)`, which scripts
the module with TorchScript. A few small changes were needed so the copied transformer
could export:

- `_to_dense_batch` avoids dynamic `*values.shape[1:]` shape construction.
- attention masking uses `-1.0e30` instead of `torch.finfo(attn.dtype).min`.
- `torch.nonzero(batch == i_system)` is used without the unsupported `as_tuple`
  keyword.
- optional tensors such as `frac_pos_dense` and `rotary_pe_cplx` are annotated as
  `Optional[torch.Tensor]`.
- forward-path error messages avoid formatted strings that TorchScript cannot parse.
- the nonessential `num_params` convenience property was removed.

A tiny two-structure, one-epoch CPU smoke test passed through training, checkpointing,
and final `.pt` export.

## How To Start Single-GPU Training

From the repository root:

```bash
cd /home/ryoji/equivarient/metatrain
```

Activate the environment:

```bash
conda activate metatrain-pet
```

Run on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m metatrain train options-structure-transformer-mptrj-salex-direct.yaml \
  -o structure-transformer-mptrj-salex.pt
```

The config already has:

```yaml
device: cuda
architecture:
  training:
    distributed: false
```

so the command above uses one visible CUDA device.

For a first small GPU smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m metatrain train options-structure-transformer-mptrj-salex-direct.yaml \
  -o structure-transformer-smoke.pt \
  -r wandb.mode=disabled \
  -r training_set.indices=[0,1,2,3,4,5,6,7] \
  -r validation_set.indices=[0,1] \
  -r architecture.model.embed_dim=64 \
  -r architecture.model.num_heads=4 \
  -r architecture.model.num_layers=1 \
  -r architecture.training.num_epochs=1 \
  -r architecture.training.batch_size=2 \
  -r architecture.training.max_atoms_per_batch=128 \
  -r architecture.training.num_workers=0
```

If the full model is too large, the first knobs to reduce are:

```bash
-r architecture.training.max_atoms_per_batch=256
-r architecture.training.batch_size=8
```

For this transformer, `max_atoms_per_batch` is especially important because the
attention cost is dense in the number of atoms.

## How To Run Parallel Training With Torchrun

The reused PET trainer supports distributed training through DDP/NCCL. In this code,
distributed mode expects:

```yaml
architecture:
  training:
    distributed: true
```

and the device should still be:

```yaml
device: cuda
```

Do not use `device=multi-gpu` for this path. The PET trainer explicitly rejects
distributed training with the `multi-gpu` device setting; each torchrun process should
instead receive one CUDA device through local rank mapping.

Single-node, 4-GPU example:

```bash
cd /home/ryoji/equivarient/metatrain
conda activate metatrain-pet

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  -m metatrain train options-structure-transformer-mptrj-salex-direct.yaml \
  -o structure-transformer-mptrj-salex-ddp.pt \
  -r device=cuda \
  -r architecture.training.distributed=true \
  -r architecture.training.distributed_port=39591
```

For a small DDP smoke run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  -m metatrain train options-structure-transformer-mptrj-salex-direct.yaml \
  -o structure-transformer-ddp-smoke.pt \
  -r device=cuda \
  -r wandb.mode=disabled \
  -r architecture.training.distributed=true \
  -r training_set.indices=[0,1,2,3,4,5,6,7] \
  -r validation_set.indices=[0,1,2,3] \
  -r architecture.model.embed_dim=64 \
  -r architecture.model.num_heads=4 \
  -r architecture.model.num_layers=1 \
  -r architecture.training.num_epochs=1 \
  -r architecture.training.max_atoms_per_batch=128 \
  -r architecture.training.num_workers=0
```

Notes for DDP:

- `torchrun --standalone` provides the `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`,
  `RANK`, and `LOCAL_RANK` variables that metatrain checks.
- The trainer uses `torch.distributed.init_process_group(backend="nccl")`.
- Each process maps to a CUDA device using `LOCAL_RANK`.
- The trainer uses distributed samplers for train/validation data.
- Composition baseline fitting and scaler fitting reduce statistics across ranks.
- Metrics also reduce across ranks.
- `max_atoms_per_batch` is effectively per rank, so the total number of atoms processed
  per step grows with the number of ranks.

## Current Limitations and Things To Watch

- This implementation is an experimental adapter, not a validated reproduction of the
  original Equiformer transformer training run.
- The transformer is non-equivariant and predicts forces/stress directly.
- It does not use PET neighbor lists.
- Dense attention can become expensive for structures with many atoms.
- The stress convention should be checked carefully against the Equiformer validation
  path before treating numbers as final apples-to-apples results.
- Multi-GPU training is supported through the reused PET DDP path, but the local machine
  used for smoke tests had a CUDA driver/PyTorch mismatch, so the DDP command itself
  should be validated on the driver-matched GPU server.


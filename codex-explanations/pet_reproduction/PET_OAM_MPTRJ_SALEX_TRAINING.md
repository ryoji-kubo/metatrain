# PET-OAM-XL-like MPtrj Training with sAlex Validation

This note documents the full training configuration added in this repo:

```text
options-pet-oam-xl-mptrj-salex.yaml
```

It trains on the full prepared MPtrj memmap dataset and validates on the prepared
30k sAlex validation split.

## What The Config Uses

The training dataset is:

```text
/home/ryoji/equivarient/metatrain/data
```

This was prepared from:

```text
/home/ryoji/equivarient/equiformer_v3/dataset/mptrj/MPtrj_2022.9_full.json
```

The validation dataset is:

```text
/home/ryoji/equivarient/metatrain/data_salex_val_30k
```

This was converted from the EquiformerV3 sAlex split:

```text
/home/ryoji/equivarient/equiformer_v3/dataset/salex/val_30k/data.aselmdb
```

The verified dataset sizes are:

```text
MPtrj train:      1,580,395 structures, 49,295,660 atoms
sAlex val_30k:      30,000 structures,    308,371 atoms
```

## Why This Is XL-Based

I used `options-pet-oam-xl-reconstructed.yaml` as the main source because the
PET-OAM-XL checkpoint is newer and closer to this local `metatrain` checkout than
the PET-OAM-L checkpoint:

```text
pet-oam-xl-v1.0.0.ckpt
model checkpoint version:   10
trainer checkpoint version: 12
current local PET versions: 13 / 13
```

The config copies the recovered PET-OAM-XL model hyperparameters:

- 10.0 A cutoff
- adaptive neighbor target of 40
- Bump cutoff function
- `d_pet: 640`
- `d_node: 2560`
- 5 GNN layers
- 3 attention layers per GNN layer
- RMSNorm, SwiGLU, PreLN
- no ZBL and no long-range Ewald branch

## Targets

Both train and validation use the same memmap keys:

```yaml
energy:
  key: e
  forces:
    key: f
  stress:
    key: s
non_conservative_force:
  key: f
non_conservative_stress:
  key: s
```

This means PET is trained on:

- energy
- conservative forces from the energy position gradient
- conservative stress from the energy strain gradient
- a direct non-conservative force head
- a direct non-conservative stress head

The released PET-OAM checkpoints used the older plural name
`non_conservative_forces`. For this new training config I used the current
metatrain-style singular target name `non_conservative_force`.

## Stress Instead Of Virial

The reconstructed checkpoints used old trainer language around `virial`, but this
config uses `stress`:

```yaml
loss:
  energy:
    stress:
      type: huber
```

This is intentional. `MemmapDataset` does not support virial targets directly, but
it does support stress targets. In the loss expander, `stress` maps to the same
energy `strain` gradient used for stress/virial-like training.

## Batch Packing

The released PET-OAM-XL checkpoint stored:

```yaml
batch_size: 4
batch_atom_bounds: [null, 400]
```

For the local full memmap training config I instead used:

```yaml
batch_size: 4
batch_atom_bounds: [null, null]
max_atoms_per_batch: 512
min_atoms_per_batch: 0
```

This uses metatrain's current `MemmapDataset`-only atom-count batch sampler. The
prepared datasets have these maximum per-structure atom counts:

```text
MPtrj max atoms:  444
sAlex max atoms: 128
```

So `max_atoms_per_batch: 512` should include every structure in both prepared
datasets. Lower this value if you hit GPU out-of-memory. If you lower it below
444, the sampler will skip oversized MPtrj structures and log how many it skips.

## Running Training

Activate the environment and run from the repo root:

```bash
cd /home/ryoji/equivarient/metatrain
conda activate metatrain-pet

python -m metatrain train options-pet-oam-xl-mptrj-salex.yaml \
  -o pet-oam-xl-mptrj-salex.pt
```

This writes logs and intermediate checkpoints under:

```text
outputs/YYYY-MM-DD/HH-MM-SS/
```

The final exported model is written to:

```text
pet-oam-xl-mptrj-salex.pt
```

and the final training checkpoint is written to:

```text
pet-oam-xl-mptrj-salex.ckpt
```

Both final files are also copied into the timestamped `outputs/...` directory.

To restart from the most recent checkpoint:

```bash
python -m metatrain train options-pet-oam-xl-mptrj-salex.yaml \
  --restart auto \
  -o pet-oam-xl-mptrj-salex.pt
```

## VSCode Debugging

Use the module entry point rather than the `mtt` shell command. A VSCode launch
configuration can call:

```text
module: metatrain
args: train options-pet-oam-xl-mptrj-salex.yaml -o pet-oam-xl-mptrj-salex-debug.pt
cwd: /home/ryoji/equivarient/metatrain
python: /home/ryoji/miniconda3/envs/metatrain-pet/bin/python
```

For debugging, temporarily reduce the workload with command-line overrides, for
example:

```bash
python -m metatrain train options-pet-oam-xl-mptrj-salex.yaml \
  -o pet-oam-xl-mptrj-salex-debug.pt \
  -r training_set.indices='[0,1,2,3,4,5,6,7]' \
  -r validation_set.indices='[0,1,2,3]' \
  -r architecture.training.max_atoms_per_batch=128
```

## Validation Performed

I validated the config without starting training by running it through the same
metatrain option expansion path used before training. It passed and expanded the
energy loss gradients to:

```text
positions
strain
```

I also instantiated the datasets from the config:

```text
train_len 1580395 train_atoms0 28
val_len 30000 val_atoms0 4
targets ['energy', 'non_conservative_force', 'non_conservative_stress']
val_targets ['energy', 'non_conservative_force', 'non_conservative_stress']
```

## Current Runtime Caveat

The `metatrain-pet` environment on this machine currently reports:

```text
torch 2.12.1+cu130
torch_cuda 13.0
cuda_available False
device_count 4
```

and PyTorch warns that the NVIDIA driver is too old for that CUDA build. The machine
has four NVIDIA L40S GPUs, but this conda environment will need a compatible
PyTorch/driver pairing before a GPU training run will start cleanly.

## Remaining Ambiguities

This config is a practical PET-OAM-XL-like MPtrj training config, not a proven exact
reproduction of the released PET-OAM-XL checkpoint.

The unrecovered pieces are:

- the private PET-OMat-XL adaptive checkpoint recorded in the released PET-OAM-XL
  checkpoint
- the exact original upstream dataset revisions and filtering steps
- the complete staged recipe, which appears to involve OMat pretraining followed by
  OAM fine-tuning rather than MPtrj-only training from random initialization

The config leaves `finetune.read_from: null`. If you obtain the correct PET-OMat-XL
source checkpoint, set that field to the `.ckpt` path before launching the run.

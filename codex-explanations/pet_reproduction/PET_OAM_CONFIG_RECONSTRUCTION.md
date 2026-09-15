# PET-OAM Config Reconstruction Notes

This note explains how the reconstructed PET-OAM config files were produced:

- `options-pet-oam-l-reconstructed.yaml`
- `options-pet-oam-xl-reconstructed.yaml`

The goal was not to invent a fresh PET training recipe, but to recover as much as
possible from the released UPET checkpoints and express it in the current
`metatrain` YAML format.

## Source Checkpoints

The checkpoints came from the UPET Hugging Face repository:

```text
lab-cosmo/upet
```

The two PET-OAM checkpoints inspected were:

```text
models/pet-oam-l-v0.1.0.ckpt
models/pet-oam-xl-v1.0.0.ckpt
```

They are PyTorch checkpoints, not exported TorchScript `.pt` models. For inspection,
they must be loaded after importing `metatomic.torch` and `metatensor.torch`, because
the checkpoint contains serialized metatomic custom classes such as
`ModelMetadata`.

The basic inspection pattern was:

```python
from huggingface_hub import hf_hub_download
import metatomic.torch
import metatensor.torch
import torch
from metatrain.utils.io import model_from_checkpoint

path = hf_hub_download(
    repo_id="lab-cosmo/upet",
    filename="pet-oam-xl-v1.0.0.ckpt",
    subfolder="models",
)

ckpt = torch.load(path, map_location="cpu", weights_only=False)

print(ckpt.keys())
print(ckpt["model_data"]["model_hypers"])
print(ckpt["model_data"]["dataset_info"])
print(ckpt["train_hypers"])

model = model_from_checkpoint(ckpt, context="export")
print(model.hypers)
```

The final `model.hypers` was important because `model_from_checkpoint` applies
metatrain's checkpoint upgrade logic. This gives the current-schema hyperparameters
that this checkout of `metatrain` will actually instantiate.

## What Was Recovered

The checkpoints contain enough information to recover:

- architecture name, `pet`
- model checkpoint version
- trainer checkpoint version
- PET model hyperparameters
- dataset target schema
- atomic types seen during training
- trainer hyperparameters
- loss settings
- optimizer and scheduler state
- model weights
- best/current model state
- in the XL case, the original fine-tuning source path

They do not contain enough information to recover:

- the original dataset files
- exact raw data keys
- filtering and deduplication rules
- exact train/validation/test split files
- complete staged pretraining recipe
- exact code commit or dependency lockfile
- the private source checkpoint used for PET-OAM-XL fine-tuning

So the reconstructed YAML files are best understood as **current-schema configs that
match the released checkpoints as closely as the checkpoint data allows**.

## PET-OAM-L Findings

Checkpoint:

```text
models/pet-oam-l-v0.1.0.ckpt
```

Raw checkpoint versions:

```text
model_ckpt_version: 3
trainer_ckpt_version: 1
```

This is an older checkpoint relative to the current repo. The raw model hypers were
sparse and required the metatrain upgrader to infer current-schema fields.

After upgrade, the recovered model hypers were:

```yaml
cutoff: 4.5
num_neighbors_adaptive: null
adaptive_cutoff_method: grid
cutoff_function: Cosine
cutoff_width: 0.2
d_pet: 512
d_head: 512
d_node: 2048
d_feedforward: 1024
num_heads: 8
num_attention_layers: 2
num_gnn_layers: 3
normalization: RMSNorm
activation: SwiGLU
attention_temperature: 1.0
transformer_type: PreLN
featurizer_type: feedforward
zbl: false
long_range:
  enable: false
  use_ewald: false
  smearing: 1.4
  kspace_resolution: 1.33
  interpolation_nodes: 5
```

The trainer hypers included:

```yaml
batch_size: 16
num_epochs: 5
learning_rate: 1.0e-4
distributed: true
checkpoint_interval: 1
log_interval: 1
grad_clip_norm: 1.0
best_model_metric: rmse_prod
scale_targets: true
```

The L checkpoint also stored fixed target scaling weights:

```yaml
fixed_scaling_weights:
  energy: 0.7483054
  non_conservative_forces: 2.8356092
  non_conservative_stress: 0.058322243
```

It also stored many fixed per-element energy baselines under the old name
`fixed_composition_weights`. In the current repo this corresponds to
`atomic_baseline`, so the reconstructed L config maps those values into:

```yaml
atomic_baseline:
  energy:
    ...
```

The stored target layout showed three outputs:

```text
energy
non_conservative_forces
non_conservative_stress
```

The `energy` target also had gradients:

```text
positions
strain
```

That means the L checkpoint trained on:

- energy
- conservative forces from the energy gradient
- conservative stress/virial from the strain gradient
- direct non-conservative force head
- direct non-conservative stress head

The old trainer loss used old naming such as `forces` and `virial`. In the
reconstructed config, these were expressed with the current loss shorthand:

```yaml
loss:
  energy:
    type: huber
    delta: 0.015
    forces:
      type: huber
      delta: 0.04
    virial:
      type: huber
      delta: 0.03
  non_conservative_forces:
    type: huber
    weight: 0.01
    delta: 0.01
  non_conservative_stress:
    type: huber
    weight: 0.01
    delta: 0.004
```

One ambiguity: the checkpoint layout says the energy has a `strain` gradient, but
the old loss block called the corresponding term `virial`. For normal file-based
datasets either `stress` or `virial` can create a strain gradient, but for metatrain
memmap datasets only `stress` is supported.

## PET-OAM-XL Findings

Checkpoint:

```text
models/pet-oam-xl-v1.0.0.ckpt
```

Raw checkpoint versions:

```text
model_ckpt_version: 10
trainer_ckpt_version: 12
```

This checkpoint is much closer to the current repo, whose PET model/trainer versions
are currently `13`.

After upgrade, the recovered model hypers were:

```yaml
cutoff: 10.0
num_neighbors_adaptive: 40
adaptive_cutoff_method: grid
cutoff_function: Bump
cutoff_width: 0.5
d_pet: 640
d_head: 640
d_node: 2560
d_feedforward: 1280
num_heads: 8
num_attention_layers: 3
num_gnn_layers: 5
normalization: RMSNorm
activation: SwiGLU
attention_temperature: 1.0
transformer_type: PreLN
featurizer_type: feedforward
zbl: false
long_range:
  enable: false
  use_ewald: false
  smearing: 1.4
  kspace_resolution: 1.33
  interpolation_nodes: 5
```

The trainer hypers were already mostly current-schema:

```yaml
batch_size: 4
batch_atom_bounds: [null, 400]
num_epochs: 1
warmup_fraction: 0.1
learning_rate: 5.0e-6
distributed: true
checkpoint_interval: 1
log_interval: 1
best_model_metric: mae_prod
grad_clip_norm: 1.0
scale_targets: true
per_structure_targets:
  - non_conservative_stress
atomic_baseline: {}
fixed_scaling_weights: {}
```

The most important XL-specific finding was the stored fine-tuning source:

```text
/capstor/scratch/cscs/fbigi/omat_new_style/xl/pet-omat-xl-adaptive.ckpt
```

This strongly indicates that the released PET-OAM-XL checkpoint was produced by
full fine-tuning from a PET-OMat-XL adaptive checkpoint. The reconstructed YAML
therefore includes:

```yaml
finetune:
  method: full
  read_from: /path/to/pet-omat-xl-adaptive.ckpt
  config: {}
  inherit_heads: {}
```

The path in the released checkpoint is private and not available on this machine. I
did not replace it with a public checkpoint because I cannot prove any public
PET-OMat-XL file is bit-identical to that private source checkpoint.

The XL target layout matched the L layout:

```text
energy
non_conservative_forces
non_conservative_stress
```

with energy gradients:

```text
positions
strain
```

So the XL config also includes energy, conservative forces/stress, and direct
non-conservative force/stress heads.

The recovered XL loss settings were:

```yaml
loss:
  energy:
    type: huber
    weight: 1.0
    reduction: mean
    delta: 0.01
    forces:
      type: huber
      weight: 1.0
      reduction: mean
      delta: 0.05
    virial:
      type: huber
      weight: 1.0
      reduction: mean
      delta: 0.05
  non_conservative_forces:
    type: huber
    weight: 0.01
    reduction: mean
    delta: 0.05
  non_conservative_stress:
    type: huber
    weight: 0.001
    reduction: mean
    delta: 0.005
```

## Output Naming

The released checkpoints expose the older plural output name:

```text
non_conservative_forces
```

Current metatrain examples generally use the singular YAML key:

```text
non_conservative_force
```

metatrain still accepts the plural name, but it may warn that the quantity name is
deprecated. For the reconstructed checkpoint-matching configs, I used the plural
form in places where preserving the released checkpoint output names seemed more
important. For new training from scratch on MPtrj, I recommend the singular form:

```yaml
non_conservative_force:
  key: f
  quantity: force
  unit: eV/A
  sample_kind: atom
  type:
    cartesian:
      rank: 1
```

## Why the Configs Have Placeholders

The checkpoints store the target schema but not the original raw data files or raw
field names. For example, they reveal that the model was trained on energy, forces,
and stress-like targets, but they do not say:

- whether the raw file used `energy`, `e`, `uncorrected_total_energy`, or another key
- where the training/validation/test files were located
- whether filters were applied before writing the dataset
- what exact split was used

Therefore, the reconstructed YAML files contain placeholders such as:

```yaml
read_from: /path/to/pet-oam-xl-train.extxyz
key: energy
key: forces
key: non_conservative_forces
```

These must be replaced with the actual prepared dataset path and keys.

For the MPtrj memmap converter in this repo, the relevant keys are:

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

## Practical Interpretation

The reconstructed configs are useful for three related but different goals:

1. **Understanding the released checkpoints**

   The files show what architecture/trainer setup the released PET-OAM checkpoints
   encode.

2. **Fine-tuning from a released checkpoint**

   Use the released `.ckpt` directly as `finetune.read_from`, then adapt the dataset
   section to your target data.

3. **Training a PET-OAM-like model on MPtrj**

   Use the recovered PET architecture and loss style, but prepare MPtrj with the
   metatrain-native memmap converter and use the MPtrj-specific keys `e`, `f`, and
   `s`.

The reconstructed configs do **not** prove the exact public PET-OAM recipe is
reproducible from public files alone. In particular, PET-OAM-XL points to a private
PET-OMat-XL adaptive checkpoint, and the UPET docs describe PET-OAM as
`OMat -> sAlex+MPtrj`, not simply MPtrj from scratch.

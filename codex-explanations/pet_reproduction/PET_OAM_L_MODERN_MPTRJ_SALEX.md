# L-Sized Modern PET-OAM-like MPtrj Training Config

This note documents:

```text
options-pet-oam-l-modern-mptrj-salex.yaml
```

It is a smaller sibling of:

```text
options-pet-oam-xl-mptrj-salex.yaml
```

Both configs train on:

```text
data
```

and validate on:

```text
data_salex_val_30k
```

## Design Choice

This is not a verbatim copy of the old PET-OAM-L checkpoint config. It keeps the
same current-repo structure as the XL MPtrj/sAlex config:

- singular `non_conservative_force`
- `stress` rather than `virial` for the memmap energy strain gradient
- `max_atoms_per_batch` batching for `MemmapDataset`
- no copied fixed target scales or fixed composition baseline

The model is shrunk toward PET-OAM-L by using:

```yaml
d_pet: 512
d_head: 512
d_node: 2048
d_feedforward: 1024
num_attention_layers: 2
num_gnn_layers: 3
```

It keeps the newer XL-style adaptive neighborhood:

```yaml
cutoff: 10.0
num_neighbors_adaptive: 40
adaptive_cutoff_method: grid
cutoff_function: Bump
cutoff_width: 0.5
```

To move closer to the old released PET-OAM-L local-neighborhood setup, change those
fields to:

```yaml
cutoff: 4.5
num_neighbors_adaptive: null
cutoff_function: Cosine
cutoff_width: 0.2
```

## Run

```bash
cd /home/ryoji/equivarient/metatrain
conda activate metatrain-pet

python -m metatrain train options-pet-oam-l-modern-mptrj-salex.yaml \
  -o pet-oam-l-modern-mptrj-salex.pt
```

For a tiny debugger run:

```bash
python -m metatrain train options-pet-oam-l-modern-mptrj-salex.yaml \
  -o pet-oam-l-modern-mptrj-salex-debug.pt \
  -r training_set.indices='[0,1,2,3,4,5,6,7]' \
  -r validation_set.indices='[0,1,2,3]' \
  -r architecture.training.max_atoms_per_batch=128
```

## Validation Performed

I validated the config through metatrain's option expansion path and instantiated
both datasets. The relevant checks were:

```text
validated options-pet-oam-l-modern-mptrj-salex.yaml
cutoff 10.0
d_pet 512
num_gnn_layers 3
energy_gradients ['positions', 'strain']
train_len 1580395
val_len 30000
targets ['energy', 'non_conservative_force', 'non_conservative_stress']
```

The same CUDA runtime caveat from the XL config still applies: the current
`metatrain-pet` environment reports a PyTorch CUDA build newer than the installed
driver supports, so GPU training will need a compatible PyTorch/driver pairing.

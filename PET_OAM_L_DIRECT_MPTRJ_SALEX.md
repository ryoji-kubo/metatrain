# Direct-Only L-Sized PET-OAM-like MPtrj/sAlex Config

This note documents:

```text
options-pet-oam-l-modern-mptrj-salex-direct.yaml
```

This config is the direct force/stress counterpart of:

```text
options-pet-oam-l-modern-mptrj-salex.yaml
```

## What Direct-Only Means Here

The config still trains scalar energy directly:

```yaml
energy:
  key: e
  quantity: energy
  unit: eV
  forces: false
  stress: false
```

But it does not request energy gradients. Therefore:

- force is not trained as `-dE/dR`
- stress is not trained as an energy strain gradient
- force is trained only through `non_conservative_force`
- stress is trained only through `non_conservative_stress`

The direct targets are:

```yaml
non_conservative_force:
  key: f
  quantity: force
  unit: eV/A
  sample_kind: atom
  type:
    cartesian:
      rank: 1
non_conservative_stress:
  key: s
  quantity: pressure
  unit: eV/A^3
  sample_kind: system
  type:
    cartesian:
      rank: 2
```

## Run

```bash
cd /home/ryoji/equivarient/metatrain
conda activate metatrain-pet

python -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-modern-mptrj-salex-direct.pt
```

For a tiny debugger run:

```bash
python -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-modern-mptrj-salex-direct-debug.pt \
  -r device=cpu \
  -r training_set.indices='[0,1,2,3,4,5,6,7]' \
  -r validation_set.indices='[0,1,2,3]' \
  -r architecture.training.num_epochs=1 \
  -r architecture.training.max_atoms_per_batch=128
```

## Validation Performed

I validated the config through metatrain's option expansion and dataset loading:

```text
validated options-pet-oam-l-modern-mptrj-salex-direct.yaml
energy_forces False
energy_stress False
loss_targets ['energy', 'non_conservative_force', 'non_conservative_stress']
energy_gradients []
train_len 1580395
val_len 30000
targets ['energy', 'non_conservative_force', 'non_conservative_stress']
```

The empty `energy_gradients []` line is the key check: it confirms that this config
does not train conservative gradient forces or conservative gradient stress.

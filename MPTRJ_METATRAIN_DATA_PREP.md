# MPtrj Data Preparation for metatrain PET

This note records the metatrain-native path for preparing MPtrj data for PET or
PET-OAM-like training. It follows the `MemmapDataset` layout shown in
`examples/0-beginner/01-data_preparation.py`.

## Recommendation

Use metatrain's `MemmapDataset` format directly. Do not point metatrain at the
Equiformer/FairChem `ase_db`/`.aselmdb` dataset unless you add a custom reader.

The Equiformer-prepared data is useful as a reference, but it depends on
FairChem's `AseDBDataset` and `ase_db_backends`. metatrain's built-in ASE reader
uses `ase.io.read`, and its large-dataset path expects either a `.zip`
`DiskDataset` or a memmap directory.

## Converter

The converter is:

```bash
python scripts/prepare_mptrj_memmap.py \
  --input /home/ryoji/equivarient/equiformer_v3/dataset/mptrj/MPtrj_2022.9_full.json \
  --output data \
  --progress-every 10000
```

For a small smoke test:

```bash
python scripts/prepare_mptrj_memmap.py \
  --output /tmp/metatrain-mptrj-smoke \
  --max-structures 1000 \
  --force
```

The output directory contains:

```text
ns.npy
na.npy
a.bin
x.bin
c.bin
e.bin
f.bin
s.bin
metadata.json
```

The memmap keys are:

- `e`: `uncorrected_total_energy`, in eV
- `f`: `force`, in eV/A
- `s`: `stress`, converted from MPtrj kBar to eV/A^3 using the same ASE sign
  convention as the Equiformer converter: `stress * (-0.1 * ase.units.GPa)`

## metatrain YAML Snippet

Use `stress`, not `virial`, with `MemmapDataset`. The metatrain memmap loader does
not support virial targets directly.

```yaml
training_set:
  systems:
    read_from: /path/to/mptrj_memmap
    length_unit: angstrom
  targets:
    energy:
      key: e
      quantity: energy
      unit: eV
      forces:
        key: f
      stress:
        key: s
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

## Caveats

- This prepares MPtrj for PET training, but it does not reproduce the full public
  PET-OAM recipe by itself. The released PET-OAM-XL checkpoint records a full
  fine-tuning stage from a private PET-OMat-XL adaptive checkpoint.
- Equiformer normalizers, tensor decomposition, and element-reference transforms are
  not copied. metatrain PET has its own `atomic_baseline` and `scale_targets`
  machinery.
- The current metatrain examples use the singular target name
  `non_conservative_force`. Some released UPET checkpoints expose the older plural
  name `non_conservative_forces`; use the singular form for new training unless you
  specifically need to preserve old output names.

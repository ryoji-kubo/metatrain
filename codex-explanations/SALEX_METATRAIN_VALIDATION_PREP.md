# Preparing sAlex `val_30k` for metatrain Validation

This note records how to prepare the same sAlex validation split used by the
EquiformerV3 MPtrj experiments so it can be used as a `validation_set` in this
`metatrain` repository.

## Local State

The EquiformerV3 README says its MPtrj runs train on MPtrj and validate on a
30k subsplit of the sAlex validation set. On this machine that split already
exists here:

```bash
/home/ryoji/equivarient/equiformer_v3/dataset/salex/val_30k/data.aselmdb
```

I converted it to metatrain's `MemmapDataset` layout here:

```bash
/home/ryoji/equivarient/metatrain/data_salex_val_30k
```

The converted split contains:

- 30,000 structures
- 308,371 atoms
- `energy`, `forces`, and `stress` labels

## If `val_30k` Needs To Be Recreated

If the Equiformer split is missing, recreate it from the downloaded sAlex
validation directory using the command documented in
`/home/ryoji/equivarient/equiformer_v3/README.md`:

```bash
cd /home/ryoji/equivarient/equiformer_v3

/home/ryoji/miniconda3/envs/equiformer_v3/bin/python \
  experimental/datasets/mptrj_create_subset_aselmdb.py \
  --source-dir dataset/salex/val \
  --target-dir dataset/salex/val_30k \
  --seed 0 \
  --num-samples 30000
```

For metatrain, the FairChem graph metadata files are not used. Metatrain reads
positions, cells, atomic numbers, and target arrays from its own memmap files.

## Convert ASE-LMDB To Metatrain Memmap

Use the converter added in this repo:

```bash
cd /home/ryoji/equivarient/metatrain

# use the equiformer_v3 env for ase_db_backends
/home/ryoji/miniconda3/envs/equiformer_v3/bin/python \
  scripts/prepare_aselmdb_memmap.py \
  --input /home/ryoji/equivarient/equiformer_v3/dataset/salex/val_30k \
  --output data_salex_val_30k \
  --progress-every 5000 \
  --force
```

I used the `equiformer_v3` environment because it already has the ASE-LMDB backend
needed to read `.aselmdb` files. You can run the same script from
`metatrain-pet` if that environment has `ase` and `ase_db_backends` installed.
The training environment only needs the converted memmap directory.

The converter writes:

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

This matches the memmap format described in
`examples/0-beginner/01-data_preparation.py` and implemented by
`metatrain.utils.data.dataset.MemmapDataset`.

## Stress Convention

The original MPtrj JSON stress needed a unit/sign conversion before writing
`s.bin`. The sAlex `.aselmdb` rows already store ASE calculator results, so this
converter copies `stress` directly and assumes ASE stress in `eV/A^3` with ASE's
sign convention.

## Validation YAML Snippet

For a PET-OAM-like run that uses conservative energy gradients and direct
non-conservative force/stress targets, add a validation set like this:

```yaml
validation_set:
  systems:
    read_from: data_salex_val_30k
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

If your training config does not include direct non-conservative heads, omit the
`non_conservative_force` and `non_conservative_stress` targets. The target names
must match the names in the training config.

## Verification

I verified a five-structure smoke conversion and the full converted dataset with
`metatrain-pet`. The full dataset loaded successfully:

```text
len 30000
atoms0 4
total_atoms 308371
system_types [57, 42, 42, 30]
```

## Remaining Ambiguity

I did not re-download sAlex from Hugging Face or verify the upstream dataset
revision/checksum. Since the Equiformer `val_30k` split already existed locally,
I preserved that local split directly. If you need exact provenance from raw
download to split, the missing piece is the precise Hugging Face dataset revision
used to create `/home/ryoji/equivarient/equiformer_v3/dataset/salex/val`.

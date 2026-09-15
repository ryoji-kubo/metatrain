# Training on a 160k MPtrj Subset

This note documents the fast subset path for starting MPtrj training without
scanning or training on the full 1,580,395-structure memmap dataset.

## Recommendation

Use an indices file against the existing prepared memmap dataset:

```text
data
```

This avoids copying structures into a second memmap directory. The subset is just a
plain text file, and metatrain will wrap the existing `MemmapDataset` in a
`torch.utils.data.Subset`.

I generated this 160k subset:

```text
indices/mptrj_160k_seed0.txt
```

It contains 160,000 randomly sampled structure indices from the full MPtrj memmap,
using seed 0. The indices are sorted after sampling so first-pass reads for baseline
and scaling statistics are more sequential.

## Regenerate The Indices

The helper script is:

```text
scripts/create_memmap_subset_indices.py
```

To regenerate the same subset:

```bash
cd /home/ryoji/equivarient/metatrain

python scripts/create_memmap_subset_indices.py \
  --dataset data \
  --num-samples 160000 \
  --seed 0 \
  --output indices/mptrj_160k_seed0.txt
```

To create a different random 160k subset, change `--seed`.

## Run The Direct-Only L-Sized Config On 160k

```bash
cd /home/ryoji/equivarient/metatrain
conda activate metatrain-pet

python -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-modern-mptrj160k-salex-direct.pt \
  -r training_set.indices=indices/mptrj_160k_seed0.txt
```

This trains on:

```text
MPtrj train subset: 160,000 structures
sAlex validation:    30,000 structures
```

## Run The Gradient+Direct L-Sized Config On 160k

```bash
python -m metatrain train options-pet-oam-l-modern-mptrj-salex.yaml \
  -o pet-oam-l-modern-mptrj160k-salex.pt \
  -r training_set.indices=indices/mptrj_160k_seed0.txt
```

The same indices file can be used with the XL config too, though XL will be much
more expensive per step.

## Why This Helps

With `atomic_baseline: {}` and `fixed_scaling_weights: {}`, metatrain fits the
composition baseline and target scales before training. On the full MPtrj memmap,
that means scanning all 1.58M structures. With `training_set.indices`, those fitting
passes scan only the 160k subset.

Likewise, if `architecture.atomic_types` is not provided, metatrain infers atomic
types from the training and validation datasets. With this subset override, the
training side of that inference uses 160k structures rather than the full MPtrj
dataset.

For even faster startup, you can additionally hard-code `architecture.atomic_types`
or reuse fixed baseline/scaling values. I did not do that here because fitting them
on the 160k subset is a cleaner first experiment.

## Materialized Subset Alternative

If random-access I/O becomes a bottleneck later, we can materialize a new physical
memmap directory containing only the 160k selected structures. That is more work and
duplicates data, so I would start with the indices file first.

## Validation Performed

I verified that metatrain can load the indices file and that it selects 160,000
structures:

```text
n 160000
min 4
max 1580375
base_len 1580395
subset_len 160000
targets ['energy', 'non_conservative_force', 'non_conservative_stress']
```

I also validated the direct-only config plus the 160k override through metatrain's
option expansion path:

```text
validated override
training_indices indices/mptrj_160k_seed0.txt
energy_gradients []
```

The empty `energy_gradients []` applies to the direct-only config and confirms that
it remains direct force/stress only.

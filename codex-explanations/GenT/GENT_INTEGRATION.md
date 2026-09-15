# GenT integration into metatrain

Date: 2026-09-14

The architecture is available as `experimental.gent`. Its neural network lives in
`src/gent_core`, separate from `src/structure_transformer_core`. The metatrain
adapter lives in `src/metatrain/experimental/gent` and reuses PET's trainer.

This is an initial atomistic adaptation of the supplied Generalized Transformer
layers, with working training, restart, and export. It is not a benchmarked GenT
potential or a claim of accuracy parity with PET/Structure Transformer.

## Starting point and implementation process

The requested `src/structure_transformer_core/gent_layers.py` was not present in
this checkout. The reference was instead
[`gent_layers_original.py`](../../src/structure_transformer_core/gent_layers_original.py),
credited to Shao Yanming and Xavier Bresson, dated Sep 10, 2026. That file provides
the attention blocks, but no atomistic encoder, prediction heads, training adapter,
or runnable model. It has been left unchanged.

I inspected that reference, Structure Transformer's extracted-core design, its
metatrain adapter, and the PET trainer and configuration. The implementation steps
were:

1. Adapt the reference layers into an independent PyTorch package. Retain their
   equations and parameter names so their numerical outputs can be compared using
   identical weights.
2. Define a tensor-only atomistic input contract. Encode periodic neighbor geometry
   into dense edge states and add energy, force, and stress readouts.
3. Add an independent metatrain adapter, using the Structure Transformer adapter's
   interface conventions and PET's composition model, scaler, and trainer.
4. Add hyperparameter discovery, a runnable MPtrj/sAlex configuration, tests, and a
   dedicated `tox -e gent-tests` environment.
5. Validate reference parity, physical symmetries, gradients, serialization, and a
   small CLI training/restart/export run before attempting any dataset-scale job.

Existing Structure Transformer/PET implementations, options, and in-progress edits
were not modified. GenT is not a mode inside Structure Transformer.

## Code boundary

| Location | Responsibility |
| --- | --- |
| [`src/gent_core/gent_layers.py`](../../src/gent_core/gent_layers.py) | RMSNorm, gated SiLU MLP, optional RoPE helpers, the three attention paths, GenT block |
| [`src/gent_core/model.py`](../../src/gent_core/model.py) | `GenTData`, dense batching, species/radial encoders, periodic image aggregation, raw tensor readouts |
| [`src/metatrain/experimental/gent/model.py`](../../src/metatrain/experimental/gent/model.py) | `System`/neighbor-list extraction, target validation, `TensorMap` conversion, selected atoms, composition/scaling, checkpoints, export |
| [`src/metatrain/experimental/gent/documentation.py`](../../src/metatrain/experimental/gent/documentation.py) | Model defaults and PET trainer hyperparameters |
| [`src/metatrain/experimental/gent/trainer.py`](../../src/metatrain/experimental/gent/trainer.py) | PET trainer re-export |

The core imports only PyTorch and the Python standard library. It does not import
metatrain, metatomic, metatensor, Structure Transformer, FairChem, or PyG. To use it
in another repository, copy/install `gent_core` and implement the `GenTData`
conversion plus any output scaling required there. Checkpoints for the neural
network alone can use `model.transformer.state_dict()`.

## What remains faithful to the supplied layers

Each GenT block has node states `x[B,N,D]` and dense edge states `e[B,N,N,D]`.

- Node self-attention attends over the nodes of each structure.
- Edge-to-node attention uses node `i` to query its edge states `e[i,j]`.
- The node update multiplies separately projected, normalized self-attention and
  edge-to-node outputs, followed by a residual and gated SiLU MLP.
- Node-to-edge attention updates each edge by attending to its two endpoint nodes.
  An edge residual and gated SiLU MLP follow.
- Q/K RMS normalization and the reference's bias choices are retained.

This differs from adding an adjacency bias to a regular Transformer: edge states
are trainable hidden representations updated throughout the network.

The standalone block supports optional complex RoPE phases. The atomistic model
passes `None`: integer sequence positions depend on arbitrary atom ordering and
would break permutation equivariance. This is a deliberate change from the
reference's default RoPE path, not a periodic-coordinate RoPE implementation.
Parity tests compare both supplied sequence phases and identity phases (equivalent
to no RoPE), using the original weights and equations on valid atoms/edges.

Other deliberate numerical changes:

- Padding is zeroed before attention and after residual updates. Reference node
  projection biases could otherwise populate padded nodes.
- Fully masked softmax rows are made finite *before* softmax. Applying
  `nan_to_num` only after softmax can leave NaNs in backward computations.
- RMSNorm retains float64 precision; lower precision normalization accumulates in
  float32.
- Self-attention uses explicit matrix multiplication and masked softmax, supporting
  second derivatives without depending on a fused attention kernel.
- RoPE handles odd head widths by leaving the final channel unchanged. Atomistic
  models have no RoPE-specific head-width restriction.

## Atomistic geometry and outputs

The adapter requests a strict, directed full neighbor list at `cutoff`. Metatomic
owns periodic image enumeration and differentiable neighbor-vector registration.
The core receives vectors pointing from each center to each neighbor image.

Atomic numbers initialize node embeddings. Gaussian radial features of edge
distances pass through an MLP and a smooth quintic cutoff envelope. The resulting
features are **summed** into the appropriate dense `(i,j)` edge state. A learnable
diagonal embedding marks self pairs. Non-neighbor pairs start with zero geometry
features and participate in subsequent dense GenT attention.

Repeated `(i,j)` entries are retained: every periodic image contributes to the
initial pair state. Individual image distances/vectors are also retained for
readout. This includes nonzero-shift self images. There is no nearest-image
selection and no Cartesian-coordinate encoding in the scalar backbone. The sum
is a lossy representation of the image multiset, not an independent hidden state
for every image.

| Raw output | Shape | Construction |
| --- | --- | --- |
| `atomic_energy` | `[A]` | Scalar MLP on final normalized node states, unpadded |
| `energy` | `[B]` | Sum of atomic energy contributions |
| `forces` | `[A,3]` | Learned scalar coefficients times image unit vectors; opposite contributions to the two endpoints |
| `stress` | `[B,3,3]` | Independently learned coefficients times image-vector outer products, summed and divided by cell volume |

The force/stress coefficients use final pair states and each image's radial
features, multiplied by the cutoff envelope. A factor of one half accounts for
the full directed neighbor list. Force assembly gives zero net force, including
when directed edge states differ. Stress is symmetric by construction. For zero
or singular cells the stress readout uses a unit normalization volume; physical
bulk stress comparisons should use nonzero-volume periodic cells.

Energy is invariant to translation, rotation/reflection, atom permutation, and
periodic rewrapping. Direct forces transform as vectors and stress as a rank-2
tensor. The default dropout is zero; symmetry tests run deterministically.

The direct heads are **not** energy derivatives, and their stress coefficients are
independent of the force coefficients. They follow the existing direct-training
target convention (`non_conservative_force`, `non_conservative_stress`). Energy
position/strain gradients can separately be requested through metatrain's usual
energy-gradient machinery, including training through those gradients. Do not
interpret the direct head outputs as those conservative derivatives.

## Input contract outside metatrain

```python
import torch
from gent_core import GeneralizedTransformer, GenTData

model = GeneralizedTransformer(embed_dim=32, num_heads=4, num_layers=2)
data = GenTData(
    atomic_numbers=torch.tensor([1, 8]),
    num_atoms=torch.tensor([2]),
    edge_vectors=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    edge_centers=torch.tensor([0, 1]),
    edge_neighbors=torch.tensor([1, 0]),
    cells=torch.zeros(1, 3, 3),
)
predictions = model(data)
```

Atoms must be concatenated in structure order; counts and indices are integer
tensors. Edge indices address that concatenated atom array and cannot cross
structures. Inputs and model must share a device and floating dtype. Structures
and batches must be nonempty. Zero-neighbor structures are supported. All image
edges inside the cutoff must be supplied in both directions. If conservative
gradients are needed, construct edge vectors differentiably from positions/cells.

## Training in this checkout

[`options-gent-mptrj-salex-direct-160k.yaml`](../../options-gent-mptrj-salex-direct-160k.yaml)
uses the same local `data` and `data_salex_val_30k` sources, atomic-type list, direct
target layouts, composition fitting, target scaling, augmentation, and loss
weights as the Structure Transformer v37 baseline. The architecture and batching
defaults are changed for GenT.

```bash
mtt train options-gent-mptrj-salex-direct-160k.yaml -o gent.pt

# Resume a GenT checkpoint:
mtt train options-gent-mptrj-salex-direct-160k.yaml \
  --restart gent.ckpt -o gent-resumed.pt
```

Activate the existing metatrain environment first. In this workspace it is
`/home/ryoji/miniconda3/envs/metatrain-pet`. When working directly from source, use
`PYTHONPATH=src` if the editable installation does not expose new packages.

Defaults are 128 channels, 4 heads, 4 layers, 32 radial functions, and a cutoff of
5 dataset length units: **2,104,544 parameters**. All three attention widths default
to `embed_dim`; each can independently be set to a positive multiple of
`num_heads`. The MLP width defaults to `int(2.25 * embed_dim)`.

The supplied configuration uses a 256-atom batch budget and a batch size of 2.
An atom budget is not a strict bound on dense pair memory. Memory grows with
`B * N_max^2 * D`, with several edge projections and per-layer activations; the
Structure Transformer baseline's 768-channel, 12-layer settings should not be
copied without profiling. This initial version does not chunk/checkpoint edge
attention. The configuration retains the baseline's online W&B setting; change
`wandb.mode` to `disabled` for a local unlogged run.

One energy target, one Cartesian atom-force target, and one Cartesian system-stress
target are supported, each with one property and one TensorMap block. Target names
are mapped by quantity/convention. Missing heads are rejected at initialization.
Energy can be requested per atom or summed per system. Selected atoms are supported
for energy and forces; direct stress requires the full structure. Restart supports
existing species and target identities; adding new species/targets is rejected.
PET-specific head-transfer/LoRA strategies have not been validated for GenT.

## Validation

Tests are in `src/gent_core/tests` and
`src/metatrain/experimental/gent/tests`. They cover reference-layer parity,
padding and fully masked gradients, finite-difference energy gradients, second
derivatives, periodic multiplicity and self images, isolated atoms, symmetry,
core/adapter agreement, target validation, selected atoms, checkpoint precision,
TorchScript save/load, exported-model consistency in float32/float64, and CPU/CUDA
forward/backward agreement.

The CLI smoke test creates four training and two validation structures, trains all
three direct targets for two CPU epochs using PET composition/scaling/augmentation,
exports a model, restarts the checkpoint for a third epoch, and loads both exported
models. It does not read the production datasets or log to W&B.

```bash
python -m pytest -q src/gent_core/tests \
  src/metatrain/experimental/gent/tests

# Equivalent dedicated environment:
tox -e gent-tests
```

The existing Python environment lacked pytest/Ruff, so validation tools were
installed under `/tmp/metatrain-gent-check-tools`, without modifying that
environment. To reproduce the validation in this session:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src:/tmp/metatrain-gent-check-tools \
  /home/ryoji/miniconda3/envs/metatrain-pet/bin/python -m pytest -q \
  src/gent_core/tests src/metatrain/experimental/gent/tests
```

### Results in this session

Environment: Python 3.11, PyTorch 2.12.1+cu130; CUDA available.

- **30 GenT tests passed**, including CUDA forward/backward in float32/float64 and
  the two-epoch CLI training plus one-epoch restart/export test.
- **26 baseline checks passed** across Structure Transformer core sync, invariant
  backbone/edge-head tests, and PET padding/isolated-atom tests.
- **One existing baseline check failed**:
  `test_v37_hyperparameters_are_exposed_to_config_and_wrapper`. The existing v37
  YAML has an in-progress deletion of `force_readout_type` and related pair-head
  options, while the test still expects `pair_cross_attention`. This configuration
  was already modified before GenT work began and was left untouched. The failure
  also reproduces when that Structure Transformer test is run alone.
- Ruff formatting and lint checks passed for all new Python files.
- The MPtrj/sAlex GenT architecture configuration passed schema validation, and
  setuptools discovers both `gent_core` and `metatrain.experimental.gent`.

The combined final test invocation reported **56 passed, 1 failed**; the failure
was the existing Structure Transformer configuration mismatch described above.
The dedicated tox command is provided for reproduction; the tests in this session
were run directly with pytest in the existing environment, not by creating a new
full tox environment.

Dataset-scale accuracy, throughput, and memory have not been measured. Global
attention also means strict finite-cutoff locality and supercell extensivity are
not guaranteed. Export advertises an infinite interaction range for this reason,
while separately requesting the finite geometry neighbor list.

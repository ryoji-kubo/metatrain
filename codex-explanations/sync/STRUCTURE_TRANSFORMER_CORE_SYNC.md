# Structure Transformer Core Sync

Date: 2026-08-31

This note tracks the effort to decouple the Structure Transformer architecture from
repository-specific training and data plumbing. The goal is to train the same neural
network architecture from both metatrain and equiformer_v3 without copying model code
between repositories.

## Goal

Use one tensor-only Structure Transformer implementation, with thin adapters in each
training repository:

```text
structure_transformer_core
  owns: tensor data contract, dense batching, coordinate encoders, transformer blocks,
        graph-attention injection, edge/readout heads, raw tensor predictions

metatrain.experimental.structure_transformer
  owns: metatomic System input, TensorMap output layout, PET trainer compatibility,
        additive composition model, target scaler, target-name mapping, export metadata

fairchem.experimental.models.transformer
  owns: FairChem model registration, PyG/FairChem batch input, FairChem output naming,
        optimizer/no-weight-decay integration, FairChem normalizer/metric conventions
```

The core should not import `metatrain`, `metatomic`, `metatensor`, `fairchem`, or
FairChem registries. It should only depend on PyTorch and optional PyG for eager dense
batching.

## Phase 1 Completed In metatrain

A new top-level package was added:

```text
src/structure_transformer_core/
  __init__.py
  coordinate_utils.py
  transformer.py
```

The package currently contains the existing metatrain Structure Transformer module,
including:

- `TransformerData`
- `StructureTransformer`
- dense batching with PyG plus Torch fallback
- Cartesian/fractional coordinate handling
- v37 torus-relative coordinate encoder
- periodic coordinate RoPE
- pair cross-attention readouts
- edge-vector readout head
- graph-attention bias hook
- TorchScript/export-friendly adjustments

The metatrain architecture wrapper now imports the core directly:

```python
from structure_transformer_core import StructureTransformer, TransformerData
```

Compatibility shims remain at the previous module paths so existing tests, configs,
and local imports keep working:

```text
src/metatrain/experimental/structure_transformer/modules/transformer.py
src/metatrain/experimental/structure_transformer/modules/coordinate_utils.py
src/metatrain/experimental/structure_transformer/modules/__init__.py
```

These shims re-export from `structure_transformer_core` and should contain no model
logic.

`pyproject.toml` was updated so Ruff treats `structure_transformer_core` as first-party
code.

## Regression Guard Added

A focused sync test was added:

```text
src/metatrain/experimental/structure_transformer/tests/test_core_sync.py
```

It checks two things:

1. the old metatrain module paths re-export the same `StructureTransformer`,
   `TransformerData`, and coordinate conversion function as the new core package;
2. the new core import path can instantiate a small model and run a forward pass that
   returns raw `energy`, `forces`, and `stress` tensors with expected shapes.

## Validation Performed

The `metatrain-pet` environment does not currently have `pytest` installed, so the full
focused pytest command could not run:

```text
/home/ryoji/miniconda3/envs/metatrain-pet/bin/python: No module named pytest
```

I ran direct Python smoke checks instead:

```bash
PYTHONPATH=/home/ryoji/metatrain/src:/home/ryoji/metatrain/src/metatrain/utils/testing \
/home/ryoji/miniconda3/envs/metatrain-pet/bin/python - <<'PY'
# imports new core, imports legacy shims, asserts object identity,
# instantiates StructureTransformer, runs a raw forward pass
PY
```

Result:

```text
core_sync_smoke ok
```

I also ran syntax compilation over the new core and the metatrain adapter package:

```bash
PYTHONPATH=/home/ryoji/metatrain/src \
/home/ryoji/miniconda3/envs/metatrain-pet/bin/python -m compileall -q \
  /home/ryoji/metatrain/src/structure_transformer_core \
  /home/ryoji/metatrain/src/metatrain/experimental/structure_transformer
```

Result: passed.

## Current Source Of Truth

As of this phase, the canonical model implementation is:

```text
src/structure_transformer_core/transformer.py
src/structure_transformer_core/coordinate_utils.py
```

The metatrain adapter source of truth remains:

```text
src/metatrain/experimental/structure_transformer/model.py
```

That adapter is intentionally metatrain-specific. It should continue to own:

- conversion from `list[System]` to `TransformerData`
- neighbor-list requests and extraction from metatomic `System`
- construction of graph-attention bias from metatomic neighbor lists
- conversion from raw tensor outputs to metatensor `TensorMap`s
- selected-atoms slicing
- PET `CompositionModel` and `Scaler` integration
- restart/export metadata

## Next Step: Sync equiformer_v3

When moving back to `/home/ryoji/equiformer_v3`, the intended process is:

1. add or vendor the same `structure_transformer_core` package;
2. replace FairChem's in-file architecture implementation with a FairChem adapter that
   imports `StructureTransformer` and `TransformerData` from the shared core;
3. preserve `@registry.register_model("transformer")` only in the FairChem adapter;
4. convert FairChem/PyG batch objects to `TransformerData` in the adapter;
5. map raw tensor outputs back to FairChem's expected keys: `energy`, `forces`,
   `stress`;
6. port any FairChem-only optimizer hooks such as `no_weight_decay` as wrapper methods
   that delegate to the core when possible;
7. add an Equiformer-side parity test that initializes the FairChem adapter and the
   core with identical weights and verifies matching raw outputs on a tiny batch.

The most important invariant is that architecture changes happen in the core first.
Wrappers may change data contracts and training integration, but they should not fork
attention blocks, coordinate encoders, readout heads, or output-head math.

## Open Design Questions

The current graph-attention bias construction still lives in the metatrain adapter
because it depends on metatomic neighbor-list objects and PET cutoff helpers. For full
cross-repo architectural parity, the next refinement should move the tensor part of
that logic into the core:

```text
edge vectors + centers + neighbors + per-system sizes -> dense graph_attention_bias
```

Then metatrain and FairChem wrappers would only be responsible for extracting edge
vectors from their local graph representation. This matters because choices like
periodic image-edge collapse by max cutoff factor are architectural choices, not
training-pipeline choices.

The package location is also temporary. Keeping `structure_transformer_core` inside
both repos is good enough for the first sync, but the durable solution is an external
installable package or a git subtree/submodule with versioned parity tests.

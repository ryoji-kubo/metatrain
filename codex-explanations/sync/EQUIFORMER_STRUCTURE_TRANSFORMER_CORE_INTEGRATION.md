# Equiformer Structure Transformer Core Integration

Date: 2026-09-01

## Goal

Migrate the Equiformer/FairChem `transformer` model entry point onto the same
`structure_transformer_core` implementation that was introduced in metatrain, so
the architecture can evolve in one place while each repository owns only its
training-pipeline adapter.

## Integration Summary

The Equiformer repo now has a vendored copy of the shared core under:

- `/home/ryoji/equiformer_v3/src/structure_transformer_core/__init__.py`
- `/home/ryoji/equiformer_v3/src/structure_transformer_core/coordinate_utils.py`
- `/home/ryoji/equiformer_v3/src/structure_transformer_core/transformer.py`

These files were copied from `/home/ryoji/metatrain/src/structure_transformer_core`
and verified to be byte-identical at the time of this migration step.

The FairChem registry-facing model file is now a thin adapter:

- `/home/ryoji/equiformer_v3/src/fairchem/experimental/models/transformer/transformer.py`

Note: the repo also exposes this tracked file as
`/home/ryoji/equiformer_v3/experimental/models/transformer/transformer.py` through
the local source symlink layout. Git status reports the tracked path as
`experimental/models/transformer/transformer.py`.

The adapter subclasses `structure_transformer_core.StructureTransformer` rather
than containing it as a child module. This preserves root-level parameter names,
which should be friendlier to checkpoints created from the previous FairChem
implementation.

The old FairChem coordinate helper is now a shim:

- `/home/ryoji/equiformer_v3/src/fairchem/experimental/models/transformer/coordinate_utils.py`

It re-exports `cartesian_to_fractional_dense` from the shared core so coordinate
behavior cannot drift separately.

Package metadata was updated so FairChem core builds include the shared package:

- `/home/ryoji/equiformer_v3/packages/fairchem-core/pyproject.toml`

Ruff import classification was updated for the new first-party package:

- `/home/ryoji/equiformer_v3/ruff.toml`

A focused adapter test was added:

- `/home/ryoji/equiformer_v3/tests/core/models/test_structure_transformer_core_sync.py`

## Adapter Boundary

The Equiformer adapter converts FairChem-style data objects into the core
`TransformerData` tuple:

- `data.atomic_numbers` or `data.z` -> `TransformerData.atomic_numbers`
- `data.pos` -> `TransformerData.pos`
- `data.batch` -> `TransformerData.batch`
- missing `data.batch` -> a single-system zero batch, matching the previous
  FairChem transformer behavior
- `data.cell` -> `TransformerData.cell`

The adapter also keeps the old `num_params` property.

## Edge Vector Head

The Equiformer adapter supports the shared core edge-vector readout when the
FairChem data object already carries graph tensors:

- `data.edge_distance_vec` or `data.distance_vec` supplies edge vectors
- `data.edge_index[1]` is passed as edge centers
- `data.edge_index[0]` is passed as edge neighbors

This matches FairChem's graph convention, where edge index row 0 is the neighbor
/source atom and row 1 is the center/target atom. If `edge_vector_head=True` but
these tensors are absent, the adapter raises a clear `ValueError` instead of
silently disabling the feature.

## Graph Attention

Update on 2026-09-01: graph-attention bias construction has now been moved into
`structure_transformer_core.graph_attention`, and the Equiformer adapter builds
that bias from FairChem `generate_graph` output. See
`GRAPH_ATTENTION_BIAS_SYNC.md` for the implementation details and validation.

## Validation Performed

Passed:

- `cmp -s` confirmed all three copied Equiformer core files match the metatrain
  core files byte-for-byte.
- `python -m compileall -q` passed for the copied core, FairChem transformer
  adapter package, and the new adapter test file in the `equiformer_v3` conda
  environment.
- A direct Python smoke with `PYTHONPATH=/home/ryoji/equiformer_v3/src` passed in
  the `equiformer_v3` conda environment. It checked:
  - the FairChem adapter is a subclass of the shared core model
  - the coordinate shim re-exports the core helper
  - FairChem-style input produces the same outputs as a direct core forward on
    the same model instance
  - missing batch defaults to a single system
  - edge-vector head wiring works with FairChem `edge_index` and
    `edge_distance_vec`
  - graph attention without a supplied bias raises, while an explicit zero bias
    is accepted

Not run as full tooling:

- `python -m pytest -q tests/core/models/test_structure_transformer_core_sync.py`
  could not run because `pytest` is not installed in the local `equiformer_v3`
  conda environment.
- `python -m ruff check ...` could not run because `ruff` is not installed in the
  local `equiformer_v3` conda environment.

## Next Sync Steps

1. Move the metatrain graph-attention bias builder into
   `structure_transformer_core`.
2. Update the metatrain wrapper to call the shared graph-attention builder.
3. Update the Equiformer adapter to build `graph_attention_bias` from FairChem
   graph tensors or generated neighbor lists.
4. Add a cross-repo parity test that loads the same state dict into both adapters
   and compares outputs on the same structure batch.
5. Decide whether `structure_transformer_core` should remain vendored in both
   repos, become a git subtree/submodule, or become a small installable package.

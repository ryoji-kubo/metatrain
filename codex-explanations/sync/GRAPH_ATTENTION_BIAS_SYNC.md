# Graph Attention Bias Sync

Date: 2026-09-01

## Goal

Implement graph-attention bias for the synced Structure Transformer architecture
without tying the shared core to either PET/metatrain neighbor-list objects or
FairChem/Equiformer graph objects.

The shared core now owns the sparse-edge-to-dense-bias math. Each repository
adapter owns the graph source:

- metatrain uses PET/metatomic requested neighbor lists.
- Equiformer/FairChem uses FairChem `GraphModelMixin.generate_graph`.

## Shared Core Change

Added the synced helper file in both repos:

- `/home/ryoji/metatrain/src/structure_transformer_core/graph_attention.py`
- `/home/ryoji/equiformer_v3/src/structure_transformer_core/graph_attention.py`

The files were verified byte-for-byte identical after this sync.

The helper exports `build_dense_graph_attention_bias`, which accepts sparse graph
edges:

- `centers`: center/target atom indices
- `neighbors`: neighbor/source atom indices
- `edge_distances`: one distance per edge
- `batch`: atom-to-system assignment

It returns a dense `[batch_size, max_atoms, max_atoms]` tensor passed directly to
`StructureTransformer.forward(graph_attention_bias=...)`.

The helper preserves the current PET-style attention-bias behavior:

- `binary` graph attention gives factor 1 to graph edges and epsilon to missing
  edges.
- `smooth_cutoff` supports `Bump` and `Cosine` cutoff factors.
- adaptive pair cutoffs support both `solver` and `grid` methods.
- self-attention diagonal entries get factor 1, therefore zero log-bias.
- missing edges receive `graph_attention_bias_strength * log(epsilon)`.

Updated both core `__init__.py` files to export `build_dense_graph_attention_bias`.

## Metatrain Adapter Change

Updated:

- `/home/ryoji/metatrain/src/metatrain/experimental/structure_transformer/model.py`
- `/home/ryoji/metatrain/src/metatrain/experimental/structure_transformer/tests/test_edge_vector_head.py`

The metatrain wrapper still requests PET/metatomic neighbor lists exactly as
before. It now gathers global sparse edge tensors from those neighbor lists and
calls the shared `build_dense_graph_attention_bias` helper instead of carrying
private cutoff/scatter methods locally.

The existing adaptive-cutoff monkeypatch test was updated to patch
`structure_transformer_core.graph_attention.get_adaptive_cutoffs_grid`, proving
that the adaptive math now lives in the core helper.

## Equiformer Adapter Change

Updated:

- `/home/ryoji/equiformer_v3/src/fairchem/experimental/models/transformer/transformer.py`
- `/home/ryoji/equiformer_v3/tests/core/models/test_structure_transformer_core_sync.py`
- `/home/ryoji/equiformer_v3/ruff.toml`

The FairChem adapter now subclasses both the shared core model and
`GraphModelMixin`. It adds the standard FairChem graph-generation knobs:

- `use_pbc`
- `use_pbc_single`
- `otf_graph`
- `max_neighbors`
- `max_radius`
- `enforce_max_neighbors_strictly`

When graph attention is enabled and no explicit `graph_attention_bias` is passed,
the adapter builds the bias from FairChem graph generation:

```python
graph = self.generate_graph(
    data,
    cutoff=self.graph_attention_cutoff,
    max_neighbors=self.max_neighbors,
    enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
    use_pbc_single=self.use_pbc_single,
)
```

The adapter supports the current `GraphData` return object and the tuple-style
return shown in the migration notes. For FairChem graphs, it maps:

- `edge_index[1]` -> centers
- `edge_index[0]` -> neighbors
- `edge_distance` -> distances

That matches FairChem's convention where row 0 is the neighbor/source atom and
row 1 is the center/target atom.

If `edge_vector_head=True`, the adapter can still consume preattached
`data.edge_index` plus `data.edge_distance_vec`/`data.distance_vec`. If no edge
vectors are attached, it generates a FairChem graph for the edge head. When graph
attention and edge-head cutoffs match, it reuses the graph generated for graph
attention.

The Ruff include list now covers `src/structure_transformer_core/**/*.py`, so
the vendored core participates in linting where Ruff is installed.

## Current Attention Method

This sync keeps the current PET-style attention mechanism: build an additive log
bias over dense atom-to-atom attention scores, then pass it into the shared
Transformer blocks.

This is intentionally not yet Equiformer's native attention injection style. The
adapter boundary now makes that future experiment cleaner: Equiformer can later
replace or augment the dense log-bias construction with an Equiformer-style edge
injection path while the metatrain adapter continues using the same core helper.

## Validation Performed

Passed:

- `cmp -s` verified the new shared `graph_attention.py` and updated core
  `__init__.py` are byte-identical between metatrain and Equiformer.
- `python -m compileall -q` passed for the updated Equiformer core, FairChem
  adapter, and focused sync test.
- `python -m compileall -q` passed for the updated metatrain core, wrapper, and
  graph-attention test file.
- Direct Equiformer adapter smoke passed. It checked:
  - adapter/core forward parity with no graph attention
  - missing batch fallback for the dense path
  - edge-vector head mapping from FairChem `edge_index`
  - graph attention built from FairChem `generate_graph`
  - graph-attention plus edge-head operation
  - adaptive-grid graph attention through the Equiformer adapter
- Direct metatrain wrapper smoke passed through the graph-attention forward path.
- Direct shared-helper smoke passed for the adaptive-grid path.

Not run as full tooling:

- `pytest` is not installed in the local `equiformer_v3` or `metatrain-pet`
  conda environments used here.
- `ruff` is not installed in the local `equiformer_v3` conda environment.

## Local Index Fast Path

Update on 2026-09-03: `build_dense_graph_attention_bias` now accepts optional
`atom_counts`. When available, the helper computes per-system local atom indices
with a vectorized offset expression instead of looping over systems:

```python
atom_offsets = torch.cumsum(atom_counts, dim=0) - atom_counts
local_indices = torch.arange(batch.numel(), device=batch.device) - atom_offsets[batch]
```

The Equiformer adapter passes `data.natoms`; the metatrain adapter passes the
per-system `len(system)` tensor. The helper still falls back to `torch.bincount`
when counts are not supplied, and it checks that `batch` is grouped by system so
wrong local indices fail loudly.

## Vectorized Bias Assembly

Update on 2026-09-03: the dense bias assembly no longer loops over systems for
self-attention diagonals or graph-edge scatter. Valid self edges are assigned with
a broadcasted diagonal mask:

```python
diagonal = torch.arange(max_atoms, device=batch.device)
valid_diagonal = diagonal.unsqueeze(0) < atom_counts.unsqueeze(1)
cutoff_factors[:, diagonal, diagonal] = valid_diagonal.to(cutoff_factors.dtype)
```

Graph edge factors are scattered with one flattened `(system, center, neighbor)`
index into the batched dense tensor. Adaptive cutoffs are computed over global
atom indices, which is equivalent to per-system computation because cross-system
edges are rejected before cutoff calculation.

A direct smoke check compared the new vectorized path with the previous
per-system reference for multi-system binary, smooth-cutoff, solver-adaptive, and
grid-adaptive cases.

## Next Sync Steps

1. Add a small command-line sync/check script that verifies the copied core files
   are byte-identical across repos.
2. Add a true cross-repo parity test that loads one state dict into both wrappers
   and compares outputs on the same batch.
3. Experiment with Equiformer's native attention injection method as a separate
   adapter/core extension, keeping the PET-style dense log-bias as the baseline.
4. Decide whether the shared core should stay vendored, move to a git subtree, or
   become an installable package.

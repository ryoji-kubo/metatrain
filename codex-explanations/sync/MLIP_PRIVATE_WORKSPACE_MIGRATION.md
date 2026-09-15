# MLIP private workspace migration after branch cleanup

Updated: 2026-09-15

Branch cleanup is complete. This guide now starts from the cleaned `main` branch
in each framework and keeps **one maintained branch, `main`, in each repository**.
The intended setup still uses private framework repositories, Git submodules,
and public `upstream` remotes.

Work in two stages:

1. **Steps 1–2: Set up the private repositories and workspace.** The models stay
   in their current locations inside each framework during this stage.
2. **Steps 3–8: Consolidate the shared models and validate the wrappers.**

Run one step at a time and stop on errors. The code reconciliation and packaging
changes are manual edits, not an automatic migration script. This document
records instructions; the migration has not been performed.

## Verified starting point

The following was checked locally before this guide was updated:

| Repository | Current branch and commit | Working tree |
| --- | --- | --- |
| `/home/ryoji/MLIP` | `main` at `f6c5b21` | Clean |
| `/home/ryoji/metatrain` | `main` at `97b3f8d` | Clean; model and wrapper changes are committed |
| `/home/ryoji/equiformer_v3` | `main` at `ff34e30` | Tracked files clean; only `matbench-discovery/` is untracked |

The old local metatrain branch `structure-transformer-v37-invariant` still exists,
but it is fully merged into `main`. It does not need to be deleted or transferred.
`matbench-discovery/` is a separate Git checkout; keep it separate from this
migration and do not stage it as framework source.

Both frameworks currently have only `origin`, pointing to their public forks.
The commands below give the **new submodule checkouts** private origins and the
original projects as upstreams. The original sibling checkouts retain their
existing remotes.

## Target layout

```text
/home/ryoji/MLIP/                     Private repository, branch main
├── .gitmodules                      Records private framework repositories
├── pyproject.toml                   Shared distribution: mlip-models
├── src/
│   ├── structure_transformer_core/
│   └── gent_core/
│       └── tests/
├── tests/                           Shared and cross-framework tests
├── notebooks/
├── metatrain/                       Private submodule, branch main
│   └── src/metatrain/experimental/  metatrain wrappers
└── equiformer_v3/                    Private submodule, branch main
    └── experimental/models/         FairChem wrappers
```

| Checkout | `origin`: save your work here | `upstream`: obtain original updates here |
| --- | --- | --- |
| MLIP | `ryoji-kubo/MLIP` (private, already exists) | None required |
| MLIP/metatrain | `ryoji-kubo/metatrain-private` (new private repository) | `metatensor/metatrain` |
| MLIP/equiformer_v3 | `ryoji-kubo/equiformer_v3-private` (new private repository) | `atomicarchitects/equiformer_v3` |

MLIP records an exact commit of each framework alongside the shared cores.
Submodules keep the frameworks' Git histories and remotes independent.
See [Git's submodule overview](https://git-scm.com/docs/gitsubmodules).

## 1. Create the private framework repositories and transfer main

On GitHub, create two **empty private repositories** named `metatrain-private`
and `equiformer_v3-private`. Leave README, license, and `.gitignore` initialization
unchecked. Use **New repository**: public forks cannot independently become
private. See [GitHub's fork visibility rules](https://docs.github.com/en/pull-requests/reference/forks).

Use these variables in one Bash session; substitute repository names if desired:

```bash
MLIP_ROOT=/home/ryoji/MLIP
METATRAIN_OLD=/home/ryoji/metatrain
EQUIFORMER_OLD=/home/ryoji/equiformer_v3
METATRAIN_PRIVATE_URL=https://github.com/ryoji-kubo/metatrain-private.git
EQUIFORMER_PRIVATE_URL=https://github.com/ryoji-kubo/equiformer_v3-private.git
MLIP_CONDA=/home/ryoji/miniconda3/bin/conda
```

Check that the framework checkouts are still on the intended `main` versions:

```bash
git -C "$METATRAIN_OLD" status --short --branch
git -C "$EQUIFORMER_OLD" status --short --branch
git -C "$METATRAIN_OLD" fetch origin --tags
git -C "$EQUIFORMER_OLD" fetch origin --tags
git -C "$METATRAIN_OLD" log --oneline main..origin/main
git -C "$EQUIFORMER_OLD" log --oneline main..origin/main
```

The last two commands should show no additional remote commits. If they do,
update the relevant local `main` before transferring it. Commit any new work you
want included, such as this revised guide, **locally** before the transfer.
There is no need to repeat the earlier model checkpoint or branch-merging steps.

If a checkout is shallow (`git rev-parse --is-shallow-repository` reports `true`),
fetch its full history with `git fetch --unshallow origin` before continuing.
If it uses Git LFS, fetch its LFS payloads from its existing origin before
transferring them to the private repository; see
[GitHub's repository duplication guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository).

Transfer **main and tags** directly to the private URLs:

```bash
git -C "$METATRAIN_OLD" push "$METATRAIN_PRIVATE_URL" main
git -C "$METATRAIN_OLD" push "$METATRAIN_PRIVATE_URL" --tags
git -C "$EQUIFORMER_OLD" push "$EQUIFORMER_PRIVATE_URL" main
git -C "$EQUIFORMER_OLD" push "$EQUIFORMER_PRIVATE_URL" --tags
```

These commands preserve the history reachable from `main` and the tags, without
creating the old feature branches in the private repositories. Tags also support
the frameworks' existing version-generation mechanisms. If Git LFS is in use,
upload the downloaded LFS objects to the private destination as well.

Verify that each new GitHub repository is private and its default branch is
`main`. GitHub issues, PRs, and repository settings are separate from Git history.
The existing public forks remain public; this step does not hide already
published code.

Keep both original folders in place until validation is complete. Their ignored
files, datasets, local environment settings, and separate Matbench checkout stay
there; only committed source is cloned into the new workspace. Preserve those
local assets through your existing backups and transfer them selectively when
needed.

## 2. Add the private frameworks as submodules under MLIP

Starting from a clean MLIP working tree:

```bash
cd "$MLIP_ROOT"
git switch main
git pull --ff-only origin main

git submodule add -b main "$METATRAIN_PRIVATE_URL" metatrain
git submodule add -b main "$EQUIFORMER_PRIVATE_URL" equiformer_v3
git config push.recurseSubmodules check
```

The destination folders must not already contain files. Git automatically sets
`origin` in each new submodule to its private URL. Add the original public
projects as additional remotes:

```bash
git -C "$MLIP_ROOT/metatrain" remote add upstream https://github.com/metatensor/metatrain.git
git -C "$MLIP_ROOT/metatrain" config remote.pushDefault origin
git -C "$MLIP_ROOT/equiformer_v3" remote add upstream https://github.com/atomicarchitects/equiformer_v3.git
git -C "$MLIP_ROOT/equiformer_v3" config remote.pushDefault origin

git -C "$MLIP_ROOT/metatrain" remote -v
git -C "$MLIP_ROOT/equiformer_v3" remote -v
git submodule status
```

Each `origin` must show its private URL and each `upstream` the original project.
If repeating setup on a checkout that already has `upstream`, inspect its URL
and use `remote set-url upstream ...` if a correction is needed.

Save this workspace setup:

```bash
cd "$MLIP_ROOT"
git add .gitmodules metatrain equiformer_v3
git diff --cached --submodule=log
git commit -m "Track private framework copies as submodules"
git push origin main
```

**Stage one is complete.** You now have private framework copies under MLIP,
with their upstream connections preserved. Continue all new development inside
these submodules. The original sibling folders remain references.

The shared models are still duplicated at this point. The remaining steps make
both frameworks use a single implementation. They can be handled separately
from the repository setup.

## 3. Establish the shared cores under MLIP

Create the shared source directory and copy the current metatrain versions as
the starting point:

```bash
mkdir -p "$MLIP_ROOT/src" "$MLIP_ROOT/tests"
cp -a "$MLIP_ROOT/metatrain/src/structure_transformer_core" "$MLIP_ROOT/src/"
cp -a "$MLIP_ROOT/metatrain/src/gent_core" "$MLIP_ROOT/src/"

git diff --no-index \
  "$MLIP_ROOT/src/structure_transformer_core" \
  "$MLIP_ROOT/equiformer_v3/src/structure_transformer_core"
```

`git diff --no-index` returns status 1 when files differ; that is expected here.
Review source changes separately from generated caches.

Reconcile the differences manually into `MLIP/src/structure_transformer_core`.
At inspection time, `transformer.py` and `graph_attention.py` differed between
the repositories. The Equiformer version includes a polynomial-envelope
graph-attention mode and associated configuration. Preserve those intentional
features while retaining the metatrain defaults and behavior.

Keep model calculations in the shared core. The wrappers should extract local
data, call the core, and format results. Preserve the public import names:

```python
from structure_transformer_core import StructureTransformer, TransformerData
from gent_core import GeneralizedTransformer, GenTData
```

GenT currently has a metatrain wrapper. This migration does not by itself create
a GenT wrapper for Equiformer; that can be a later addition using the same core.

Keep the existing core tests with the copied source initially. One current GenT
test needs a path correction: `src/gent_core/tests/test_core.py` looks for
`gent_layers_original.py` under `structure_transformer_core`, but that file is
under `gent_core`. In `test_reference_parity`, replace the reference-file lookup
with:

```python
import gent_core

path = Path(gent_core.__file__).with_name("gent_layers_original.py")
```

The test already imports `Path`. This lookup works with the new location and
with an installed package. Keep the original source attribution and notices
when extracting files into MLIP.

## 4. Make the cores an installable Python distribution

Create `/home/ryoji/MLIP/pyproject.toml` with this initial configuration:

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "mlip-models"
version = "0.1.0"
description = "Shared model cores for MLIP framework adapters"
requires-python = ">=3.10"
dependencies = ["torch"]

[project.optional-dependencies]
test = ["pytest"]
pyg = ["torch-geometric"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["structure_transformer_core*", "gent_core*"]
exclude = ["*.tests", "*.tests.*"]
namespaces = false

[tool.pytest.ini_options]
testpaths = ["src/gent_core/tests", "tests"]
```

`mlip-models` is the distribution installed by pip; the import names remain
`structure_transformer_core` and `gent_core`. Both can live in one distribution
and one Git repository. There is no need to publish it to PyPI.

Package discovery is limited to MLIP's `src/`, so installing MLIP does not also
package either framework submodule. The Structure Transformer core currently
has an optional PyG dense-batching path and a Torch fallback. Start with the
existing working Torch versions in each environment, then record the versions
you validate. The broad dependency above is not a claim of compatibility with
every Torch version. See [setuptools package discovery](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html).

Ensure MLIP's `.gitignore` covers Python build artifacts, for example:

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
/build/
/dist/
```

## 5. Make both frameworks depend on the shared distribution

After reconciling the cores, remove the duplicate core packages from the **new
submodule checkouts**. The cores are already tracked on the cleaned framework
branches, so these removals will be recorded in Git:

```bash
git -C "$MLIP_ROOT/metatrain" rm -r src/structure_transformer_core src/gent_core
git -C "$MLIP_ROOT/equiformer_v3" rm -r src/structure_transformer_core
```

Keep the framework wrappers and compatibility shims. They already import the
core package names, which will now resolve to MLIP's installed distribution.

Make these packaging and test-configuration edits:

| File, relative to MLIP | Required change |
| --- | --- |
| `metatrain/pyproject.toml` | Add `"mlip-models>=0.1,<0.2"` to `[project].dependencies`. Keep package discovery within the framework's own `src/`. |
| `equiformer_v3/packages/fairchem-core/pyproject.toml` | Add the same dependency. Remove `"src/structure_transformer_core"` from both the sdist and wheel `only-include` lists. |
| `metatrain/tox.ini` | Remove the old `{toxinidir}/src/gent_core/tests` test path. Run core tests from MLIP; retain metatrain adapter tests here. Install the MLIP package in isolated environments that run those adapters. |
| `equiformer_v3/ruff.toml` and framework lint settings | Remove obsolete core source paths. Lint shared cores from MLIP and wrappers from their framework. |

For local tox environments under this layout, a dependency path such as
`{toxinidir}/..` points from `MLIP/metatrain` to the MLIP distribution. Add it to
the relevant environment's existing `deps`, preserving other dependencies.
CI must check out the MLIP workspace and install the shared distribution too.
A framework-only checkout needs a separately supplied `mlip-models` package to
run the private adapters.

Keep the existing framework-facing class structure and checkpoint parameter
names during extraction. In particular, the FairChem wrapper currently
subclasses the core, while metatrain wraps it. Changing that relationship can
affect old checkpoints even when the mathematical model is unchanged.

## 6. Install the workspace into separate development environments

Keep separate environments for metatrain and Equiformer. To preserve the current
working environments during migration, clone them first:

```bash
"$MLIP_CONDA" create --name mlip-metatrain --clone metatrain-pet
"$MLIP_CONDA" create --name mlip-equiformer --clone equiformer_v3
```

Both source environments were found during the 2026-09-15 inspection; substitute
the names of your working environments if they have changed. If the new
environment names already exist, inspect them before reusing them.

Reinstall the local packages from the new paths:

```bash
"$MLIP_CONDA" run -n mlip-metatrain python -m pip install --no-deps \
  -e "$MLIP_ROOT" -e "$MLIP_ROOT/metatrain"
"$MLIP_CONDA" run -n mlip-equiformer python -m pip install --no-deps \
  -e "$MLIP_ROOT" -e "$MLIP_ROOT/equiformer_v3/packages/fairchem-core"

"$MLIP_CONDA" run -n mlip-metatrain python -m pip install pytest
"$MLIP_CONDA" run -n mlip-equiformer python -m pip install pytest
"$MLIP_CONDA" run -n mlip-metatrain python -m pip check
"$MLIP_CONDA" run -n mlip-equiformer python -m pip check
```

`--no-deps` here preserves the existing working dependency stack while replacing
the local editable packages. Resolve missing requirements reported by `pip check`
before continuing. For a new machine, establish the frameworks' dependencies
using their environment documentation before using these installation commands.

An editable installation makes imports use the development source. Both
environments can therefore use the same files in MLIP; new Python processes see
source edits without copying the files. Restart long-running processes and
notebook kernels after edits. See [pip's local project installation guide](https://pip.pypa.io/en/stable/topics/local-project-installs/).

The Equiformer package currently uses relative symlinks, including
`packages/fairchem-core/src -> ../../src` and
`src/fairchem/experimental/models -> ../../../experimental/models/`. Preserve
that tracked layout. Make wrapper edits through `experimental/models/`, the
underlying tracked path.

Update launch configurations, scripts, and notebook setup to use the new
environments and workspace paths. The current metatrain `.vscode/launch.json`
contains `/home/ryoji/metatrain` working directories. Review other old paths with:

```bash
rg -n --hidden -g '!.git' -g '!*.ipynb' \
  '/home/ryoji/(metatrain|equiformer_v3)|PYTHONPATH' \
  "$MLIP_ROOT"
```

Review matches individually: some paths point to datasets that should stay where
they are. Remove stale `PYTHONPATH` entries that could load old source copies.
Ignored data, results, local IDE settings, and separately cloned dependencies are
not copied by submodule cloning; transfer needed local files selectively or
configure their existing locations.

## 7. Verify imports and behavior

First verify that both environments load both cores from MLIP:

```bash
cd "$MLIP_ROOT"
for mlip_env in mlip-metatrain mlip-equiformer; do
  "$MLIP_CONDA" run -n "$mlip_env" python -c '
import sys
from pathlib import Path
import structure_transformer_core
import gent_core

expected = (Path(sys.argv[1]) / "src").resolve()
for module in (structure_transformer_core, gent_core):
    actual = Path(module.__file__).resolve()
    print(module.__name__, actual)
    assert actual.is_relative_to(expected), (actual, expected)
' "$MLIP_ROOT"
done
```

The paths must start with `/home/ryoji/MLIP/src/`. Also inspect the framework paths:

```bash
"$MLIP_CONDA" run -n mlip-metatrain python -c \
  'import metatrain; print(metatrain.__file__)'
"$MLIP_CONDA" run -n mlip-equiformer python -c \
  'from fairchem.experimental.models.transformer import transformer; print(transformer.__file__)'
```

Those must resolve inside the appropriate MLIP submodules. If they point to an
original sibling checkout or an unexpected installed copy, fix installation or
path settings before trusting test results.

Run the existing focused tests:

```bash
cd "$MLIP_ROOT"
"$MLIP_CONDA" run -n mlip-metatrain python -m pytest -q src/gent_core/tests
"$MLIP_CONDA" run -n mlip-equiformer python -m pytest -q src/gent_core/tests

cd "$MLIP_ROOT/metatrain"
"$MLIP_CONDA" run -n mlip-metatrain python -m pytest -q \
  src/metatrain/experimental/structure_transformer/tests/test_core_sync.py \
  src/metatrain/experimental/structure_transformer/tests/test_edge_vector_head.py \
  src/metatrain/experimental/gent/tests

cd "$MLIP_ROOT/equiformer_v3"
"$MLIP_CONDA" run -n mlip-equiformer python -m pytest -q \
  tests/core/models/test_structure_transformer_core_sync.py
```

These are instructions to run tests; this document does not claim they already
pass. Earlier sync notes reported missing pytest in the original environments,
and the current GenT reference-path issue is described in step 3.

Before treating the migration as complete:

1. Exercise both the original graph-attention settings and the reconciled
   Equiformer envelope option.
2. Add a cross-framework parity check under MLIP's `tests/`: use one set of core
   weights, equivalent inputs and neighbor graphs, matched units and settings,
   and compare raw predictions. Run the wrappers in separate processes if their
   dependencies require separate environments. Account for wrapper scaling and
   additive models when comparing outputs.
3. Run a small forward/backward or training step in each framework, and check
   loading an existing checkpoint and any export path you use.
4. For jobs that use built packages, also validate regular installations of the
   shared distribution and wrappers. Editable installs alone do not verify wheel
   contents or deployment behavior.

Keep model behavior changes separate from directory and packaging changes where
possible, so failures can be attributed to the right change.

## 8. Publish the framework changes, then the shared cores

After the checks in step 7 pass, commit the wrapper, packaging, and duplicate
removals inside each framework. The new submodule checkouts should still be on
`main`. Review the staged changes and explicitly stage any new files you added:

```bash
cd "$MLIP_ROOT/metatrain"
git add -u
git diff --cached --stat
git diff --cached --check
git commit -m "Use shared model cores from MLIP"
git push origin main

cd "$MLIP_ROOT/equiformer_v3"
git add -u
git diff --cached --stat
git diff --cached --check
git commit -m "Use shared model cores from MLIP"
git push origin main
```

Then commit MLIP's shared source, packaging, tests, and matching framework
commit references. Stage any other new setup files you deliberately created:

```bash
cd "$MLIP_ROOT"
git add pyproject.toml src tests metatrain equiformer_v3
git add -p
git status --short
git diff --cached --submodule=log
git diff --cached --check
git commit -m "Share model cores between metatrain and Equiformer"
git push origin main
```

Always push the framework commits first so another machine can fetch the
commits referenced by MLIP. A parent push does not upload submodule changes.
The `push.recurseSubmodules=check` setting helps detect missing child commits.

## Check a fresh clone

Confirm that all three repositories can be downloaded together:

```bash
mlip_check=$(mktemp -d /home/ryoji/mlip-check.XXXXXX)
git clone --branch main --recurse-submodules \
  https://github.com/ryoji-kubo/MLIP.git "$mlip_check"
git -C "$mlip_check" submodule status --recursive
git -C "$mlip_check" status --short
```

A new user or machine needs access to all three private repositories. Repeat
the installations and import checks in separate validation environments with
`MLIP_ROOT` set to the new clone. Keep working environments pointed at the
primary workspace.

For an existing clone, `git submodule update --init --recursive` checks out the
versions recorded by MLIP. Submodules are normally checked out at detached HEADs;
before editing a framework, switch it to `main`. Its branch may be newer than
MLIP's pinned version, so review that difference before recording it in MLIP.

Extra `upstream` remotes and local Git settings are not copied by cloning.
Repeat the upstream setup from step 2 in each fresh workspace. `.gitmodules`
provides the private origin URLs and branch preferences; MLIP still pins exact
commits. See [Git's submodule commands](https://git-scm.com/docs/git-submodule).

Keep the original folders until experiments, notebooks, and data paths work
from MLIP. Archiving those folders is a separate, later cleanup task.

## Later: merge changes from the original projects

You can keep working on `main` in both frameworks. Start from clean working trees
and bring MLIP's own `main` up to date before recording a framework update.
For example, update metatrain with:

```bash
cd /home/ryoji/MLIP
git switch main
git pull --ff-only origin main
git submodule update --init --recursive

cd /home/ryoji/MLIP/metatrain
git switch main
git pull --ff-only origin main
git fetch upstream
git merge upstream/main
```

Resolve any conflicts, stage the resolutions, and finish an interrupted merge
with `git merge --continue`. Use `git merge --abort` to abandon an in-progress
merge. Run the relevant wrapper tests from step 7 before publishing the result.
If tests fail after a completed merge, resolve those failures before pushing.

After validation:

```bash
cd /home/ryoji/MLIP/metatrain
git push origin main

cd /home/ryoji/MLIP
git add metatrain
git diff --cached --submodule=log
git commit -m "Update metatrain to tested upstream integration"
git push origin main
```

For Equiformer, use the same sequence inside `MLIP/equiformer_v3`, then stage
`equiformer_v3` in MLIP. Each framework's `upstream` refers to its own original
project, so updates can be merged and tested independently.

Use ordinary merges for upstream updates to retain their ancestry. If branch
protection requires a PR, use a temporary review branch and merge that PR with a
merge commit. Only `main` needs to remain as the maintained branch.

`git submodule update --init --recursive` restores MLIP's recorded versions.
`git submodule update --remote` follows the private framework branches. Neither
command performs the merge from the original public upstream described above.

## Where to commit daily changes

| Change | Commit location |
| --- | --- |
| Attention, encoders, heads, or other shared model calculations | MLIP |
| Shared notebooks, cross-framework tests, or workspace setup | MLIP |
| metatrain wrapper or trainer integration | `MLIP/metatrain`, then update its commit reference in MLIP |
| Equiformer/FairChem wrapper or trainer integration | `MLIP/equiformer_v3`, then update its commit reference in MLIP |
| A change spanning core and wrappers | Test together; push child commits, then commit the matching core and child references in MLIP |

Use root-level experiment records to save the MLIP commit and environment
versions. A clean MLIP commit identifies the shared core source and both framework
commits. Record uncommitted changes separately if you run an experiment from a
dirty working tree.

## Related design notes

- [Structure Transformer core extraction](STRUCTURE_TRANSFORMER_CORE_SYNC.md)
- [Equiformer wrapper integration](EQUIFORMER_STRUCTURE_TRANSFORMER_CORE_INTEGRATION.md)
- [Graph-attention core extraction](GRAPH_ATTENTION_BIAS_SYNC.md)

These notes describe the earlier duplicated-core arrangement. After migration,
the canonical cores live in `MLIP/src/`, and the framework repositories contain
the adapters that use them.

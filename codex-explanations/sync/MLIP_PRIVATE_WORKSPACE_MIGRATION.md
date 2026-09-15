# Organizing MLIP, metatrain, and equiformer_v3

Date: 2026-09-14

This guide describes the migration to a private MLIP workspace with shared model
code and two independently maintained framework repositories. Writing this guide
does not perform the migration. Run the steps in order, inspect each result, and
stop if a command fails. The code reconciliation and packaging edits are manual
steps; this document is not a script to execute all at once.

## Intended result

```text
/home/ryoji/MLIP/                     Git repository: private MLIP
├── .gitmodules                      Framework URLs and branch preferences
├── pyproject.toml                   Installable distribution: mlip-models
├── src/
│   ├── structure_transformer_core/  One shared Structure Transformer core
│   └── gent_core/                   One shared GenT core
│       └── tests/                   Existing GenT core tests
├── tests/                           Additional core and integration tests
├── notebooks/
├── metatrain/                       Git submodule: private metatrain copy
│   └── src/metatrain/experimental/  metatrain-specific wrappers
└── equiformer_v3/                    Git submodule: private Equiformer copy
    └── experimental/models/         FairChem-specific wrappers
```

There will be three private GitHub repositories:

| Repository | Owns | Public upstream |
| --- | --- | --- |
| `ryoji-kubo/MLIP` | Shared cores, workspace configuration, shared experiments | None required |
| `ryoji-kubo/metatrain-private` | metatrain and its local wrappers | `metatensor/metatrain` |
| `ryoji-kubo/equiformer_v3-private` | Equiformer/FairChem and its local wrappers | `atomicarchitects/equiformer_v3` |

The two new names are suggestions; substitute your chosen names consistently.
Equiformer's upstream was confirmed from the public
[fork metadata](https://api.github.com/repos/ryoji-kubo/equiformer_v3).

A submodule retains its own Git history and remotes. MLIP stores a reference to a
particular commit of each framework, alongside the matching shared model code.
MLIP does not store the framework files as ordinary parent-repository files.
[Git's submodule overview](https://git-scm.com/docs/gitsubmodules) explains this
relationship.

Each framework's `origin` will point to its private repository; its `upstream`
will point to the original public project. Nesting repositories under private
MLIP does not change their visibility. GitHub requires forks of public
repositories to remain public, so this guide creates independent private copies
with the original history. See [GitHub's fork visibility rules](https://docs.github.com/en/pull-requests/reference/forks).

## 1. Preserve the current work

At inspection time:

- MLIP is on `main`, with a clean working tree.
- metatrain is on `structure-transformer-v37-invariant`, with modified and
  untracked files, including both extracted cores.
- equiformer_v3 is on `main`, with modified and untracked files, including its
  extracted Structure Transformer core.

The migration must start from those working versions. A fresh clone of the
current public forks would omit the uncommitted changes and untracked cores.

Use these variables in one Bash session:

```bash
MLIP_ROOT=/home/ryoji/MLIP
METATRAIN_OLD=/home/ryoji/metatrain
EQUIFORMER_OLD=/home/ryoji/equiformer_v3
METATRAIN_PRIVATE_URL=https://github.com/ryoji-kubo/metatrain-private.git
EQUIFORMER_PRIVATE_URL=https://github.com/ryoji-kubo/equiformer_v3-private.git
MLIP_CONDA=/home/ryoji/miniconda3/bin/conda
```

Check the state again:

```bash
git -C "$MLIP_ROOT" status --short --branch
git -C "$METATRAIN_OLD" status --short --branch
git -C "$EQUIFORMER_OLD" status --short --branch
du -sh "$MLIP_ROOT" "$METATRAIN_OLD" "$EQUIFORMER_OLD"
df -h /home/ryoji
```

Make a filesystem backup or snapshot before changing the repositories. If there
is sufficient disk space, the following creates a separate backup directory:

```bash
mlip_backup=$(mktemp -d /home/ryoji/mlip-migration-backup.XXXXXX)
cp -a "$MLIP_ROOT" "$mlip_backup/MLIP"
cp -a "$METATRAIN_OLD" "$mlip_backup/metatrain"
cp -a "$EQUIFORMER_OLD" "$mlip_backup/equiformer_v3"
```

Record the backup path. These copies include `.git`, untracked files, and ignored
files. Symlinks remain symlinks; independently back up external targets if needed.
Pause writers such as training jobs while taking a consistent backup, or use a
filesystem snapshot. A Git bundle alone would not preserve uncommitted files.

Keep the original folders throughout validation. The new workspace will use new
submodule checkouts, so there is no need to move or delete either original folder.

## 2. Create the two private GitHub repositories

On GitHub, use **New repository** to create:

1. `ryoji-kubo/metatrain-private`, with visibility **Private**.
2. `ryoji-kubo/equiformer_v3-private`, with visibility **Private**.

Leave both empty: do not initialize a README, `.gitignore`, or license. These are
independent repositories; do not use the Fork button. Keep the existing private
MLIP repository.

Git history will be transferred by pushing your existing branches. Issues, PRs,
Actions secrets, and repository settings need separate setup if you want them.
The existing public forks will remain public; creating private copies does not
remove already published code.

## 3. Checkpoint the working versions and push them privately

First fetch the public forks' branch information while `origin` still refers to
them:

```bash
git -C "$METATRAIN_OLD" fetch origin --tags
git -C "$METATRAIN_OLD" branch -a
git -C "$EQUIFORMER_OLD" fetch origin --tags
git -C "$EQUIFORMER_OLD" branch -a
```

If a checkout is shallow (`git rev-parse --is-shallow-repository` prints `true`),
run `git fetch --unshallow origin` in that checkout before proceeding. If it uses
Git LFS, run `git lfs fetch --all origin` now, while `origin` still points to the
public fork, to download its LFS payloads.

The later `push --all` transfers all **local branches**. For any additional
remote-only branch you want to preserve, first create a local branch at its
`origin/BRANCH` commit using `git branch BRANCH origin/BRANCH`. Do not replace an
existing local branch. Stashes and ignored files remain in the original checkout
and backup; they are not transferred by pushing branches.

Now change the push destinations to the new private repositories:

```bash
git -C "$METATRAIN_OLD" remote set-url origin "$METATRAIN_PRIVATE_URL"
git -C "$METATRAIN_OLD" remote set-url --push origin "$METATRAIN_PRIVATE_URL"
git -C "$METATRAIN_OLD" remote add upstream https://github.com/metatensor/metatrain.git
git -C "$METATRAIN_OLD" config remote.pushDefault origin

git -C "$EQUIFORMER_OLD" remote set-url origin "$EQUIFORMER_PRIVATE_URL"
git -C "$EQUIFORMER_OLD" remote set-url --push origin "$EQUIFORMER_PRIVATE_URL"
git -C "$EQUIFORMER_OLD" remote add upstream https://github.com/atomicarchitects/equiformer_v3.git
git -C "$EQUIFORMER_OLD" config remote.pushDefault origin

git -C "$METATRAIN_OLD" remote -v
git -C "$EQUIFORMER_OLD" remote -v
```

Both fetch and push entries for each `origin` must show its private URL. If an
`upstream` already exists when you perform this migration, check its URL and use
`remote set-url upstream ...` instead of adding it again.

Create a branch named `mlip-integration` in each original checkout, starting from
its current branch. For metatrain, this preserves the work based on
`structure-transformer-v37-invariant`; do not switch to `main` first.

```bash
cd "$METATRAIN_OLD"
git switch -c mlip-integration
git add -p
git add src/structure_transformer_core src/gent_core
git add src/metatrain/experimental/structure_transformer src/metatrain/experimental/gent
git add codex-explanations/sync
git status --short
```

`git add -p` lets you review modifications to tracked files. Explicitly stage the
other new configuration files, scripts, documentation, and tests needed for your
working version. Inspect `git ls-files --others --exclude-standard` to find them.
Review staged changes before committing:

```bash
git diff --cached --stat
git diff --cached --check
git commit -m "Checkpoint models and wrappers before MLIP migration"
```

Do the same for Equiformer:

```bash
cd "$EQUIFORMER_OLD"
git switch -c mlip-integration
git add -p
git add src/structure_transformer_core
git add experimental/models/transformer
git add tests/core/models/test_structure_transformer_core_sync.py
git status --short
```

Stage any additional required experiment configurations and source files, then:

```bash
git diff --cached --stat
git diff --cached --check
git commit -m "Checkpoint models and wrappers before MLIP migration"
```

Keep datasets, checkpoints, generated outputs, and the separate
`matbench-discovery/` checkout out of these source commits. Avoid staging that
nested checkout as an accidental embedded repository. Record its version and
installation separately if your experiments use it.

Before proceeding, ensure all code needed to reproduce the current working
models is committed. Unrelated local files may remain in the original folders,
but the new submodule clones will only receive committed files.

Push the local branches and tags to the verified private destinations:

```bash
git -C "$METATRAIN_OLD" push -u origin --all
git -C "$METATRAIN_OLD" push origin --tags
git -C "$EQUIFORMER_OLD" push -u origin --all
git -C "$EQUIFORMER_OLD" push origin --tags
```

If either repository uses Git LFS, run `git lfs push --all origin` inside that
checkout to upload the LFS payloads downloaded earlier. `origin` now points to
its private destination. Git branches alone do not copy LFS payloads; see
[GitHub's duplication guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository).

On GitHub, set `mlip-integration` as the default branch of each new private
framework repository. The original `main` and feature branches remain available.

## 4. Add the frameworks as submodules under MLIP

Start with a clean MLIP working tree, and update its existing `main`:

```bash
cd "$MLIP_ROOT"
git switch main
git pull --ff-only origin main
git switch -c organize-shared-models

git submodule add -b mlip-integration "$METATRAIN_PRIVATE_URL" metatrain
git submodule add -b mlip-integration "$EQUIFORMER_PRIVATE_URL" equiformer_v3
git config push.recurseSubmodules check
git submodule status
```

The `metatrain/` and `equiformer_v3/` paths must not already contain files. These
commands clone from the private repositories, including the checkpoint commits.
The `-b` option records a branch preference; MLIP still records exact commits.
See [Git's submodule commands](https://git-scm.com/docs/git-submodule).

The new clones do not inherit remotes from the original local checkouts. Add
their upstreams explicitly:

```bash
git -C "$MLIP_ROOT/metatrain" remote add upstream https://github.com/metatensor/metatrain.git
git -C "$MLIP_ROOT/metatrain" config remote.pushDefault origin
git -C "$MLIP_ROOT/equiformer_v3" remote add upstream https://github.com/atomicarchitects/equiformer_v3.git
git -C "$MLIP_ROOT/equiformer_v3" config remote.pushDefault origin
```

Do not add these two directories to MLIP's `.gitignore`. Git should track them as
submodules, with their URLs in `.gitmodules`.

## 5. Establish the shared cores under MLIP

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

## 6. Make the cores an installable Python distribution

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

## 7. Make both frameworks depend on the shared distribution

After reconciling the cores, remove the duplicate core packages from the **new
submodule checkouts**. Step 3 made them tracked files, so these removals will be
recorded in Git:

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

## 8. Install the workspace into separate development environments

Keep separate environments for metatrain and Equiformer. To preserve the current
working environments during migration, clone them first:

```bash
"$MLIP_CONDA" create --name mlip-metatrain --clone metatrain-pet
"$MLIP_CONDA" create --name mlip-equiformer --clone equiformer_v3
```

These source environment names come from the existing sync notes; substitute
the names of your actual working environments if they have changed. If the new
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

## 9. Verify imports and behavior

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
and the current GenT reference-path issue is described in step 5.

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

## 10. Publish the framework commits, then the MLIP commit

Review and commit the wrapper, packaging, and duplicate-removal changes inside
each framework. Stage any additional files you deliberately changed in step 7:

```bash
cd "$MLIP_ROOT/metatrain"
git add -u
git diff --cached --stat
git diff --cached --check
git commit -m "Use shared model cores from MLIP"
git push origin mlip-integration

cd "$MLIP_ROOT/equiformer_v3"
git add -u
git diff --cached --stat
git diff --cached --check
git commit -m "Use shared model cores from MLIP"
git push origin mlip-integration
```

Then stage MLIP's shared source, packaging, and exact submodule commits:

```bash
cd "$MLIP_ROOT"
git add .gitmodules pyproject.toml src metatrain equiformer_v3
git add -p
git status --short
```

Explicitly stage the new tests and any new workspace setup files you added.
Review the complete staged change and publish the workspace branch:

```bash
git diff --cached --submodule=log
git diff --cached --check
git commit -m "Organize shared model cores and private framework submodules"
git push -u origin organize-shared-models
```

Publishing the framework commits first ensures that a new clone can fetch every
commit referenced by MLIP. A parent commit records submodule commit IDs; it does
not upload their contents. The local `push.recurseSubmodules=check` setting adds
a useful check, but keep the explicit child-first publishing order.

## 11. Confirm that the workspace can be recreated

Before merging the MLIP branch into `main`, try a separate clone:

```bash
mlip_check=$(mktemp -d /home/ryoji/mlip-check.XXXXXX)
git clone --branch organize-shared-models --recurse-submodules \
  https://github.com/ryoji-kubo/MLIP.git "$mlip_check"
git -C "$mlip_check" submodule status --recursive
git -C "$mlip_check" status --short
```

Anyone doing this needs access to all three private repositories. On a separate
validation environment or machine, repeat the installations and import checks
with `MLIP_ROOT` set to the new clone. Do not repoint the working development
environments just to validate a disposable clone.

Once the clone and relevant tests work, merge `organize-shared-models` into MLIP's
`main`. Return the primary workspace to the merged branch:

```bash
cd /home/ryoji/MLIP
git switch main
git pull --ff-only origin main
git submodule update --init --recursive
```

Future normal clones, run from the directory where you want a new MLIP folder,
can use:

```bash
git clone --recurse-submodules https://github.com/ryoji-kubo/MLIP.git
```

If you cloned without submodules, initialize them from inside MLIP:

```bash
git submodule update --init --recursive
```

Fresh submodule checkouts are normally at detached HEADs. Before making changes,
switch to `mlip-integration`, or create a feature branch from the pinned commit.
Switching to a branch may select a different commit from the one recorded in
MLIP; inspect that difference before updating the parent reference.

Additional `upstream` remotes and local Git settings are not inherited by a new
clone. Repeat the upstream setup from step 4 in each new workspace. `.gitmodules`
records the private origins, not those extra local remotes.

Keep the original folders and backups until your normal experiments, data paths,
and notebook environments work from MLIP. Archiving the old copies can be a later
cleanup task.

## 12. Bring in upstream changes after migration

Work with a clean tree. The following example updates metatrain; choose a new
sync branch name for each update:

```bash
cd /home/ryoji/MLIP/metatrain
git switch mlip-integration
git pull --ff-only origin mlip-integration
git fetch upstream
git switch -c sync/metatrain-2026-09-14
git merge upstream/main
```

If there are conflicts, resolve them, stage the resolutions, and finish with
`git merge --continue`. To abandon an in-progress merge, use `git merge --abort`.
Run the relevant wrapper tests against the shared cores, then publish the branch:

```bash
git push -u origin HEAD
```

Open a PR in **metatrain-private**, targeting `mlip-integration`. Merge it using a
normal merge commit so the upstream ancestry remains recorded. Avoid squash
merging this upstream-sync PR, which would discard that ancestry relationship.

After the PR is merged:

```bash
cd /home/ryoji/MLIP/metatrain
git switch mlip-integration
git pull --ff-only origin mlip-integration

cd /home/ryoji/MLIP
git add metatrain
git diff --cached --submodule=log
git commit -m "Update metatrain to tested upstream integration"
git push origin HEAD
```

Do the equivalent for `equiformer_v3`, using its own sync branch and its own
`upstream/main`. You can update and validate the two frameworks independently.

`git submodule update --init --recursive` checks out the versions recorded by
MLIP. It does not merge original upstream changes. With this setup,
`git submodule update --remote` would follow the private framework branch; it
also does not perform the upstream integration described above.

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

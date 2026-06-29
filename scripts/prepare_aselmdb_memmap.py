#!/usr/bin/env python
"""Convert ASE DB / ASE-LMDB data to metatrain's MemmapDataset layout.

This is intended for validation splits such as the sAlex ``val_30k`` subset used
by the EquiformerV3 MPtrj experiments. The output follows the same layout as
``scripts/prepare_mptrj_memmap.py`` and the memmap example in
``examples/0-beginner/01-data_preparation.py``:

    ns.npy
    na.npy
    a.bin
    x.bin
    c.bin
    e.bin
    f.bin
    s.bin

The input can be one ``.aselmdb`` file or a directory containing one or more
``.aselmdb`` files. ASE-LMDB support comes from the ``ase_db_backends`` package,
which is already available in the Equiformer/FairChem environment on this
machine.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from ase.stress import voigt_6_to_full_3x3_stress

try:
    import ase.db
except ImportError as exc:  # pragma: no cover - import guard for user environments
    raise SystemExit(
        "Could not import ase.db. Install ASE and, for .aselmdb files, "
        "ase_db_backends, or run this script from the Equiformer/FairChem conda env."
    ) from exc


def discover_databases(input_path: Path) -> list[Path]:
    """Return ASE database files to read in deterministic order."""

    if input_path.is_file():
        if input_path.name.endswith("-lock"):
            raise ValueError(f"{input_path} looks like a lock file, not a dataset")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    databases = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix == ".aselmdb" and not path.name.endswith("-lock")
    )
    if not databases:
        raise FileNotFoundError(f"No .aselmdb files found in {input_path}")
    return databases


def iter_rows(databases: list[Path]) -> Iterator[tuple[Path, Any]]:
    """Yield ``(database_path, row)`` from each ASE database."""

    for database in databases:
        db = ase.db.connect(str(database))
        for row in db.select():
            yield database, row


def row_num_atoms(row: Any) -> int:
    """Get the atom count from an ASE DB row without rebuilding atoms when possible."""

    natoms = getattr(row, "natoms", None)
    if natoms is None:
        natoms = len(row.toatoms())
    return int(natoms)


def calculator_result(atoms: Any, key: str) -> Any:
    if atoms.calc is None:
        raise ValueError("ASE Atoms object has no calculator/results attached")

    results = getattr(atoms.calc, "results", None)
    if results is None or key not in results:
        available = sorted(results.keys()) if isinstance(results, dict) else []
        raise KeyError(
            f"Calculator result {key!r} is missing. Available results: {available}"
        )
    return results[key]


def stress_to_full_matrix(stress: Any) -> np.ndarray:
    """Convert ASE-style stress arrays to a full 3x3 float32 matrix."""

    stress_array = np.asarray(stress, dtype=np.float32)
    if stress_array.shape == (3, 3):
        return stress_array
    if stress_array.shape == (9,):
        return stress_array.reshape(3, 3)
    if stress_array.shape == (6,):
        return np.asarray(
            voigt_6_to_full_3x3_stress(stress_array), dtype=np.float32
        )
    raise ValueError(f"Unexpected stress shape: {stress_array.shape}")


def atoms_to_arrays(
    atoms: Any,
    *,
    energy_key: str,
    forces_key: str,
    stress_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32, np.ndarray, np.ndarray]:
    """Extract one ASE Atoms object into arrays matching metatrain's memmap schema."""

    numbers = np.asarray(atoms.numbers, dtype=np.int32)
    positions = np.asarray(atoms.get_positions(), dtype=np.float32)
    cell = np.asarray(atoms.get_cell()[:], dtype=np.float32)
    energy = np.float32(calculator_result(atoms, energy_key))
    forces = np.asarray(calculator_result(atoms, forces_key), dtype=np.float32)
    stress = stress_to_full_matrix(calculator_result(atoms, stress_key))

    if positions.shape != (len(numbers), 3):
        raise ValueError(f"Unexpected positions shape: {positions.shape}")
    if forces.shape != (len(numbers), 3):
        raise ValueError(f"Unexpected forces shape: {forces.shape}")
    if cell.shape != (3, 3):
        raise ValueError(f"Unexpected cell shape: {cell.shape}")
    if stress.shape != (3, 3):
        raise ValueError(f"Unexpected stress shape after conversion: {stress.shape}")

    return numbers, positions, cell, energy, forces, stress


def count_structures(
    databases: list[Path],
    *,
    max_structures: int | None,
    progress_every: int,
) -> np.ndarray:
    natoms: list[int] = []
    for i, (_, row) in enumerate(iter_rows(databases), start=1):
        natoms.append(row_num_atoms(row))
        if progress_every and i % progress_every == 0:
            print(f"counted {i} structures")
        if max_structures is not None and i >= max_structures:
            break
    return np.asarray(natoms, dtype=np.int64)


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Output directory {path} already exists. Pass --force to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_memmap(
    input_path: Path,
    output_dir: Path,
    *,
    energy_key: str,
    forces_key: str,
    stress_key: str,
    max_structures: int | None,
    force: bool,
    progress_every: int,
) -> None:
    databases = discover_databases(input_path)
    print("Reading databases:")
    for database in databases:
        print(f"  {database}")

    print("Pass 1/2: counting structures and atoms")
    natoms = count_structures(
        databases,
        max_structures=max_structures,
        progress_every=progress_every,
    )
    ns = int(len(natoms))
    if ns == 0:
        raise ValueError("No structures found in the input databases")
    na = np.concatenate([[0], np.cumsum(natoms, dtype=np.int64)])

    print(f"Preparing {ns} structures and {int(na[-1])} atoms")
    prepare_output_dir(output_dir, force=force)

    np.save(output_dir / "ns.npy", ns)
    np.save(output_dir / "na.npy", na)

    atomic_numbers_mm = np.memmap(
        output_dir / "a.bin", dtype="int32", mode="w+", shape=(na[-1],)
    )
    positions_mm = np.memmap(
        output_dir / "x.bin", dtype="float32", mode="w+", shape=(na[-1], 3)
    )
    cells_mm = np.memmap(
        output_dir / "c.bin", dtype="float32", mode="w+", shape=(ns, 3, 3)
    )
    energies_mm = np.memmap(
        output_dir / "e.bin", dtype="float32", mode="w+", shape=(ns, 1)
    )
    forces_mm = np.memmap(
        output_dir / "f.bin", dtype="float32", mode="w+", shape=(na[-1], 3)
    )
    stress_mm = np.memmap(
        output_dir / "s.bin", dtype="float32", mode="w+", shape=(ns, 3, 3)
    )

    print("Pass 2/2: writing memmap arrays")
    for i, (database, row) in enumerate(iter_rows(databases)):
        if i >= ns:
            break

        atoms = row.toatoms()
        numbers, positions, cell, energy, forces, stress = atoms_to_arrays(
            atoms,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
        )
        start = na[i]
        stop = na[i + 1]
        if stop - start != len(numbers):
            raise ValueError(
                f"Atom count changed between passes for row {i + 1} in {database}"
            )

        atomic_numbers_mm[start:stop] = numbers
        positions_mm[start:stop] = positions
        cells_mm[i] = cell
        energies_mm[i, 0] = energy
        forces_mm[start:stop] = forces
        stress_mm[i] = stress

        if progress_every and (i + 1) % progress_every == 0:
            print(f"wrote {i + 1} structures")

    for array in [
        atomic_numbers_mm,
        positions_mm,
        cells_mm,
        energies_mm,
        forces_mm,
        stress_mm,
    ]:
        array.flush()

    metadata = {
        "source": str(input_path),
        "databases": [str(database) for database in databases],
        "num_structures": ns,
        "num_atoms": int(na[-1]),
        "energy_key": energy_key,
        "forces_key": forces_key,
        "stress_key": stress_key,
        "stress_conversion": (
            "none; input stress is assumed to be ASE stress in eV/A^3 with ASE "
            "sign convention"
        ),
        "row_order": "database files sorted by name, rows in ASE DB select() order",
        "memmap_keys": {
            "atomic_types": "a",
            "positions": "x",
            "cell": "c",
            "energy": "e",
            "forces": "f",
            "stress": "s",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Finished writing {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to one .aselmdb file or a directory containing .aselmdb files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the metatrain MemmapDataset.",
    )
    parser.add_argument(
        "--energy-key",
        default="energy",
        help="ASE calculator result key to use for energy.",
    )
    parser.add_argument(
        "--forces-key",
        default="forces",
        help="ASE calculator result key to use for forces.",
    )
    parser.add_argument(
        "--stress-key",
        default="stress",
        help="ASE calculator result key to use for stress.",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Optional limit for smoke tests or subsets.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5_000,
        help="Print progress every N structures; use 0 to disable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    args = parser.parse_args()

    write_memmap(
        args.input,
        args.output,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        max_structures=args.max_structures,
        force=args.force,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()

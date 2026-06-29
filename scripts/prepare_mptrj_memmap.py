#!/usr/bin/env python
"""Convert MPtrj JSON data to metatrain's MemmapDataset layout.

The output follows the format shown in examples/0-beginner/01-data_preparation.py:

    ns.npy
    na.npy
    a.bin
    x.bin
    c.bin
    e.bin
    f.bin
    s.bin

The MPtrj JSON file is large, so this script streams it instead of calling
json.load(). It intentionally avoids FairChem/Equiformer-specific dataset classes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterator

import ase.units
import numpy as np
from ase.data import atomic_numbers
from ase.stress import voigt_6_to_full_3x3_stress


KBAR_TO_EV_PER_A3_ASE_STRESS = -0.1 * ase.units.GPa


class JsonStream:
    """Small streaming JSON parser for MPtrj's nested-object layout."""

    def __init__(self, path: Path, chunk_size: int = 1024 * 1024) -> None:
        self.file = path.open("r", encoding="utf-8")
        self.chunk_size = chunk_size
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.pos = 0
        self.eof = False

    def close(self) -> None:
        self.file.close()

    def _fill(self) -> None:
        if self.eof:
            return
        chunk = self.file.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def _compact(self) -> None:
        if self.pos > self.chunk_size:
            self.buffer = self.buffer[self.pos :]
            self.pos = 0

    def _ensure(self, n: int = 1) -> bool:
        while len(self.buffer) - self.pos < n and not self.eof:
            self._fill()
        return len(self.buffer) - self.pos >= n

    def skip_ws(self) -> None:
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer) or self.eof:
                self._compact()
                return
            self._fill()

    def peek(self) -> str | None:
        self.skip_ws()
        if not self._ensure(1):
            return None
        return self.buffer[self.pos]

    def expect(self, char: str) -> None:
        found = self.peek()
        if found != char:
            raise ValueError(f"Expected {char!r}, found {found!r}")
        self.pos += 1
        self._compact()

    def read_value(self) -> Any:
        self.skip_ws()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self._fill()
            else:
                self.pos = end
                self._compact()
                return value

    def read_string(self) -> str:
        value = self.read_value()
        if not isinstance(value, str):
            raise ValueError(f"Expected a JSON string, found {type(value).__name__}")
        return value


def iter_mptrj_entries(path: Path) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(material_id, frame_id, frame_data)`` from the MPtrj JSON file."""

    stream = JsonStream(path)
    try:
        stream.expect("{")
        first_material = True
        while True:
            char = stream.peek()
            if char == "}":
                stream.expect("}")
                break
            if not first_material:
                stream.expect(",")

            material_id = stream.read_string()
            stream.expect(":")
            stream.expect("{")

            first_frame = True
            while True:
                char = stream.peek()
                if char == "}":
                    stream.expect("}")
                    break
                if not first_frame:
                    stream.expect(",")

                frame_id = stream.read_string()
                stream.expect(":")
                frame = stream.read_value()
                if not isinstance(frame, dict):
                    raise ValueError(
                        f"Expected frame {material_id}/{frame_id} to be an object"
                    )
                yield material_id, frame_id, frame
                first_frame = False

            first_material = False
    finally:
        stream.close()


def _stress_to_full_matrix(stress: Any) -> np.ndarray:
    stress_array = np.asarray(stress, dtype=np.float32)
    if stress_array.shape == (3, 3):
        full_stress = stress_array
    elif stress_array.shape == (9,):
        full_stress = stress_array.reshape(3, 3)
    elif stress_array.shape == (6,):
        full_stress = voigt_6_to_full_3x3_stress(stress_array)
    else:
        raise ValueError(f"Unexpected stress shape: {stress_array.shape}")
    return np.asarray(full_stress, dtype=np.float32) * KBAR_TO_EV_PER_A3_ASE_STRESS


def frame_to_arrays(
    frame: dict[str, Any],
    *,
    energy_key: str,
    force_key: str,
    stress_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.float32, np.ndarray]:
    """Extract one MPtrj frame into arrays matching metatrain's memmap schema."""

    structure = frame["structure"]
    sites = structure["sites"]
    numbers = np.asarray(
        [atomic_numbers[site["species"][0]["element"]] for site in sites],
        dtype=np.int32,
    )
    positions = np.asarray([site["xyz"] for site in sites], dtype=np.float32)
    cell = np.asarray(structure["lattice"]["matrix"], dtype=np.float32)
    energy = np.float32(frame[energy_key])
    forces = np.asarray(frame[force_key], dtype=np.float32)
    stress = _stress_to_full_matrix(frame[stress_key])

    if positions.shape != (len(numbers), 3):
        raise ValueError(f"Unexpected positions shape: {positions.shape}")
    if forces.shape != (len(numbers), 3):
        raise ValueError(f"Unexpected forces shape: {forces.shape}")
    if cell.shape != (3, 3):
        raise ValueError(f"Unexpected cell shape: {cell.shape}")
    if stress.shape != (3, 3):
        raise ValueError(f"Unexpected stress shape after conversion: {stress.shape}")

    return numbers, positions, cell, energy, forces, stress


def count_frames(
    input_path: Path,
    *,
    energy_key: str,
    force_key: str,
    stress_key: str,
    max_structures: int | None,
    progress_every: int,
) -> np.ndarray:
    natoms: list[int] = []
    for i, (_, _, frame) in enumerate(iter_mptrj_entries(input_path), start=1):
        numbers, _, _, _, _, _ = frame_to_arrays(
            frame,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
        )
        natoms.append(len(numbers))
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
    force_key: str,
    stress_key: str,
    max_structures: int | None,
    force: bool,
    progress_every: int,
) -> None:
    print("Pass 1/2: counting structures and atoms")
    natoms = count_frames(
        input_path,
        energy_key=energy_key,
        force_key=force_key,
        stress_key=stress_key,
        max_structures=max_structures,
        progress_every=progress_every,
    )
    ns = int(len(natoms))
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
    for i, (material_id, frame_id, frame) in enumerate(iter_mptrj_entries(input_path)):
        if i >= ns:
            break
        numbers, positions, cell, energy, forces, stress = frame_to_arrays(
            frame,
            energy_key=energy_key,
            force_key=force_key,
            stress_key=stress_key,
        )
        start = na[i]
        stop = na[i + 1]
        if stop - start != len(numbers):
            raise ValueError(
                f"Atom count changed between passes for {material_id}/{frame_id}"
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
        "num_structures": ns,
        "num_atoms": int(na[-1]),
        "energy_key": energy_key,
        "force_key": force_key,
        "stress_key": stress_key,
        "stress_conversion": "stress * (-0.1 * ase.units.GPa), kBar to eV/A^3 with ASE sign convention",
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
        default=Path(
            "/home/ryoji/equivarient/equiformer_v3/dataset/mptrj/"
            "MPtrj_2022.9_full.json"
        ),
        help="Path to MPtrj_2022.9_full.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the metatrain MemmapDataset.",
    )
    parser.add_argument(
        "--energy-key",
        default="uncorrected_total_energy",
        help="MPtrj frame key to use for energy.",
    )
    parser.add_argument("--force-key", default="force", help="MPtrj force key.")
    parser.add_argument("--stress-key", default="stress", help="MPtrj stress key.")
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Optional limit for smoke tests or subsets.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
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
        force_key=args.force_key,
        stress_key=args.stress_key,
        max_structures=args.max_structures,
        force=args.force,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()

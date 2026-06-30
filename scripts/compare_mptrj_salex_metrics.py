#!/usr/bin/env python
"""Compare sAlex validation MAEs from metatrain and Equiformer/FairChem logs.

This intentionally compares the closest like-for-like validation metrics:

- energy: metatrain "validation energy MAE (per atom)" vs FairChem
  "val/energy_per_atom_mae"
- force: metatrain "validation non_conservative_force MAE (per atom)" vs
  FairChem "val/forces_mae"
- stress: metatrain "validation non_conservative_stress MAE" from the training
  loop vs FairChem "val/stress_mae"

FairChem summary values are in eV, eV/A, and eV/A^3, so this script multiplies
them by 1000 to match metatrain's meV-style log units.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METATRAIN_COLUMNS = {
    "energy": "validation energy MAE (per atom)",
    "force": "validation non_conservative_force MAE (per atom)",
    "stress": "validation non_conservative_stress MAE",
}

EQUIFORMER_KEYS = {
    "energy": "val/energy_per_atom_mae",
    "force": "val/forces_mae",
    "stress": "val/stress_mae",
}

UNITS = {
    "energy": "meV/atom",
    "force": "meV/A",
    "stress": "meV/A^3",
}


def _is_epoch(value: str) -> bool:
    try:
        int(value.strip())
    except ValueError:
        return False
    return True


def read_last_metatrain_epoch(csv_path: Path) -> dict[str, float]:
    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    last_data_row = None
    for row in rows[1:]:
        if row and _is_epoch(row[0]):
            last_data_row = row

    if last_data_row is None:
        raise ValueError(f"No epoch rows found in {csv_path}")

    values = {}
    for name, column in METATRAIN_COLUMNS.items():
        idx = header.index(column)
        values[name] = float(last_data_row[idx])
    return values


def read_equiformer_summary(summary_path: Path) -> dict[str, float]:
    summary = json.loads(summary_path.read_text())
    return {
        name: float(summary[key]) * 1000.0 for name, key in EQUIFORMER_KEYS.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metatrain-csv",
        type=Path,
        default=Path("outputs/2026-06-29/07-47-01/train.csv"),
    )
    parser.add_argument(
        "--equiformer-summary",
        type=Path,
        default=Path(
            "/home/ryoji/equivarient/equiformer_v3/logs/omat24/equiformer_v3/"
            "logs/wandb/2026-05-27-14-24-00-mptrj_direct_160k_NoDeNS/wandb/"
            "run-20260527_142306-2026-05-27-14-24-00-mptrj_direct_160k_"
            "NoDeNS/files/wandb-summary.json"
        ),
    )
    args = parser.parse_args()

    metatrain = read_last_metatrain_epoch(args.metatrain_csv)
    equiformer = read_equiformer_summary(args.equiformer_summary)

    print("Comparable sAlex validation MAEs")
    print(f"{'metric':<8} {'metatrain':>12} {'equiformer':>12} {'unit':>10}")
    for metric in ["energy", "force", "stress"]:
        print(
            f"{metric:<8} "
            f"{metatrain[metric]:>12.4f} "
            f"{equiformer[metric]:>12.4f} "
            f"{UNITS[metric]:>10}"
        )


if __name__ == "__main__":
    main()

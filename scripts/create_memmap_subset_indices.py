#!/usr/bin/env python
"""Create a random subset index file for a metatrain MemmapDataset.

The output is a plain text file with one integer index per line, suitable for
``training_set.indices`` / ``validation_set.indices`` in metatrain option files.
This is useful when the full memmap dataset is already prepared and we want a
fast smaller training run without physically copying structures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the metatrain MemmapDataset directory containing ns.npy.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of structure indices to sample without replacement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible subsampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the text file containing one index per line.",
    )
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help=(
            "Keep sampled indices in random order. By default, sampled indices are "
            "sorted for more sequential first-pass memmap reads."
        ),
    )
    args = parser.parse_args()

    ns_path = args.dataset / "ns.npy"
    if not ns_path.exists():
        raise FileNotFoundError(f"Could not find {ns_path}")

    num_structures = int(np.load(ns_path))
    if args.num_samples < 0:
        raise ValueError("--num-samples must be non-negative")
    if args.num_samples > num_structures:
        raise ValueError(
            f"Requested {args.num_samples} samples, but dataset only has "
            f"{num_structures} structures"
        )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(num_structures, size=args.num_samples, replace=False)
    if not args.shuffle_output:
        indices.sort()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, indices, fmt="%d")
    print(
        f"Wrote {len(indices)} indices from {num_structures} structures to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()

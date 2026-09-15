import torch


def cartesian_to_fractional_dense(
    pos_dense: torch.Tensor,
    cell: torch.Tensor,
    wrap: bool = True,
) -> torch.Tensor:
    # Same convention as pymatgen Lattice.get_fractional_coords for row-vector
    # Cartesian positions and a row-stacked 3x3 lattice matrix.
    frac_pos = torch.linalg.solve(
        cell.transpose(1, 2),
        pos_dense.transpose(1, 2),
    ).transpose(1, 2)
    if wrap:
        frac_pos = torch.remainder(frac_pos, 1.0)
    return frac_pos

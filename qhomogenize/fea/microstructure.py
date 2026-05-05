"""
Microstructure generators: classical numpy reference for the indicator p_e.

For the POC the canonical test microstructure is a single circular inclusion
on a Cartesian L x L mesh.  Element e is "solid" (p_e = 1) iff its center
lies inside the disk; "void" (p_e = 0) otherwise.

Conventions
-----------
- Element index e in [0, L^2) is in column-major order: e = i_x + L * i_y,
  matching the Andreassen-Andreasen FE assembly used in
  FEA2DHomogenize.py.  This convention is used everywhere in qhomogenize.
- Element CENTER for index (i_x, i_y) is at ((i_x + 0.5) / L, (i_y + 0.5) / L)
  on the unit cell [0,1]^2.

The disk parameters (cx, cy, r) are defined on this unit-cell scale.
"""

from __future__ import annotations

import numpy as np


def disk_indicator_array(L: int, cx: float = 0.5, cy: float = 0.5, r: float = 0.3) -> np.ndarray:
    """Return the L x L binary indicator array p[i_y, i_x] for a circular inclusion.

    Element (i_x, i_y) is solid iff its center is strictly inside the disk
    of radius r centered at (cx, cy) on the unit cell.

    Parameters
    ----------
    L : int
        Mesh size (number of elements per side).  L^2 elements total.
    cx, cy : float, default 0.5, 0.5
        Disk center on the unit cell [0,1]^2.
    r : float, default 0.3
        Disk radius on the unit cell.

    Returns
    -------
    p : ndarray, shape (L, L), dtype int
        p[i_y, i_x] = 1 if element (i_x, i_y) is solid, 0 otherwise.
        First index is row (y), second is column (x); this matches the
        convention of generate_microstructure in MicrostructureExamples.py.
    """
    ix = np.arange(L)
    iy = np.arange(L)
    Ix, Iy = np.meshgrid(ix, iy, indexing="xy")  # Ix[iy,ix] = ix; Iy[iy,ix] = iy
    cx_pix = (Ix + 0.5) / L
    cy_pix = (Iy + 0.5) / L
    p = ((cx_pix - cx) ** 2 + (cy_pix - cy) ** 2 < r ** 2).astype(int)
    return p


def disk_indicator_grid_units(L: int, cx_unit: float = 0.5, cy_unit: float = 0.5, r_unit: float = 0.3):
    """Convert disk parameters from unit-cell coordinates to integer grid units.

    The quantum indicator-oracle comparator works in integer grid coordinates
    (i_x, i_y) in [0, L) and tests

        (2 i_x + 1 - 2 L cx)^2 + (2 i_y + 1 - 2 L cy)^2 < (2 L r)^2

    after multiplying through by (2 L)^2 to clear the half-integer center
    offset and avoid any floating-point arithmetic in the quantum circuit.

    This routine returns the integer constants (Cx, Cy, R2) such that the
    classical comparator becomes (2 i_x + 1 - Cx)^2 + (2 i_y + 1 - Cy)^2 < R2,
    where Cx = round(2 L cx), Cy = round(2 L cy), R2 = round((2 L r)^2).

    Returns
    -------
    Cx, Cy, R2 : int
    """
    Cx = int(round(2 * L * cx_unit))
    Cy = int(round(2 * L * cy_unit))
    R2 = int(round((2 * L * r_unit) ** 2))
    return Cx, Cy, R2


def p_vector(L: int, cx: float = 0.5, cy: float = 0.5, r: float = 0.3) -> np.ndarray:
    """Return the L^2-length p-vector indexed by element-major index e = i_x + L*i_y.

    This is the form consumed by the block-diagonal mask M_p = diag(p) ⊗ I_8.
    """
    p_grid = disk_indicator_array(L, cx, cy, r)  # shape (L, L), [iy, ix]
    # element-major: e = ix + L*iy  =>  flatten in column-major (Fortran) order on (iy, ix)
    # i.e. iterate ix slowest, iy fastest.  But our grid is [iy, ix].
    # Equivalent: take p_grid.T and flatten in row-major.
    p_flat = p_grid.T.flatten(order="C")
    return p_flat


def p_vector_from_array(micro: np.ndarray) -> np.ndarray:
    """Convert a 2D microstructure array (shape (L, L)) to the element-major p-vector.

    Adapter for `MicrostructureExamples.generate_microstructure(L, L, type=..., volume_fraction=...)`,
    which produces an `(L, L)` binary array indexed as `micro[i_x, i_y]` (row=x, col=y under
    its `np.meshgrid(x, y, indexing='ij')` convention).

    Note the convention difference: `MicrostructureExamples` uses indexing='ij' (so the
    first index is x and the second is y), while `disk_indicator_array` uses indexing='xy'
    (so the first index is y).  This adapter handles the transposition.

    Parameters
    ----------
    micro : ndarray, shape (L, L)
        Binary microstructure array.  Either {0, 1} integer or {0.0, 1.0} float.

    Returns
    -------
    p : ndarray, shape (L^2,), dtype int
        Element-major p-vector with e = i_x + L * i_y.
    """
    if micro.ndim != 2 or micro.shape[0] != micro.shape[1]:
        raise ValueError(f"Expected square 2D array, got shape {micro.shape}")
    L = micro.shape[0]
    # MicrostructureExamples.generate_microstructure uses indexing='ij' so micro[i_x, i_y].
    # Our element-major flattening is e = i_x + L * i_y, which corresponds to flattening
    # with i_x varying FASTEST.  For a numpy array with shape (L, L) and indexing micro[i_x, i_y],
    # row-major (C) flattening goes (i_x=0, i_y=0), (i_x=0, i_y=1), ..., which has i_y fastest.
    # We want i_x fastest, so flatten in column-major (Fortran) order.
    return (micro.astype(int)).flatten(order="F")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for L in (4, 8, 16):
        p = disk_indicator_array(L, cx=0.5, cy=0.5, r=0.3)
        Cx, Cy, R2 = disk_indicator_grid_units(L, 0.5, 0.5, 0.3)
        # Verify the integer-grid comparator gives the same result.
        ix = np.arange(L)
        iy = np.arange(L)
        Ix, Iy = np.meshgrid(ix, iy, indexing="xy")
        p_check = ((2 * Ix + 1 - Cx) ** 2 + (2 * Iy + 1 - Cy) ** 2 < R2).astype(int)
        assert np.array_equal(p, p_check), f"L={L}: integer-grid comparator mismatch"
        n_solid = int(p.sum())
        vol_frac = n_solid / (L * L)
        # disk of r=0.3 has area pi*r^2 ~ 0.283
        print(f"L={L}: solid={n_solid}/{L*L} (vol frac ~ {vol_frac:.3f}), Cx={Cx}, Cy={Cy}, R2={R2}")

    # Print a small disk
    p = disk_indicator_array(8, cx=0.5, cy=0.5, r=0.3)
    print("\nL=8 disk:")
    for row in p:
        print("  " + " ".join("#" if v else "." for v in row))

    # element-major flattening sanity
    pv = p_vector(4, cx=0.5, cy=0.5, r=0.3)
    assert pv.shape == (16,), f"p_vector shape: {pv.shape}"

    # Adapter sanity check: a manual (L, L) array round-trips.
    L = 4
    micro = np.zeros((L, L), dtype=int)
    # Manually mark elements (i_x=1, i_y=2) and (i_x=2, i_y=2):
    micro[1, 2] = 1
    micro[2, 2] = 1
    pv2 = p_vector_from_array(micro)
    # element-major e = i_x + L * i_y, so (1, 2) -> e=9, (2, 2) -> e=10.
    expected = np.zeros(16, dtype=int)
    expected[9] = 1
    expected[10] = 1
    assert np.array_equal(pv2, expected), f"adapter mismatch: {pv2} vs {expected}"
    print("\nOK")

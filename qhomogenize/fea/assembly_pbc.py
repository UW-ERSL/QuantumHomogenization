"""
Finite-element assembly map under periodic boundary conditions on a Cartesian mesh.

Two views of the same map are provided:

1. **Numpy / scipy.sparse form**: an explicit (N_local x N_global) {0,1}-valued
   matrix `A` with N_local = 8 * L^2 (8 DOFs per bilinear-quad element), and
   N_global = 2 * L^2 (2 DOFs per node, with PBC identifying boundary nodes).

2. **Arithmetic form**: closed-form integer functions
       global_dof(e, ell) -> j_global
   where e in [0, L^2) is the element index (column-major: e = i_x + L*i_y),
   ell in [0, 8) is the local-DOF index within the element, and j_global in
   [0, 2*L^2) is the corresponding global DOF index after PBC.  This form
   is the spec for the quantum oracles O_c, O_r in task1.assembly_oracles.

Local-DOF convention (matches Andreassen-Andreasen / FEA2DHomogenize.py):
    Element node order: lower-left, lower-right, upper-right, upper-left.
    DOF order: (u_x, u_y) at each node.
    So ell = 0..7 corresponds to:
        ell=0: corner 0 (LL), u_x        local_node = 0, dim = 0
        ell=1: corner 0 (LL), u_y        local_node = 0, dim = 1
        ell=2: corner 1 (LR), u_x        local_node = 1, dim = 0
        ell=3: corner 1 (LR), u_y        local_node = 1, dim = 1
        ell=4: corner 2 (UR), u_x        local_node = 2, dim = 0
        ell=5: corner 2 (UR), u_y        local_node = 2, dim = 1
        ell=6: corner 3 (UL), u_x        local_node = 3, dim = 0
        ell=7: corner 3 (UL), u_y        local_node = 3, dim = 1

For element (i_x, i_y), the four corner offsets (in (dx, dy)) are
    LL=(0,0), LR=(1,0), UR=(1,1), UL=(0,1)
and the global node coordinate (under PBC, mod L) is
    (n_x, n_y) = ((i_x + dx) % L, (i_y + dy) % L).
The global node index (column-major) is n = n_x + L * n_y, and the global DOF
index for dimension d in {0, 1} is j_global = 2 * n + d.

This is the closed-form expression that the quantum O_c / O_r oracles
implement.  It is implemented in pure Python here for ground-truth comparison.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


# Local-DOF table: for ell in 0..7, (corner_index, dim_index)
_LOCAL_DOF_TABLE = np.array([
    [0, 0], [0, 1],   # LL
    [1, 0], [1, 1],   # LR
    [2, 0], [2, 1],   # UR
    [3, 0], [3, 1],   # UL
], dtype=int)

# Corner offsets (dx, dy) for corners 0..3 in the Andreassen-Andreasen convention
_CORNER_OFFSETS = np.array([
    [0, 0],   # LL
    [1, 0],   # LR
    [1, 1],   # UR
    [0, 1],   # UL
], dtype=int)


def n_dofs(L: int) -> tuple[int, int]:
    """Return (N_global, N_local) for an L x L mesh.

    N_global = 2 * L^2 (2 DOFs per node, L^2 unique nodes under PBC).
    N_local  = 8 * L^2 (8 DOFs per element, L^2 elements).
    """
    return 2 * L * L, 8 * L * L


def global_dof(e: int, ell: int, L: int) -> int:
    """Closed-form arithmetic form of A: map (e, ell) -> j_global under PBC.

    Parameters
    ----------
    e : int in [0, L^2)
        Element index, column-major: e = i_x + L * i_y.
    ell : int in [0, 8)
        Local-DOF index within the element.
    L : int
        Mesh size.

    Returns
    -------
    j_global : int in [0, 2 * L^2)
        Global DOF index after PBC identification.
    """
    i_x = e % L
    i_y = e // L
    corner = ell // 2
    dim = ell % 2
    dx, dy = _CORNER_OFFSETS[corner]
    n_x = (i_x + dx) % L
    n_y = (i_y + dy) % L
    n = n_x + L * n_y
    return 2 * n + dim


def assembly_matrix(L: int) -> csr_matrix:
    """Build the explicit sparse (N_local x N_global) assembly matrix A.

    A[k, j] = 1 iff k = 8 * e + ell maps to j = global_dof(e, ell, L).
    Each row of A has exactly one 1, so A is {0,1}-valued with row sparsity 1.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (8 * L^2, 2 * L^2)
    """
    N_global, N_local = n_dofs(L)
    rows = []
    cols = []
    for e in range(L * L):
        for ell in range(8):
            k = 8 * e + ell
            j = global_dof(e, ell, L)
            rows.append(k)
            cols.append(j)
    data = np.ones(len(rows), dtype=float)
    A = coo_matrix((data, (rows, cols)), shape=(N_local, N_global)).tocsr()
    return A


def column_sparsities(L: int) -> tuple[int, int]:
    """Return (S_c, S_r) for A.

    S_r = max row sparsity.  Each row of A has exactly one 1, so S_r = 1.
    S_c = max column sparsity.  Each global DOF is shared by the four elements
          incident at the corresponding node, so S_c = 4 in 2D.
    """
    return 4, 1


# ---------------------------------------------------------------------------
# Quick test: shape, S_c, S_r, A^T A diagonal (each global DOF used exactly 4 times)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for L in (2, 4, 8):
        A = assembly_matrix(L)
        N_global, N_local = n_dofs(L)
        assert A.shape == (N_local, N_global), f"L={L}: shape {A.shape}"

        # Row sparsity = 1 for every row
        row_sums = np.array(A.sum(axis=1)).ravel()
        assert np.all(row_sums == 1), f"L={L}: row sums {row_sums}"

        # Column sparsity = 4 for every column (4 elements share each node)
        col_sums = np.array(A.sum(axis=0)).ravel()
        assert np.all(col_sums == 4), f"L={L}: col sums {col_sums[:8]}..."

        # A^T A = 4 * I
        AtA = (A.T @ A).toarray()
        assert np.allclose(AtA, 4 * np.eye(N_global)), f"L={L}: A^T A != 4 I"

        Sc, Sr = column_sparsities(L)
        assert Sc == 4 and Sr == 1

        print(f"L={L}: A shape {A.shape}, S_c={Sc}, S_r={Sr}, A^T A = 4 I  OK")

    # Spot-check global_dof against the assembled A
    L = 4
    A = assembly_matrix(L)
    A_dense = A.toarray()
    for e in range(L * L):
        for ell in range(8):
            j = global_dof(e, ell, L)
            assert A_dense[8 * e + ell, j] == 1.0
            # And no other entry in that row
            assert A_dense[8 * e + ell].sum() == 1.0
    print("global_dof closed form matches assembly_matrix at L=4   OK")

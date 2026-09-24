"""macroload -- the macroscopic-strain load of the periodic two-phase cell.

Computational homogenization drives the cell by a constant macroscopic strain
eps0 = (exx, eyy, gxy) and solves for the periodic fluctuation w,

    K^chi w = F(eps0),      F(eps0) = sum_e E_e A_e^T f^e eps0,
    f^e = int_{Omega_e} B^T D dOmega       (8 x 3, at unit modulus).

The element stiffness of a Q4 element is scale free in two dimensions, since
B ~ h^-1 and dOmega ~ h^2, so the operator of the block-encoding paper is
unchanged on a unit cell with h = 1/N. The load is not: f^e carries one factor
of h, and the cell volume is 1.

The structural fact the preconditioning argument rests on is that the load is
supported on the phase interface. At a node whose four incident elements carry
the same modulus E, the four element contributions are the homogeneous-cell
contributions scaled by that one modulus, and those sum to zero because a
constant strain state is in equilibrium on a periodic mesh. So

    F_i = 0   at every node i with chi constant on its four incident elements,

identically, at every contrast and every N, in both phases at once. The load
therefore cannot excite displacements supported strictly inside either phase,
which is what leaves the compliant-phase interior modes unexcited.

    from .macroload import load, interior_nodes, HYDROSTATIC, SHEAR
    F = load(m=5, chi=chi, E1=1e-4, E2=1.0, eps0=HYDROSTATIC)

Layout follows pyblockencode: degree of freedom d * N^2 + y * N + x, and
chi[ex, ey] is the element indicator, 1 in the inclusion.
"""
from __future__ import annotations

import numpy as np

from pyblockencode import CORNERS

HYDROSTATIC = np.array([1.0, 1.0, 0.0])
SHEAR = np.array([0.0, 0.0, 1.0])


# ==========================================================================
# 1. the element load
# ==========================================================================
def element(nu: float = 0.3, h: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Plane-stress Q4 stiffness (scale free) and load f^e = int B^T D.

    Returns (8 x 8, 8 x 3). The stiffness is independent of h in 2D; the load
    is proportional to h.
    """
    D = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu ** 2)
    g = 1 / np.sqrt(3)
    Ke, fe = np.zeros((8, 8)), np.zeros((8, 3))
    for xi in (-g, g):
        for eta in (-g, g):
            dN = np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                           [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]]) / 4
            J = (h / 2) * np.eye(2)
            dNx = np.linalg.solve(J, dN)
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2 * i], B[1, 2 * i + 1] = dNx[0, i], dNx[1, i]
                B[2, 2 * i], B[2, 2 * i + 1] = dNx[1, i], dNx[0, i]
            Ke += B.T @ D @ B * np.linalg.det(J)
            fe += B.T @ D * np.linalg.det(J)
    return Ke, fe


# ==========================================================================
# 2. assembly
# ==========================================================================
def load(m: int, chi: np.ndarray, E1: float, E2: float, eps0: np.ndarray,
         nu: float = 0.3) -> np.ndarray:
    """F(eps0) = sum_e E_e A_e^T f^e eps0, on the unit cell with h = 1/N."""
    N = 2 ** m
    _, fe = element(nu, 1.0 / N)
    v = fe @ np.asarray(eps0, float)
    E = E2 + (E1 - E2) * chi                                # E[ex, ey]
    F = np.zeros(2 * N * N)
    for li, (ox, oy) in enumerate(CORNERS):
        ex, ey = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        idx = ((ey + oy) % N) * N + (ex + ox) % N
        for d in (0, 1):
            np.add.at(F, d * N * N + idx.ravel(),
                      (E * v[2 * li + d]).ravel())
    return F


def stiffness(m: int, chi: np.ndarray, E1: float, E2: float,
              nu: float = 0.3) -> np.ndarray:
    """The two-phase operator K^chi, assembled densely for verification."""
    from pyblockencode import _assemble_periodic
    Ke, _ = element(nu, 1.0 / 2 ** m)
    K = _assemble_periodic(Ke, 2 ** m, 2, chi=chi, E1=E1, E2=E2)
    return 0.5 * (K + K.T)


# ==========================================================================
# 3. the phase-interior nodes
# ==========================================================================
def interior_nodes(chi: np.ndarray) -> np.ndarray:
    """Nodes whose four incident elements carry the same modulus.

    Node (ix, iy) touches the elements at offsets (0,0), (-1,0), (0,-1) and
    (-1,-1), which is ELEM_OFFSETS of the block-encoding paper read from the
    node. Returns a boolean array indexed [ix, iy].
    """
    s = chi > 0.5
    four = [np.roll(np.roll(s, -a, axis=0), -b, axis=1)
            for a in (0, -1) for b in (0, -1)]
    return np.logical_or.reduce(four) == np.logical_and.reduce(four)


def interior_dofs(chi: np.ndarray) -> np.ndarray:
    """Degree-of-freedom indices at the phase-interior nodes."""
    N = chi.shape[0]
    ix, iy = np.nonzero(interior_nodes(chi))
    return np.concatenate([d * N * N + iy * N + ix for d in (0, 1)])


# ==========================================================================
if __name__ == "__main__":
    from pyblockencode import _chi_square

    print("----- output --------------------------------------------------")
    print(f"{'m':>3} {'r':>8} {'interior':>9} {'max |F| there':>14} "
          f"{'max |F| all':>12} {'homogeneous':>12}")
    for m in (3, 4, 5):
        chi = _chi_square(0.25, m)
        idx = interior_dofs(chi)
        for r in (1e1, 1e4, 1e8):
            F = load(m, chi, 1.0 / r, 1.0, HYDROSTATIC)
            F0 = load(m, chi, 1.0, 1.0, HYDROSTATIC)
            print(f"{m:>3} {r:>8.0e} {len(idx) // 2:>9} "
                  f"{np.abs(F[idx]).max():>14.2e} {np.abs(F).max():>12.5f} "
                  f"{np.abs(F0).max():>12.2e}")
    print("---------------------------------------------------------------")

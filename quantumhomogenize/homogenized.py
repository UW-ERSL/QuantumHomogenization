"""moduli -- the homogenized tensor as quadratic forms in the resolvent.

The homogenized tensor separates into a term available classically and a
correction that is the only quantum-hard object,

    C^H = C^Voigt - |Omega|^{-1} F^T (K^chi)^+ F,
    C^Voigt = <E> D_1,      <E> = v_f E_1 + (1 - v_f) E_2,

with D_1 the plane-stress matrix at unit modulus. The classical term is the
energy of the affine field, which is exact on a Q4 element because a constant
strain state is representable, so no element solve is needed.

Contracting with a test strain collapses the correction to a single scalar:

    K_H = (1/4) 1^T C^H 1,   1 = (1,1,0),      G_H = e3^T C^H e3.

Each is one quadratic form F^T (K^chi)^+ F in the resolvent, so the readout is
two scalars independent of N, and the complexity lower bound in
xi = ||A^{-1} b|| of Tong et al. does not apply: no solution state is prepared
at either end. Full anisotropic C^H in two dimensions needs six forms, three
diagonal and three by polarization.

The ratio of correction to modulus is the precision amplification factor of the
readout, not an error: relative precision eps on a modulus needs relative
precision eps * (C^H / correction) on the quantum-computed term. The ratio
stays below about 1.6 up to contrast 100 on a connected stiff phase.

    from .homogenized import homogenized, bulk_shear
    KH, GH, ratios = bulk_shear(m=5, chi=chi, r=10)

The stiff phase must be the connected one. Putting the stiff phase in the
inclusion instead makes the connected phase vanishingly soft at high contrast,
the modulus collapses, and the correction comes to dominate it; that
configuration is outside the declared scope.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .macroload import HYDROSTATIC, SHEAR, load, stiffness

TOL = 1e-9


def plane_stress(nu: float = 0.3) -> np.ndarray:
    """D_1, the plane-stress matrix at unit modulus."""
    return np.array([[1, nu, 0], [nu, 1, 0],
                     [0, 0, (1 - nu) / 2]]) / (1 - nu ** 2)


def voigt(chi: np.ndarray, E1: float, E2: float,
          nu: float = 0.3) -> np.ndarray:
    """C^Voigt = <E> D_1, the energy of the affine field."""
    vf = float(chi.mean())
    return (vf * E1 + (1 - vf) * E2) * plane_stress(nu)


def correction(m: int, chi: np.ndarray, E1: float, E2: float,
               eps0: np.ndarray, nu: float = 0.3) -> float:
    """F^T (K^chi)^+ F on the unit cell, solved off the rigid translations."""
    K = stiffness(m, chi, E1, E2, nu)
    F = load(m, chi, E1, E2, eps0, nu)
    w, V = eigh(K)
    keep = w > TOL * w.max()
    c = V[:, keep].T @ F
    return float((c ** 2 / w[keep]).sum())


def homogenized(m: int, chi: np.ndarray, E1: float, E2: float,
                nu: float = 0.3) -> np.ndarray:
    """Full C^H, three diagonal forms plus three by polarization."""
    basis = np.eye(3)
    C0 = voigt(chi, E1, E2, nu)
    d = [correction(m, chi, E1, E2, basis[i], nu) for i in range(3)]
    C = C0.copy()
    for i in range(3):
        C[i, i] -= d[i]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        s = correction(m, chi, E1, E2, basis[i] + basis[j], nu)
        C[i, j] = C[j, i] = C0[i, j] - 0.5 * (s - d[i] - d[j])
    return C


def bulk_shear(m: int, chi: np.ndarray, r: float, nu: float = 0.3):
    """K_H and G_H from one quadratic form each, with the correction ratios."""
    E1, E2 = 1.0 / r, 1.0
    C0 = voigt(chi, E1, E2, nu)
    out = []
    for eps0, scale in ((HYDROSTATIC, 0.25), (SHEAR, 1.0)):
        v = scale * float(eps0 @ C0 @ eps0)
        c = scale * correction(m, chi, E1, E2, eps0, nu)
        out.append((v - c, c / (v - c)))
    return out[0][0], out[1][0], (out[0][1], out[1][1])


# ==========================================================================
if __name__ == "__main__":
    from pyblockencode import _chi_square

    print("----- output --------------------------------------------------")
    print("Two scalars from one quadratic form each, against full C^H")
    print(f"{'m':>3} {'r':>6} {'K_H':>10} {'from C^H':>10} {'G_H':>10} "
          f"{'from C^H':>10} {'corr/K_H':>9} {'corr/G_H':>9}")
    for m in (3, 4, 5):
        for r in (2, 10, 100):
            chi = _chi_square(0.25, m)
            KH, GH, ratio = bulk_shear(m, chi, r)
            C = homogenized(m, chi, 1.0 / r, 1.0)
            print(f"{m:>3} {r:>6} {KH:>10.6f} "
                  f"{HYDROSTATIC @ C @ HYDROSTATIC / 4:>10.6f} "
                  f"{GH:>10.6f} {SHEAR @ C @ SHEAR:>10.6f} "
                  f"{ratio[0]:>9.4f} {ratio[1]:>9.4f}")

    print("\nHomogeneous cell reproduces the bulk material")
    chi = _chi_square(0.25, 4)
    C = homogenized(4, chi, 1.0, 1.0)
    print(f"  max |C^H - D_1| = {np.abs(C - plane_stress(0.3)).max():.2e}")
    print("---------------------------------------------------------------")

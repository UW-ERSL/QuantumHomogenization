"""conditioning -- worst-case against effective conditioning, and the
polynomial degree that follows.

The worst case of Proposition 3 is the contrast, and it is attained only on
modes the problem never excites. The macroscopic-strain load vanishes on
phase-interior nodes, so it is supported on the interface, while the small
eigenvalues of M belong to displacements interior to the compliant phase. In
the true-void limit those two supports are disjoint and the orthogonality is
exact; the preconditioner does not interfere, since for v = A^{1/2} y

    < v , A^{-1/2} F > = < y , F > ,

and the conjugation cancels. At finite contrast the overlap stays at the
double-precision floor over many decades, so the truncation is exact to
anything double precision can resolve.

A QSVT approximation to the inverse may therefore be built on the effective
interval rather than the worst-case one. Because the observable is a quadratic
form, the bias is linear in the discarded weight rather than square root:

    | Q - Qhat |  <=  (4 eps' / 3 delta) ||f||^2  +  w_out ( kappa + 4 kappa_eff / 3 ),

and with Q >= ||f||^2 the relative truncation error is about w_out * kappa.

The degree of an odd polynomial approximating 1/x on [delta, 1] to eps is
O(delta^-1 log(1/eps)), so the saving is the ratio of the two condition
numbers. The effective constant is independent of contrast but grows
logarithmically with the mesh, so the degree claim is O(log N), not O(1).

    from .conditioning import table_effective, table_degree
"""
from __future__ import annotations

import numpy as np

from .macroload import HYDROSTATIC, SHEAR
from .precondition import (discarded_weight, kappa, kappa_effective, spectrum)


def qsp_degree(kap: float, eps: float = 1e-6) -> float:
    """Degree of the standard odd approximation to 1/x on [1/kap, 1].

    The Gilyen-Su-Low-Wiebe construction gives O(kappa log(1/eps)); the
    constant is the same for both intervals, so only the ratio is reported.
    """
    return kap * np.log(1.0 / eps)


# ==========================================================================
def table_effective(ms=(3, 4, 5), rhos=(1e1, 1e2, 1e3, 1e4), vf=0.25):
    from pyblockencode import _chi_square

    print("Effective against worst-case conditioning, square inclusion "
          f"vf = {vf}")
    print(f"{'m':>3} {'N':>4} {'rho':>8} {'worst':>10} {'eff hyd':>9} "
          f"{'eff shr':>9} {'degree ratio':>13}")
    for m in ms:
        chi = _chi_square(vf, m)
        for rho in rhos:
            lam, U, Fp = spectrum(m, chi, 1.0 / rho, 1.0)
            kw = kappa(lam)
            kh = kappa_effective(lam, U, Fp[:, 0])
            ks = kappa_effective(lam, U, Fp[:, 1])
            print(f"{m:>3} {2 ** m:>4} {rho:>8.0e} {kw:>10.1f} {kh:>9.3f} "
                  f"{ks:>9.3f} {kw / max(kh, ks):>13.1f}")


def table_truncation(ms=(3, 4), rhos=(1e2, 1e4, 1e6, 1e8), vf=0.25):
    from pyblockencode import _chi_square

    print("\nDiscarded weight and the bias it induces in the readout")
    print(f"{'m':>3} {'rho':>8} {'n_soft':>7} {'w_out':>11} "
          f"{'bias / Q':>11}")
    for m in ms:
        chi = _chi_square(vf, m)
        for rho in rhos:
            lam, U, Fp = spectrum(m, chi, 1.0 / rho, 1.0)
            cut = 10.0 / rho
            w, bias = discarded_weight(lam, U, Fp[:, 0], cut)
            print(f"{m:>3} {rho:>8.0e} {int((lam < cut).sum()):>7} "
                  f"{w:>11.2e} {bias:>11.2e}")


def table_budget(m=5, rho=1e4, vf=0.25):
    from pyblockencode import _chi_square

    print(f"\nInsensitivity to the weight budget, m = {m}, rho = {rho:.0e}")
    lam, U, Fp = spectrum(m, _chi_square(vf, m), 1.0 / rho, 1.0)
    print(f"{'delta':>10} {'eff hyd':>9} {'eff shr':>9}")
    for d in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        print(f"{d:>10.0e} {kappa_effective(lam, U, Fp[:, 0], d):>9.3f} "
              f"{kappa_effective(lam, U, Fp[:, 1], d):>9.3f}")


# ==========================================================================
if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    table_effective()
    table_truncation()
    table_budget()
    print("---------------------------------------------------------------")

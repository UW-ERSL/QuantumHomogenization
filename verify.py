"""verify -- every proposition the paper states, checked in one run.

    python verify.py

Each block asserts the closed form against a measurement before printing, so a
number in the paper cannot go stale relative to the code.
"""
from __future__ import annotations

import numpy as np

from pyblockencode import _chi_square
from quantumhomogenize.conditioning import table_budget, table_effective, table_truncation
from quantumhomogenize.macroload import HYDROSTATIC, SHEAR, interior_dofs, load
from quantumhomogenize.homogenized import bulk_shear, homogenized, plane_stress
from quantumhomogenize.precondition import (isometry_residual, kappa, kappa_effective,
                          spectrum)
from quantumhomogenize.fourier_symbol import (mode_operator_matrix, reference, symbol, symbol_inverse,
                    symbol_inv_sqrt)

NU = 0.3


def prop1(ms=(2, 3, 4)):
    """The mean part is block-circulant with a closed-form invertible symbol."""
    print("Proposition 1: symbol against assembly, and its closed-form inverses")
    print(f"{'m':>3} {'nu':>6} {'symbol':>10} {'inverse':>10} {'sqrt':>10}")
    for m in ms:
        for nu in (0.0, NU, 1 / 3, 0.45):
            S, Si, Sh = symbol(m, nu), symbol_inverse(m, nu), symbol_inv_sqrt(m, nu)
            e1 = np.abs(mode_operator_matrix(S) - reference(m, nu)).max()
            P = np.einsum("xyde,xyef->xydf", Si, S)
            P[0, 0] = np.eye(2)
            e2 = np.abs(P - np.eye(2)).max()
            e3 = np.abs(np.einsum("xyde,xyef->xydf", Sh, Sh) - Si).max()
            assert max(e1, e2, e3) < 1e-12
            print(f"{m:>3} {nu:>6.3f} {e1:>10.2e} {e2:>10.2e} {e3:>10.2e}")


def prop2(ms=(2, 3, 4)):
    print("\nProposition 2: V^dag V = E_0^{-1} P and ||V|| = E_0^{-1/2}")
    print(f"{'m':>3} {'E_0':>7} {'residual':>10} {'||V||':>9} {'target':>9}")
    for m in ms:
        for E0 in (1.0, 7.5):
            res, nv = isometry_residual(m, NU, E0)
            assert res < 1e-12 and abs(nv - E0 ** -0.5) < 1e-12
            print(f"{m:>3} {E0:>7.2f} {res:>10.2e} {nv:>9.5f} "
                  f"{E0 ** -0.5:>9.5f}")


def prop3(ms=(3, 4)):
    print("\nProposition 3: spec(M) = [1/rho, 1] exactly, alpha_M = 1")
    print(f"{'m':>3} {'nu':>6} {'rho':>8} {'lam_min * rho':>14} "
          f"{'lam_max':>9} {'kappa/rho':>10}")
    for m in ms:
        chi = _chi_square(0.25, m)
        for nu in (0.0, NU, 0.45):
            for rho in (1e2, 1e4):
                lam, _, _ = spectrum(m, chi, 1.0 / rho, 1.0, nu)
                nz = lam[lam > 1e-9 * lam.max()]
                assert abs(kappa(lam) / rho - 1) < 1e-8
                assert abs(nz.max() - 1) < 1e-8
                print(f"{m:>3} {nu:>6.2f} {rho:>8.0e} {nz.min() * rho:>14.8f} "
                      f"{nz.max():>9.6f} {kappa(lam) / rho:>10.6f}")


def prop4(ms=(3, 4, 5)):
    print("\nProposition 4: the load vanishes on phase-interior nodes")
    print(f"{'m':>3} {'interior':>9} {'of':>6} {'max |F| there':>14} "
          f"{'max |F| all':>12}")
    for m in ms:
        chi = _chi_square(0.25, m)
        idx = interior_dofs(chi)
        worst, allmax = 0.0, 0.0
        for r in (1e1, 1e4, 1e8):
            F = load(m, chi, 1.0 / r, 1.0, HYDROSTATIC)
            worst = max(worst, np.abs(F[idx]).max())
            allmax = max(allmax, np.abs(F).max())
        assert worst < 1e-14
        print(f"{m:>3} {len(idx) // 2:>9} {4 ** m:>6} {worst:>14.2e} "
              f"{allmax:>12.5f}")


def prop8(ms=(3, 4, 5)):
    print("\nProposition 8: two scalars from one quadratic form each")
    print(f"{'m':>3} {'r':>6} {'K_H':>10} {'G_H':>10} {'vs full C^H':>12}")
    for m in ms:
        chi = _chi_square(0.25, m)
        for r in (2, 10, 100):
            KH, GH, _ = bulk_shear(m, chi, r)
            C = homogenized(m, chi, 1.0 / r, 1.0)
            e = max(abs(KH - HYDROSTATIC @ C @ HYDROSTATIC / 4),
                    abs(GH - SHEAR @ C @ SHEAR))
            assert e < 1e-12
            print(f"{m:>3} {r:>6} {KH:>10.6f} {GH:>10.6f} {e:>12.2e}")
    C = homogenized(4, _chi_square(0.25, 4), 1.0, 1.0)
    e = np.abs(C - plane_stress(NU)).max()
    assert e < 1e-12
    print(f"  homogeneous cell returns D_1 to {e:.2e}")


if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    prop1(); prop2(); prop3(); prop4(); prop8()
    print()
    table_effective(); table_truncation(); table_budget()
    print("---------------------------------------------------------------")
    print("all propositions asserted")

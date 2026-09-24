"""scope -- where the effective constant holds, and where it degrades.

Proposition 5 bounds the effective condition number by a constant determined by
the stiff phase and independent of contrast. That constant is a property of the
geometry, and it degrades when the stiff ligaments thin relative to the mesh.
The centred-square family sweeps the void fraction continuously at an oracle
cost flat in the mesh, so the boundary can be located without changing the
encoding.

Two sweeps:
  - the constant against void fraction at fixed mesh, which is the scope figure;
  - the constant against mesh at fixed void fraction, which separates a genuine
    geometric constant from a mesh artefact. The dyadic square is self-similar
    across m, so its geometry does not change with refinement.
"""
from __future__ import annotations

import numpy as np

from .macroload import HYDROSTATIC, SHEAR
from .microstructure import ligament, oracle_cost, square_t
from .precondition import extremes, kappa_eff_free

RHO = 1e4


def table_voidfraction(m=5, ts=None):
    N = 2 ** m
    ts = ts or [t for t in range(1, N // 2) if t % 2 == 0]
    print(f"Effective constant against void fraction, m = {m}, rho = {RHO:.0e}")
    print(f"{'t':>4} {'side':>5} {'v_f':>8} {'ligament':>9} {'eff hyd':>9} "
          f"{'eff shr':>9} {'CX':>6}")
    for t in ts:
        chi = square_t(m, t)
        kh = kappa_eff_free(m, chi, 1.0 / RHO, 1.0, HYDROSTATIC)
        ks = kappa_eff_free(m, chi, 1.0 / RHO, 1.0, SHEAR)
        _, cx = oracle_cost(m, t)
        print(f"{t:>4} {N - 2 * t:>5} {((N - 2 * t) / N) ** 2:>8.4f} "
              f"{ligament(m, t):>9.0f} {kh:>9.3f} {ks:>9.3f} {cx:>6}")


def table_mesh(vf=0.25, ms=(3, 4, 5, 6)):
    print(f"\nSelf-similar square at v_f = {vf}, against the mesh")
    print(f"{'m':>3} {'N':>4} {'worst':>9} {'eff hyd':>9} {'eff shr':>9} "
          f"{'CX':>6}")
    for m in ms:
        N = 2 ** m
        t = int(round((N - N * np.sqrt(vf)) / 2))
        chi = square_t(m, t)
        lo, hi = extremes(m, chi, 1.0 / RHO, 1.0)
        _, cx = oracle_cost(m, t)
        print(f"{m:>3} {N:>4} {hi / lo:>9.1f} "
              f"{kappa_eff_free(m, chi, 1.0/RHO, 1.0, HYDROSTATIC):>9.3f} "
              f"{kappa_eff_free(m, chi, 1.0/RHO, 1.0, SHEAR):>9.3f} {cx:>6}")


if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    table_voidfraction(m=5)
    table_mesh()
    print("---------------------------------------------------------------")

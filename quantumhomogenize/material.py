"""material -- the plane-strain constitutive law, the one source of D.

The cell is a transverse section of a continuous-fiber composite: the fiber
axis is out of plane, the load lies in the transverse plane, and the strain
along the fiber axis vanishes. That is plane strain,

    D(E, nu) = E / ((1 + nu)(1 - 2 nu)) [[ 1 - nu,  nu,      0            ],
                                         [ nu,      1 - nu,  0            ],
                                         [ 0,       0,       (1 - 2nu)/2  ]],

in Voigt order (exx, eyy, gxy) with engineering shear gxy = 2 exy, so that
D_33 = mu. Every element matrix, symbol, load and Voigt bound in the package
is built from this one function.

Plane strain at (E, nu) is plane stress at

    E* = E / (1 - nu^2),     nu* = nu / (1 - nu),

exactly. `plane_stress_equivalent` returns that map. It is used to check the
package against the plane-stress element of `pyblockencode`, and it carries
the block-encoding paper's closed-form subnormalizations over by substitution,

    single phase:  alpha = E (33 - 32 nu) / (6 (1 + nu)(1 - 2 nu)),
    two phase:     alpha = [ (E1 + E2)(33 - 32 nu)/2
                             + |E1 - E2| (18 - 16 nu + 3 |1 - 4 nu|) ]
                           / (6 (1 + nu)(1 - 2 nu)).

The iY component vanishes at nu = 1/4 (nu* = 1/3), and both forms diverge as
nu -> 1/2, the incompressible limit. Both are checked against
`pyblockencode.element_alpha` on the plane-strain element in verify.py.

    from .material import plane_strain
    D = plane_strain(nu=0.3)                 # unit modulus, 3 x 3
"""
from __future__ import annotations

import numpy as np


def plane_strain(nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    """D(E, nu), the plane-strain matrix. 3 x 3, engineering shear."""
    return E / ((1 + nu) * (1 - 2 * nu)) * np.array(
        [[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])


def plane_stress_equivalent(nu: float = 0.3,
                            E: float = 1.0) -> tuple[float, float]:
    """(E*, nu*) at which the plane-stress matrix equals plane_strain(nu, E)."""
    return E / (1 - nu ** 2), nu / (1 - nu)


def alpha_closed_form(nu: float = 0.3, E: float = 1.0) -> float:
    """Single-phase LCU subnormalization (L = 17), under plane strain."""
    return E * (33 - 32 * nu) / (6 * (1 + nu) * (1 - 2 * nu))


def alpha_two_phase(nu: float, E1: float, E2: float) -> float:
    """Two-phase LCU subnormalization (L = 57), under plane strain."""
    return ((E1 + E2) * (33 - 32 * nu) / 2
            + abs(E1 - E2) * (18 - 16 * nu + 3 * abs(1 - 4 * nu))
            ) / (6 * (1 + nu) * (1 - 2 * nu))


# ==========================================================================
if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    print(f"{'nu':>6} {'nu*':>8} {'|D - D*(stress)|':>18}")
    for nu in (0.0, 0.2, 0.3, 0.45):
        Es, ns = plane_stress_equivalent(nu)
        Ds = Es / (1 - ns ** 2) * np.array(
            [[1, ns, 0], [ns, 1, 0], [0, 0, (1 - ns) / 2]])
        print(f"{nu:>6.2f} {ns:>8.4f} {np.abs(plane_strain(nu) - Ds).max():>18.2e}")
    print("---------------------------------------------------------------")

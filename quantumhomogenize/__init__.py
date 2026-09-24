"""
quantumhomogenize -- quantum homogenization via fast inversion, 2D elasticity.

Computational homogenization of a periodic two-phase cell on a quantum computer.
The operator is block-encoded by the companion package `pyblockencode`; this
package preconditions it by fast inversion of the mean part, establishes the
conditioning results that follow, and reduces the homogenized moduli to scalar
quadratic forms in the resolvent.

    from quantumhomogenize import square, bulk_shear, encode, PRECONDITIONER

    chi = square(m=3, vf=0.25)              # centred square inclusion
    KH, GH, ratio = bulk_shear(3, chi, r=10)
    circuit, info = encode(PRECONDITIONER(nu=0.3), m=3, materialize=True)

Layout
------
    fourier_symbol   closed-form 2x2 symbol of the mean part and its inverses
    fastinvert       block encoding of A^{-1/2}: QFT pair, multiplexed rotation
    macroload        macroscopic-strain load; the phase-interior node set
    microstructure   centred squares at any volume fraction, with oracle costs
    homogenized      C^H, the Voigt term, bulk and shear moduli
    precondition     M = A^{-1/2} K^chi A^{-1/2}; spectrum, kappa, kappa_eff
    conditioning     effective against worst case; truncation bias
    qspdegree        polynomial degree on the effective interval
    fastsolve        end to end: precondition, QSVT, modulus
    scope            where the effective constant holds and where it degrades
    qsvt             ChebIter polynomial, QSP phases, solver (vendored)

The only external package needed is `pyblockencode`, plus `pyqsp` for the phase
angles in `qsvt`. Nothing is imported from a sibling clone.

Conventions follow the block-encoding paper: N = 2^m, degree of freedom
d*N^2 + y*N + x with x low and y high and the displacement qubit most
significant, chi[ex, ey] the inclusion indicator, plane stress with
C = E/(1-nu^2). The unit cell has h = 1/N: the Q4 stiffness is scale free in
two dimensions so the operator is unchanged, the load carries one factor of h,
and |Omega| = 1.
"""
__version__ = "0.1.0"

from .fourier_symbol import symbol, symbol_inverse, symbol_inv_sqrt
from .fastinvert import PRECONDITIONER, FastInversion, encode
from .macroload import HYDROSTATIC, SHEAR, load, interior_nodes, interior_dofs
from .microstructure import square, square_t, oracle_cost, ligament
from .homogenized import bulk_shear, homogenized, voigt, plane_stress
from .precondition import (preconditioned, spectrum, kappa, kappa_effective,
                           effective_interval_dense, extremes, kappa_eff_free)
from .qsvt import ChebIterPolynomial, QSVT

__all__ = [
    "symbol", "symbol_inverse", "symbol_inv_sqrt",
    "PRECONDITIONER", "FastInversion", "encode",
    "HYDROSTATIC", "SHEAR", "load", "interior_nodes", "interior_dofs",
    "square", "square_t", "oracle_cost", "ligament",
    "bulk_shear", "homogenized", "voigt", "plane_stress",
    "preconditioned", "spectrum", "kappa", "kappa_effective",
    "effective_interval_dense", "extremes", "kappa_eff_free",
    "ChebIterPolynomial", "QSVT",
]

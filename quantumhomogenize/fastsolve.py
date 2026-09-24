"""fastsolve -- the homogenized modulus from a QSVT solve on the preconditioned
operator.

This closes the loop the block-encoding paper left open in its conclusions: the
end-to-end cost of a QSVT solve on these encodings. The chain is

    K^chi  --precondition-->  M = A^{-1/2} K^chi A^{-1/2}
           --QSVT p(M) ~ M^{-1}-->  f^T M^{-1} f
           --Proposition 8-->  K_H  or  G_H.

The readout is the quadratic form itself, not the solution state, so what the
solver must return is  f^T p(M) f  and not  p(M)|f>. The solver in `qsvt`
returns the post-selected direction and the real-part norm, which give that
contraction directly:

    f^T p(M) f  =  (f . u_dir) * tau * norm_real,

with tau the polynomial normalisation. Because the observable is a quadratic
form, no amplitude of the solution vector is ever read individually and the
xi parameter of Tong et al. never enters.

The degree is set by the effective interval, not the worst case, which is
Proposition 6. The base polynomial is ChebIterPolynomial, the closed-form
minimiser of the relative residual max |x p(x) - 1|, which is the criterion
this readout inherits.

    from .fastsolve import solve_modulus, table_endtoend
    table_endtoend()

Simulation cost is the binding constraint: the statevector has 2^(n+2)
amplitudes for n = 1 + 2m data qubits, and the circuit depth is the polynomial
degree, so m = 2 and m = 3 are affordable and m = 4 is not.
"""
from __future__ import annotations

import numpy as np

from .qsvt import QSVT

from .macroload import HYDROSTATIC, SHEAR                       
from .microstructure import square_t                            
from .homogenized import correction, plane_stress, voigt             
from .precondition import (apply_half_inverse, effective_interval_dense,
                          kappa, preconditioned, project,
                          spectrum, zero_mean_basis)           

NU = 0.3
SCALE = 1.01      # keeps every singular value strictly inside (0, 1)


# ==========================================================================
# 1. the reduced problem the solver sees
# ==========================================================================
def reduced(m: int, chi: np.ndarray, rho: float, eps0=HYDROSTATIC,
            nu: float = NU):
    """M and the preconditioned load, on a register of 1 + 2m qubits.

    M is singular on the two rigid translations, and the zero-mean subspace has
    dimension 2 N^2 - 2, which is not a power of two. Rather than compress to a
    non-addressable size, the translations are kept and M is DEFLATED on them:
    the block is replaced by the identity. Those directions then carry
    eigenvalue 1, sit inside the approximation interval, and receive no weight
    from the load, which Proposition 4 places on the interface. The padded
    operator has the same spectrum as M elsewhere, so the condition number and
    the quadratic form are both unchanged.
    """
    from .macroload import load

    E1, E2 = 1.0 / rho, 1.0
    M = preconditioned(m, chi, E1, E2, nu)
    N = 2 ** m
    e = np.zeros((2 * N * N, 2))
    e[:N * N, 0] = e[N * N:, 1] = 1.0 / N
    M = M + e @ e.T                       # deflate: translations -> eigenvalue 1
    M = M / SCALE                         # lam_max = 1 exactly is not
                                          # block-encodable: sqrt(I-M^2)
                                          # is singular there.

    F = load(m, chi, E1, E2, eps0, nu)
    f = project(m, apply_half_inverse(m, F, nu, max(E1, E2)))
    return 0.5 * (M + M.T), f


def solve_modulus(m: int, chi: np.ndarray, rho: float, eps0=HYDROSTATIC,
                  eps: float = 1e-3, nu: float = NU, use_effective=True):
    """f^T M^{-1} f by QSVT, against the classical value.

    Returns (quantum, classical, relative error, degree, success probability).
    """
    M, f = reduced(m, chi, rho, eps0, nu)
    nrm = np.linalg.norm(f)
    b = f / nrm

    lam, U, Fp = spectrum(m, chi, 1.0 / rho, 1.0, nu)
    col = 0 if eps0 is HYDROSTATIC else 1
    if use_effective:
        a, top = effective_interval_dense(lam, U, Fp[:, col])
        kap = top / a
    else:
        kap = kappa(lam)

    solver = QSVT(M, b, kappa=kap, target_error=eps)
    u_dir, prob, norm_real = solver.solve()

    # f^T M^{-1} f = ||f||^2 * b^T M^{-1} b, and the solver returns the
    # direction of p(M)b together with the norm that rescales it. u_dir is
    # a direction, so its global sign is arbitrary; the form is positive
    # definite, which fixes it.
    quantum = abs(float(b @ u_dir)) * solver.tau * norm_real * nrm ** 2 / SCALE
    classical = float(f @ np.linalg.solve(M * SCALE, f))
    return (quantum, classical, abs(quantum - classical) / abs(classical),
            solver.degree, prob)


# ==========================================================================
# 2. the modulus
# ==========================================================================
def modulus_from_solve(m: int, chi: np.ndarray, rho: float, kind="bulk",
                       eps: float = 1e-3, nu: float = NU):
    """K_H or G_H with the correction supplied by the QSVT solve."""
    eps0, scale = ((HYDROSTATIC, 0.25) if kind == "bulk" else (SHEAR, 1.0))
    E1, E2 = 1.0 / rho, 1.0
    C0 = voigt(chi, E1, E2, nu)
    q, c, rel, d, prob = solve_modulus(m, chi, rho, eps0, eps, nu)
    v = scale * float(eps0 @ C0 @ eps0)
    return v - scale * q, v - scale * c, rel, d, prob


# ==========================================================================
if __name__ == "__main__":
    import contextlib
    import io

    print("----- output --------------------------------------------------")
    print("End to end: the correction term by QSVT on the preconditioned "
          "operator")
    print(f"{'m':>3} {'rho':>6} {'load':>6} {'degree':>7} {'quantum':>11} "
          f"{'classical':>11} {'rel err':>9} {'P_succ':>8}")
    for m in (2, 3):
        chi = square_t(m, 2 ** m // 4)
        for rho in (10.0, 100.0):
            for name, e0 in (("hyd", HYDROSTATIC), ("shr", SHEAR)):
                with contextlib.redirect_stdout(io.StringIO()):
                    q, c, rel, d, p = solve_modulus(m, chi, rho, e0, eps=1e-3)
                print(f"{m:>3} {rho:>6.0f} {name:>6} {d:>7} {q:>11.6f} "
                      f"{c:>11.6f} {rel:>9.2e} {p:>8.4f}")

    print("\nThe modulus itself, m = 3, eps = 1e-3")
    print(f"{'rho':>6} {'kind':>6} {'quantum':>11} {'classical':>11} "
          f"{'rel err':>9}")
    chi = square_t(3, 2)
    for rho in (10.0, 100.0):
        for kind in ("bulk", "shear"):
            with contextlib.redirect_stdout(io.StringIO()):
                q, c, rel, d, p = modulus_from_solve(3, chi, rho, kind,
                                                     eps=1e-3)
            print(f"{rho:>6.0f} {kind:>6} {q:>11.6f} {c:>11.6f} "
                  f"{abs(q - c) / abs(c):>9.2e}")
    print("---------------------------------------------------------------")

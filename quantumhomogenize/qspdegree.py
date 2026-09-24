"""qspdegree -- the polynomial degree on the effective interval against the
worst-case interval.

Proposition 6. A QSVT approximation to the inverse may be built on the interval
the load occupies rather than on the whole spectrum of M, provided the bias from
the discarded modes stays below the target accuracy. Because the observable is a
quadratic form, that bias is linear in the discarded weight rather than square
root:

    | Q - Qhat |  <=  eps ||f||^2  +  w_out ( kappa + 4 kappa_eff / 3 ),

and with Q >= ||f||^2 the relative truncation error is about w_out * kappa.

The base polynomial is ChebIterPolynomial from `qsvt`, vendored from
UW-ERSL/SpectralCorrectionQSVT: the closed-form minimiser of the RELATIVE residual max |x p(x) - 1| on [a, 1]
(Gribling, Kerenidis and Szilagyi, arXiv:2109.04248, Corollary 8). The relative
criterion is the one this readout needs: the quantity computed is

    Q = F^T K^{-1} F = sum_k w_k / lambda_k,

so the error that propagates is sum_k w_k | p(lambda_k) - 1/lambda_k |, and
weighting by lambda_k turns it into exactly the criterion ChebIter minimises.
An absolute-criterion polynomial is stricter by up to kappa here, which would
overstate the degree and flatter the comparison.

Its residual equioscillates, so the achieved error is exact rather than a
bound, eps = 1 / cosh(n arccosh z0) with z0 = (1+a^2)/(1-a^2) and n = (d+1)/2,
and mindegree inverts that in closed form.

    from .qspdegree import table_degree
    table_degree()

The saving is the ratio of the two degrees, which is not the ratio of the two
condition numbers: the degree depends on a through arccosh z0 rather than
through 1/a, and the two agree only as a tends to zero.
"""
from __future__ import annotations

import numpy as np

from .qsvt import ChebIterPolynomial as CI

from .macroload import HYDROSTATIC, SHEAR                      
from .microstructure import square_t                           
from .precondition import (discarded_weight, effective_interval_dense,
                          kappa, spectrum)                    

EPS = 1e-6


# ==========================================================================
def degree(a: float, eps: float = EPS) -> int:
    """Minimal odd degree for the relative criterion on [a, 1]."""
    return int(CI.mindegree(eps, a))


def achieved(d: int, a: float) -> float:
    """Equioscillating residual of that degree. Exact, not a bound."""
    return float(CI.error_for_degree(d, a))


def residual(p, lam, w) -> float:
    """sum_k w_k | lambda_k p(lambda_k) - 1 |, the bias the readout inherits."""
    return float((w * np.abs(lam * p(lam) - 1.0)).sum())


# ==========================================================================
def table_degree(ms=(3, 4, 5), rhos=(1e2, 1e3, 1e4), vf=0.25, eps=EPS):
    """The cost claim: degree on the effective interval against the worst case."""
    print(f"Degree for the relative criterion at eps = {eps:.0e}, "
          f"square v_f = {vf}")
    print(f"{'m':>3} {'rho':>7} {'kappa':>9} {'k_eff':>7} {'d worst':>9} "
          f"{'d eff':>7} {'saving':>8} {'trunc/Q':>10}")
    for m in ms:
        chi = square_t(m, 2 ** m // 4)
        for rho in rhos:
            lam, U, Fp = spectrum(m, chi, 1.0 / rho, 1.0)
            f = Fp[:, 0]
            kw = kappa(lam)
            a, b = effective_interval_dense(lam, U, f)
            dw, de = degree(1.0 / kw, eps), degree(a / b, eps)
            _, rel = discarded_weight(lam, U, f, a)
            print(f"{m:>3} {rho:>7.0e} {kw:>9.1f} {b / a:>7.3f} {dw:>9} "
                  f"{de:>7} {dw / de:>8.1f} {rel:>10.2e}")


def table_budget(m=4, rho=1e4, vf=0.25):
    """The degree must cover the whole budget, polynomial plus truncation."""
    chi = square_t(m, 2 ** m // 4)
    lam, U, Fp = spectrum(m, chi, 1.0 / rho, 1.0)
    f = Fp[:, 0]
    a, b = effective_interval_dense(lam, U, f)
    _, rel = discarded_weight(lam, U, f, a)
    w = (U.T @ f) ** 2
    w /= w.sum()
    print(f"\nError budget at m = {m}, rho = {rho:.0e}. Truncation contributes "
          f"{rel:.1e} and does not move with the degree.")
    print(f"{'eps':>10} {'d eff':>7} {'certificate':>13} {'measured':>11} "
          f"{'+ trunc':>11}")
    for eps in (1e-3, 1e-6, 1e-9, 1e-12):
        d = degree(a / b, eps)
        p = CI.poly(d, a / b)
        meas = residual(p, lam[lam >= a] / b, w[lam >= a])
        print(f"{eps:>10.0e} {d:>7} {achieved(d, a / b):>13.2e} "
              f"{meas:>11.2e} {meas + rel:>11.2e}")


def table_loads(m=4, rho=1e4, vf=0.25, eps=EPS):
    """The two loads occupy different intervals and cost different degrees."""
    chi = square_t(m, 2 ** m // 4)
    lam, U, Fp = spectrum(m, chi, 1.0 / rho, 1.0)
    print(f"\nThe two loads at m = {m}, rho = {rho:.0e}")
    print(f"{'load':>12} {'interval':>22} {'k_eff':>8} {'degree':>8}")
    for name, col in (("hydrostatic", 0), ("shear", 1)):
        a, b = effective_interval_dense(lam, U, Fp[:, col])
        print(f"{name:>12} {'[' + f'{a:.4f}, {b:.4f}' + ']':>22} "
              f"{b / a:>8.3f} {degree(a / b, eps):>8}")


# ==========================================================================
if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    table_degree()
    table_budget()
    table_loads()
    print("---------------------------------------------------------------")

"""precondition -- fast-inversion preconditioning of the two-phase cell.

The two-phase operator splits into a mean part and a contrast part, Equation
(42) of the block-encoding paper. The mean part is block-circulant, so it is
fast invertible through the closed-form symbol of `symbol.py`. Composing a
block encoding of its inverse with one of the contrast part would multiply
subnormalizations and inflate alpha by the condition number of the mean part,
which is of order N^2, cancelling the gain. Instead the inverse square root is
split onto both sides,

    M = A^{-1/2} K^chi A^{-1/2},      A = E_0 K,   E_0 = max(E1, E2),

with K the homogeneous cell at unit modulus. Writing K^chi = G^dag W E G for
the discrete gradient G, the quadrature weights W and the element moduli E,

    M = V^dag E V,      V = W^{1/2} G A^{-1/2},      V^dag V = E_0^{-1} P,

so V is a partial isometry scaled by E_0^{-1/2}, with norm independent of N,
and M is assembled as a symmetric product rather than as a product of block
encodings. Two consequences follow for every nu, every N and every contrast:

    spec(M) = [ 1/rho, 1 ] exactly,     rho = max(E1,E2) / min(E1,E2),

so the subnormalization is 1 and the condition number is the contrast alone.
Both are tight: E_min K <= K^chi <= E_max K in the Loewner order, with equality
attained whenever either phase fills a neighbourhood.

Taking A as the stiff-phase operator rather than the mean part (E1+E2)/2 K is a
choice of scalar. Both are the same block-circulant operator and both are
equally fast invertible; the stiff-phase normalization is the one that makes
the spectrum [1/rho, 1] and the subnormalization 1 rather than 2.

    from .precondition import preconditioned, spectrum
    M = preconditioned(m=5, chi=chi, E1=1e-4, E2=1.0)
    lam, U, Fp = spectrum(m=5, chi=chi, E1=1e-4, E2=1.0)

All applications of A^{-1/2} go through the symbol and the FFT, so forming M
costs O(N^4 log N) rather than the O(N^6) of a dense eigendecomposition.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh

from .macroload import HYDROSTATIC, SHEAR, load, stiffness
from .fourier_symbol import apply_mode_operator, symbol, symbol_inv_sqrt

TOL = 1e-9


# ==========================================================================
# 1. the preconditioned operator
# ==========================================================================
def half_inverse(m: int, nu: float = 0.3, E0: float = 1.0) -> np.ndarray:
    """Mode-wise A^{-1/2} with A = E0 * (homogeneous cell at unit modulus)."""
    return symbol_inv_sqrt(m, nu, 1.0) / np.sqrt(E0)


def apply_half_inverse(m: int, v: np.ndarray, nu: float = 0.3,
                       E0: float = 1.0) -> np.ndarray:
    """A^{-1/2} v, through the symbol and the FFT."""
    return apply_mode_operator(half_inverse(m, nu, E0), v).real


def preconditioned(m: int, chi: np.ndarray, E1: float, E2: float,
                   nu: float = 0.3) -> np.ndarray:
    """M = A^{-1/2} K^chi A^{-1/2}, dense, symmetrised."""
    K = stiffness(m, chi, E1, E2, nu)
    E0 = max(E1, E2)
    M = apply_half_inverse(m, K @ apply_half_inverse(m, np.eye(K.shape[0]),
                                                     nu, E0).T, nu, E0)
    return 0.5 * (M + M.T)


def zero_mean_basis(m: int, nu: float = 0.3) -> np.ndarray:
    """Orthonormal basis of the complement of the rigid translations.

    The zero mode of the symbol is the only one annihilated, and it carries
    the two constant displacement fields, so the complement is spanned by the
    Fourier modes other than (0, 0).
    """
    N = 2 ** m
    e = np.zeros((2 * N * N, 2))
    e[:N * N, 0] = e[N * N:, 1] = 1.0 / N
    Q, _ = np.linalg.qr(np.hstack([e, np.eye(2 * N * N)]))
    return Q[:, 2:]                      # the translations are columns 0, 1


def project(m: int, v: np.ndarray) -> np.ndarray:
    """Remove the two rigid translations. Matrix free."""
    N = 2 ** m
    w = v.reshape(2, N * N, -1) if v.ndim > 1 else v.reshape(2, N * N, 1)
    w = w - w.mean(axis=1, keepdims=True)
    return w.reshape(2 * N * N, -1)[:, 0] if v.ndim == 1 else w.reshape(
        2 * N * N, -1)


# ==========================================================================
# 2. matrix-free application
# ==========================================================================
def apply_stiffness(m: int, chi: np.ndarray, E1: float, E2: float,
                    v: np.ndarray, nu: float = 0.3) -> np.ndarray:
    """K^chi v by element gather and scatter, without forming K^chi.

    Cost O(N^2) per column against O(N^4) for the dense operator, and the
    element matrix is the same 8 x 8 at every element.
    """
    from .macroload import element
    from pyblockencode import CORNERS

    N = 2 ** m
    Ke, _ = element(nu, 1.0 / N)
    E = E2 + (E1 - E2) * chi                                # E[ex, ey]
    single = v.ndim == 1
    V = v.reshape(-1, 1) if single else v

    ex, ey = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    idx = [(d * N * N + ((ey + oy) % N) * N + (ex + ox) % N).ravel()
           for ox, oy in CORNERS for d in (0, 1)]           # local dof order

    u = np.stack([V[i] for i in idx])                       # 8 x nel x ncol
    w = np.einsum("ab,bec->aec", Ke, u) * E.ravel()[None, :, None]
    out = np.zeros_like(V)
    for a, i in enumerate(idx):
        np.add.at(out, i, w[a])
    return out[:, 0] if single else out


def apply_M(m: int, chi: np.ndarray, E1: float, E2: float, v: np.ndarray,
            nu: float = 0.3) -> np.ndarray:
    """M v = A^{-1/2} K^chi A^{-1/2} v, projected off the rigid translations."""
    E0 = max(E1, E2)
    w = apply_half_inverse(m, project(m, v), nu, E0)
    w = apply_stiffness(m, chi, E1, E2, w, nu)
    return project(m, apply_half_inverse(m, w, nu, E0))


def operator(m: int, chi: np.ndarray, E1: float, E2: float, nu: float = 0.3):
    """M as a scipy LinearOperator on the full space, singular on the
    translations, which Lanczos never leaves the projected subspace to see."""
    n = 2 * 4 ** m
    return LinearOperator((n, n), dtype=float, matvec=lambda v: apply_M(
        m, chi, E1, E2, v, nu))


# ==========================================================================
# 2. spectrum and conditioning
# ==========================================================================
def spectrum(m: int, chi: np.ndarray, E1: float, E2: float, nu: float = 0.3,
             loads=(HYDROSTATIC, SHEAR)):
    """Eigenpairs of M on the zero-mean subspace, and the preconditioned loads.

    Returns (lam, U, Fp) with lam ascending, U the eigenvectors in the reduced
    basis, and Fp the loads mapped through A^{-1/2} into that basis.
    """
    Q = zero_mean_basis(m, nu)
    M = preconditioned(m, chi, E1, E2, nu)
    lam, U = eigh(Q.T @ M @ Q)
    F = np.column_stack([load(m, chi, E1, E2, e, nu) for e in loads])
    return lam, U, Q.T @ apply_half_inverse(m, F, nu, max(E1, E2))


def kappa(lam: np.ndarray) -> float:
    """Worst-case condition number over the nonzero spectrum."""
    nz = lam[lam > TOL * lam.max()]
    return float(nz.max() / nz.min())


# ==========================================================================
# 3. extremal quantities without the full spectrum
# ==========================================================================
def extremes(m: int, chi: np.ndarray, E1: float, E2: float, nu: float = 0.3,
             k: int = 2) -> tuple[float, float]:
    """(lam_min, lam_max) of M on the zero-mean subspace, by Lanczos.

    Proposition 3 is a Loewner sandwich, E_min K <= K^chi <= E_max K, so these
    two numbers are the whole of it and the interior spectrum is never needed.
    """
    A = operator(m, chi, E1, E2, nu)
    hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False)[0])

    # lam_min of M on the zero-mean subspace is hi minus the largest eigenvalue
    # of (hi P - M), which is again a matrix-vector product. Shift-invert would
    # need a factorization the operator does not have.
    n = A.shape[0]
    B = LinearOperator((n, n), dtype=float, matvec=lambda v: (
        hi * project(m, v) - apply_M(m, chi, E1, E2, v, nu)))
    gap = float(eigsh(B, k=k, which="LA", return_eigenvectors=False).max())
    return hi - gap, hi


def effective_interval(m: int, chi: np.ndarray, E1: float, E2: float,
                       eps0=HYDROSTATIC, nu: float = 0.3, delta: float = 1e-8,
                       ncv: int = 80) -> tuple[float, float]:
    """The interval carrying all but `delta` of the load's spectral weight.

    Lanczos started from the preconditioned load explores exactly the Krylov
    space the load occupies, so the modes it is orthogonal to never enter. The
    smallest Ritz value carrying weight above the budget is the lower endpoint.
    """
    E0 = max(E1, E2)
    f = project(m, apply_half_inverse(
        m, load(m, chi, E1, E2, eps0, nu), nu, E0))
    f /= np.linalg.norm(f)

    # Lanczos from the load. The Ritz values are the spectrum M shows to f, and
    # the squared first components of the tridiagonal eigenvectors are the
    # spectral weights, so the modes f is orthogonal to never appear.
    n = min(ncv, 2 * 4 ** m - 2)
    q, a, b = f, [], []
    Q = [f]
    prev = np.zeros_like(f)
    beta = 0.0
    for j in range(n):
        w = apply_M(m, chi, E1, E2, q, nu) - beta * prev
        alpha = float(w @ q)
        w = w - alpha * q
        for u in Q:                               # full reorthogonalisation
            w = w - (w @ u) * u
        a.append(alpha)
        beta = float(np.linalg.norm(w))
        if beta < 1e-13 or j == n - 1:
            break
        b.append(beta)
        prev, q = q, w / beta
        Q.append(q)

    from scipy.linalg import eigh_tridiagonal
    lam, S = eigh_tridiagonal(np.array(a), np.array(b))
    wt = S[0] ** 2
    wt /= wt.sum()
    lo = np.searchsorted(np.cumsum(wt), delta / 2)
    return float(lam[lo]), float(lam[-1])


def kappa_eff_free(m: int, chi: np.ndarray, E1: float, E2: float,
                   eps0=HYDROSTATIC, nu: float = 0.3,
                   delta: float = 1e-8) -> float:
    """Effective condition number, matrix free."""
    a, b = effective_interval(m, chi, E1, E2, eps0, nu, delta)
    return b / a


def kappa_effective(lam: np.ndarray, U: np.ndarray, f: np.ndarray,
                    delta: float = 1e-8) -> float:
    """Condition number of the smallest interval carrying all but `delta`
    of the load's spectral weight.

    The budget is on the discarded weight, not on a per-mode tolerance, which
    is the quantity the polynomial-degree argument needs. The endpoints are
    insensitive to `delta` across many orders of magnitude, because the excited
    modes and the compliant-phase cluster are separated by a genuine gap.
    """
    a, b = effective_interval_dense(lam, U, f, delta)
    return b / a


def discarded_weight(lam: np.ndarray, U: np.ndarray, f: np.ndarray,
                     cut: float) -> tuple[float, float]:
    """Normalised weight below `cut`, and its contribution to f^T M^+ f.

    The second number is the bias a polynomial built above `cut` incurs in the
    readout. It is linear in the discarded weight because the observable is a
    quadratic form.
    """
    w = (U.T @ f) ** 2
    w /= w.sum()
    s = lam < cut
    q = (w / lam).sum()
    return float(w[s].sum()), float((w[s] / lam[s]).sum() / q)


def effective_interval_dense(lam: np.ndarray, U: np.ndarray, f: np.ndarray,
                             delta: float = 1e-8) -> tuple[float, float]:
    """Interval the polynomial must cover, from a full spectrum.

    Only the lower endpoint is set by the load. The upper endpoint is
    lam_max of M itself, because the polynomial has to be accurate wherever
    M has spectrum, not only where the load has weight.
    """
    w = (U.T @ f) ** 2
    w /= w.sum()
    lo = np.searchsorted(np.cumsum(w), delta / 2)
    return float(lam[lo]), float(lam.max())


# ==========================================================================
# 3. the isometry bound
# ==========================================================================
def isometry_residual(m: int, nu: float = 0.3, E0: float = 1.0) -> tuple:
    """Check V^dag V = E0^{-1} P and ||V|| = E0^{-1/2} without forming V.

    V^dag V = A^{-1/2} K A^{-1/2} with K the homogeneous cell, which is the
    preconditioned operator of a one-phase cell. Returns the residual against
    E0^{-1} P and the norm of V.
    """
    N = 2 ** m
    ones = np.ones((N, N))
    VtV = preconditioned(m, ones, E0, E0, nu) / E0    # A^{-1/2} K A^{-1/2}

    # P is the orthogonal projector off the two rigid translations, which are
    # the only vectors A^{-1/2} annihilates.
    e = np.zeros((2 * N * N, 2))
    e[:N * N, 0] = e[N * N:, 1] = 1.0 / N
    P = np.eye(2 * N * N) - e @ e.T
    return (float(np.abs(VtV - P / E0).max()),
            float(np.sqrt(np.linalg.norm(VtV, 2))))


# ==========================================================================
if __name__ == "__main__":
    from pyblockencode import _chi_square

    print("----- output --------------------------------------------------")
    print("Proposition 2: V^dag V = E_0^{-1} P,  ||V|| = E_0^{-1/2}")
    print(f"{'m':>3} {'E_0':>7} {'residual':>10} {'||V||':>9} "
          f"{'E_0^-1/2':>9}")
    for m in (2, 3, 4):
        for E0 in (1.0, 7.5):
            res, nv = isometry_residual(m, 0.3, E0)
            print(f"{m:>3} {E0:>7.2f} {res:>10.2e} {nv:>9.5f} "
                  f"{1 / np.sqrt(E0):>9.5f}")

    print("\nProposition 3: spec(M) = [1/rho, 1], alpha_M = 1")
    print(f"{'m':>3} {'nu':>6} {'rho':>8} {'lam_min':>11} {'lam_max':>9} "
          f"{'kappa':>12}")
    for m in (3, 4):
        for nu in (0.0, 0.3, 0.45):
            for rho in (1e2, 1e4):
                chi = _chi_square(0.25, m)
                lam, _, _ = spectrum(m, chi, 1.0 / rho, 1.0, nu)
                nz = lam[lam > TOL * lam.max()]
                print(f"{m:>3} {nu:>6.2f} {rho:>8.0e} {nz.min():>11.4e} "
                      f"{nz.max():>9.6f} {kappa(lam):>12.4f}")
    print("---------------------------------------------------------------")

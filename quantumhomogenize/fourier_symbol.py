"""symbol -- the Fourier symbol of the homogeneous periodic Q4 elasticity cell.

The mean part of the two-phase split (Eq. 42 of the block-encoding paper) is
block-circulant on the periodic grid, so the spatial DFT block-diagonalises it
into one 2x2 acoustic tensor per Fourier mode,

    Khat(p, q) = [[ Sxx, Sxy ],
                  [ Sxy, Syy ]],      p, q in {0, ..., N-1},

with, writing tx = 2 pi p / N, ty = 2 pi q / N and C = E / (1 - nu^2),

    kx = 4 sin^2(tx/2),   mx = (2 + cos tx) / 3,   sx = sin tx,

    Sxx = C ( kx my + (1-nu)/2 mx ky ),
    Syy = C ( (1-nu)/2 kx my + mx ky ),
    Sxy = C (1+nu)/2 sx sy.

These are the eigenvalues of K1 = circ(-1,2,-1), M1 = (1/6) circ(1,4,1) and
G1 = (1/2) circ(-1,0,1) substituted into Equations (15) to (17). The symbol is
real and symmetric at every mode, so it is Hermitian, and it is positive
definite off the zero mode.

The point of the closed form is that both the inverse and the inverse square
root of a 2x2 symmetric matrix are rational in its entries and two square
roots. With s = sqrt(det) and t = tr,

    Khat^{-1}    = [[ Syy, -Sxy ], [ -Sxy, Sxx ]] / s^2,
    Khat^{-1/2}  = [[ Syy + s, -Sxy ], [ -Sxy, Sxx + s ]] / (s sqrt(t + 2s)).

No eigendecomposition and no iteration is required, which is what places the
mean part inside Definition 2 of Tong et al.: a block encoding of its
pseudoinverse costs a QFT pair, a mode-wise preparation, and an inverse QFT,
independently of its condition number.

The zero mode carries the two rigid translations. Both inverses are set to zero
there, giving the Moore-Penrose pseudoinverse on the zero-mean subspace.

    from .fourier_symbol import symbol, symbol_inv_sqrt
    S  = symbol(m=5, nu=0.3)          # (N, N, 2, 2)
    Sh = symbol_inv_sqrt(m=5, nu=0.3)

Conventions follow pyblockencode: the global degree of freedom is
d * N^2 + y * N + x, so the displacement component is the most significant
index and the spatial registers are x low, y high.
"""
from __future__ import annotations

import numpy as np

TOL = 1e-12


# ==========================================================================
# 1. the symbol
# ==========================================================================
def _factors(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigenvalues of K1, M1 and sin(theta), indexed by the 1D mode."""
    t = 2 * np.pi * np.arange(2 ** m) / 2 ** m
    return 4 * np.sin(t / 2) ** 2, (2 + np.cos(t)) / 3, np.sin(t)


def symbol(m: int, nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    """The 2x2 acoustic tensor at every mode. Shape (N, N, 2, 2), real."""
    k, mu, s = _factors(m)
    C = E / (1 - nu ** 2)
    kx, ky = k[:, None], k[None, :]
    mx, my = mu[:, None], mu[None, :]
    sx, sy = s[:, None], s[None, :]

    out = np.empty((2 ** m, 2 ** m, 2, 2))
    out[..., 0, 0] = C * (kx * my + (1 - nu) / 2 * mx * ky)
    out[..., 1, 1] = C * ((1 - nu) / 2 * kx * my + mx * ky)
    out[..., 0, 1] = out[..., 1, 0] = C * (1 + nu) / 2 * sx * sy
    return out


def _det_tr(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = S[..., 0, 0] * S[..., 1, 1] - S[..., 0, 1] * S[..., 1, 0]
    return d, S[..., 0, 0] + S[..., 1, 1]


def symbol_inverse(m: int, nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    """Mode-wise inverse, zero at the zero mode. Closed form, no eigensolve."""
    S = symbol(m, nu, E)
    det, _ = _det_tr(S)
    out = np.zeros_like(S)
    ok = det > TOL * det.max()
    out[ok, 0, 0], out[ok, 1, 1] = S[ok, 1, 1], S[ok, 0, 0]
    out[ok, 0, 1] = out[ok, 1, 0] = -S[ok, 0, 1]
    return out / np.where(ok, det, 1.0)[..., None, None]


def symbol_inv_sqrt(m: int, nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    """Mode-wise inverse square root, zero at the zero mode.

    For a 2x2 symmetric positive definite S with s = sqrt(det) and t = tr,
    S^{1/2} = (S + s I) / sqrt(t + 2s), hence S^{-1/2} = adj(S + s I) /
    (s sqrt(t + 2s)). Two square roots and no eigendecomposition.
    """
    S = symbol(m, nu, E)
    det, tr = _det_tr(S)
    ok = det > TOL * det.max()
    s = np.sqrt(np.where(ok, det, 1.0))
    out = np.zeros_like(S)
    out[..., 0, 0], out[..., 1, 1] = S[..., 1, 1] + s, S[..., 0, 0] + s
    out[..., 0, 1] = out[..., 1, 0] = -S[..., 0, 1]
    return np.where(ok, 1.0, 0.0)[..., None, None] * out / (
        s * np.sqrt(tr + 2 * s))[..., None, None]


# ==========================================================================
# 2. applying a mode-wise operator
# ==========================================================================
def _to_field(v: np.ndarray, N: int) -> np.ndarray:
    """(2 N^2,) or (2 N^2, k) laid out d*N^2 + y*N + x -> (2, N, N, k)."""
    w = v.reshape(2, N, N, -1)              # d, y, x, column
    return np.swapaxes(w, 1, 2)             # d, x, y, column


def _from_field(w: np.ndarray, n: int) -> np.ndarray:
    v = np.swapaxes(w, 1, 2).reshape(2 * w.shape[1] * w.shape[2], -1)
    return v[:, 0] if n == 1 else v


def apply_mode_operator(T: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply the mode-wise 2x2 operator T to one vector or a set of columns.

    Forward DFT on both spatial registers, a 2x2 multiply per mode, inverse
    DFT. Cost O(N^2 log N) per column, against O(N^4) for a dense apply.
    """
    N = T.shape[0]
    single = v.ndim == 1
    w = _to_field(np.atleast_2d(v.T).T if single else v, N)
    f = np.fft.fft2(w, axes=(1, 2))
    g = np.einsum("xyde,exyc->dxyc", T, f)
    out = np.fft.ifft2(g, axes=(1, 2))
    return _from_field(out, 1 if single else v.shape[1])


def mode_operator_matrix(T: np.ndarray) -> np.ndarray:
    """Dense form of a mode-wise operator, for verification at small m."""
    N = T.shape[0]
    return apply_mode_operator(T, np.eye(2 * N * N, dtype=complex)).real


# ==========================================================================
# 3. dense reference, assembled without reference to the symbol
# ==========================================================================
def reference(m: int, nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    """The homogeneous cell assembled element by element from quadrature."""
    from pyblockencode import _q4_element, _assemble_periodic
    return _assemble_periodic(_q4_element(nu, E), 2 ** m, 2)


# ==========================================================================
if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    print(f"{'m':>3} {'nu':>6} {'sym vs assembly':>17} {'inv':>10} "
          f"{'inv sqrt':>10} {'zero modes':>11}")
    for m in (2, 3, 4):
        for nu in (0.0, 0.3, 1 / 3, 0.45):
            K = reference(m, nu)
            S, Si, Sh = (symbol(m, nu), symbol_inverse(m, nu),
                         symbol_inv_sqrt(m, nu))
            e_sym = np.abs(mode_operator_matrix(S) - K).max()

            # Khat^{-1} Khat = P and (Khat^{-1/2})^2 = Khat^{-1}, mode-wise.
            P = np.einsum("xyde,xyef->xydf", Si, S)
            P[0, 0] = np.eye(2)
            e_inv = np.abs(P - np.eye(2)).max()
            e_hlf = np.abs(np.einsum("xyde,xyef->xydf", Sh, Sh) - Si).max()

            nz = int((np.linalg.eigvalsh(K) < 1e-9 * np.abs(K).max()).sum())
            print(f"{m:>3} {nu:>6.3f} {e_sym:>17.2e} {e_inv:>10.2e} "
                  f"{e_hlf:>10.2e} {nz:>11}")
    print("---------------------------------------------------------------")

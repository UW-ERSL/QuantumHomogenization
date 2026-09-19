"""
Closed-form Fourier symbol of the unperturbed stiffness operator K_0.

For a uniform periodic Cartesian mesh of M x M bilinear-quad elements, the
global stiffness K_0 = A^T (I (x) K_e) A is block-circulant under PBC.  The
DFT block-diagonalizes it into a per-mode 2x2 symbol Khat_0(k) which depends
only on the inter-node coupling stencil derived from the 8x8 element template.

The stencil has 9 distinct 2x2 coupling blocks C_{dx, dy} for (dx, dy) in
{-1, 0, 1}^2, computed by summing the 2x2 corner-pair sub-blocks of K_e for
each of the four elements incident to a node under PBC.

References
----------
- The block-circulant structure of K_0 under PBC is standard in homogenization
  theory (e.g., Moulinec-Suquet 1995); the closed-form Fourier symbol follows
  directly.
- The derivation of the 9 coupling blocks from K_e is in the proposal text
  for the Aim 2 POC.

This module is purely classical (numpy only); the quantum block encoding is
in task4.build_USymbolInv.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from qhomogenize.fea.element_template import element_template


# ---------------------------------------------------------------------------
# Inter-node coupling blocks
# ---------------------------------------------------------------------------

# Each row of this table corresponds to one (dx, dy) coupling and lists the
# (c_i, c_j) corner-pairs from K_e that contribute. See proposal derivation.
#
# Andreassen-Andreasen corner ordering: 0=LL, 1=LR, 2=UR, 3=UL.

_CONTRIB_TABLE: Dict[Tuple[int, int], list] = {
    ( 0,  0): [(2, 2), (3, 3), (0, 0), (1, 1)],   # self
    (+1,  0): [(3, 2), (0, 1)],                    # east
    (-1,  0): [(2, 3), (1, 0)],                    # west
    ( 0, +1): [(0, 3), (1, 2)],                    # north
    ( 0, -1): [(3, 0), (2, 1)],                    # south
    (+1, +1): [(0, 2)],                            # NE diagonal
    (-1, -1): [(2, 0)],                            # SW
    (+1, -1): [(3, 1)],                            # SE
    (-1, +1): [(1, 3)],                            # NW
}


def _corner_block(Ke: np.ndarray, ci: int, cj: int) -> np.ndarray:
    """Extract the 2x2 sub-block K_e[2*ci:2*ci+2, 2*cj:2*cj+2]."""
    return Ke[2 * ci : 2 * ci + 2, 2 * cj : 2 * cj + 2]


def coupling_blocks(
    lam: float = 1.0,
    mu: float = 1.0,
    a: float = 0.5,
    b: float = 0.5,
    phi_deg: float = 90.0,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Compute the nine 2x2 inter-node coupling blocks C_{dx, dy}.

    Returns a dict mapping (dx, dy) in {-1, 0, 1}^2 to a 2x2 numpy array.
    """
    Ke_lam, Ke_mu, _, _ = element_template(a=a, b=b, phi_deg=phi_deg)
    Ke = lam * Ke_lam + mu * Ke_mu

    blocks: Dict[Tuple[int, int], np.ndarray] = {}
    for delta, contribs in _CONTRIB_TABLE.items():
        block = np.zeros((2, 2), dtype=Ke.dtype)
        for ci, cj in contribs:
            block += _corner_block(Ke, ci, cj)
        blocks[delta] = block
    return blocks


# ---------------------------------------------------------------------------
# Sanity checks on the coupling blocks (called by tests)
# ---------------------------------------------------------------------------

def check_symmetry(blocks: Dict[Tuple[int, int], np.ndarray]) -> float:
    """Verify C_{-dx, -dy} = C_{dx, dy}^T (consequence of K_e symmetry).

    Returns the max absolute deviation across all four off-diagonal pairs.
    """
    pairs = [((+1, 0), (-1, 0)), ((0, +1), (0, -1)),
             ((+1, +1), (-1, -1)), ((+1, -1), (-1, +1))]
    max_err = 0.0
    for d_pos, d_neg in pairs:
        err = float(np.max(np.abs(blocks[d_pos].T - blocks[d_neg])))
        max_err = max(max_err, err)
    return max_err


def check_zero_row_sum(blocks: Dict[Tuple[int, int], np.ndarray]) -> float:
    """Verify sum_{delta} C_{delta} = 0 (rigid-translation null space).

    The stencil's coupling sums must vanish so that constant displacement
    fields produce zero force.  Returns the max absolute entry of the sum.
    """
    total = sum(blocks.values())
    return float(np.max(np.abs(total)))


# ---------------------------------------------------------------------------
# The Fourier symbol
# ---------------------------------------------------------------------------

def Khat0(
    M: int,
    lam: float = 1.0,
    mu: float = 1.0,
    a: float | None = None,
    b: float | None = None,
    phi_deg: float = 90.0,
) -> np.ndarray:
    """Compute the per-mode 2x2 Fourier symbol Khat_0(k) for all k in [0, M)^2.

    Returns array of shape (M, M, 2, 2) with Khat0[kx, ky] = the 2x2 symbol
    at Fourier mode (kx, ky).

    Element half-widths default to 1/(2M) (unit cell partitioned into M x M
    equal squares), matching the convention used elsewhere in qhomogenize.
    """
    if a is None:
        a = 1.0 / (2 * M)
    if b is None:
        b = 1.0 / (2 * M)

    blocks = coupling_blocks(lam=lam, mu=mu, a=a, b=b, phi_deg=phi_deg)
    Khat = np.zeros((M, M, 2, 2), dtype=complex)
    for kx in range(M):
        for ky in range(M):
            for (dx, dy), C in blocks.items():
                phase = np.exp(2j * np.pi * (kx * dx + ky * dy) / M)
                Khat[kx, ky] += C * phase
    return Khat


def Khat0_inv(
    M: int,
    lam: float = 1.0,
    mu: float = 1.0,
    a: float | None = None,
    b: float | None = None,
    phi_deg: float = 90.0,
) -> np.ndarray:
    """Compute the per-mode 2x2 inverse Khat_0^{-1}(k) for all k in [0, M)^2.

    Zero-mode handling: Khat0_inv[0, 0] is set to the zero matrix to
    implement the Moore-Penrose pseudoinverse on the zero-mean subspace
    (rigid translations are projected out).

    Returns array of shape (M, M, 2, 2).
    """
    Khat = Khat0(M, lam=lam, mu=mu, a=a, b=b, phi_deg=phi_deg)
    Khat_inv = np.zeros_like(Khat)
    for kx in range(M):
        for ky in range(M):
            if (kx, ky) == (0, 0):
                Khat_inv[0, 0] = 0.0
                continue
            Khat_inv[kx, ky] = np.linalg.inv(Khat[kx, ky])
    return Khat_inv


# ---------------------------------------------------------------------------
# Building K_0 from the symbol via inverse DFT (for testing the derivation)
# ---------------------------------------------------------------------------

def K0_from_symbol(M: int, **kwargs) -> np.ndarray:
    """Reconstruct K_0 of size (2 M^2, 2 M^2) by inverse DFT of the symbol.

    Convention: the global DOF index j is decomposed as j = 2*node + dim,
    where node = nx + M*ny and dim in {0, 1}.  This matches qhomogenize's
    layout where the dim bit is bit 0 of the global DOF.
    """
    Khat = Khat0(M, **kwargs)  # (M, M, 2, 2)

    N = 2 * M * M  # total DOFs
    K = np.zeros((N, N), dtype=complex)

    # Build by inverse DFT:
    #   K[node1, dim1; node2, dim2]
    #     = (1/M^2) * sum_{kx, ky} exp(2*pi*i * (kx*(n1x-n2x) + ky*(n1y-n2y))/M)
    #                              * Khat[kx, ky][dim1, dim2]
    # where node = nx + M*ny.
    for n1x in range(M):
        for n1y in range(M):
            n1 = n1x + M * n1y
            for n2x in range(M):
                for n2y in range(M):
                    n2 = n2x + M * n2y
                    block = np.zeros((2, 2), dtype=complex)
                    for kx in range(M):
                        for ky in range(M):
                            phase = np.exp(2j * np.pi * (kx * (n1x - n2x) + ky * (n1y - n2y)) / M)
                            block += phase * Khat[kx, ky]
                    block /= M * M
                    K[2 * n1 : 2 * n1 + 2, 2 * n2 : 2 * n2 + 2] = block

    return K  # complex due to numerical noise; test will check imag part is ~0


# ---------------------------------------------------------------------------
# Self-tests / module __main__ block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Symbol derivation: closed-form Fourier symbol of K_0")
    print("=" * 72)

    # 1. Coupling blocks satisfy C_{-d} = C_{d}^T
    print("\n--- 1. Symmetry of coupling blocks ---")
    a = 0.25  # arbitrary, M=2 default
    blocks = coupling_blocks(lam=1.0, mu=1.0, a=a, b=a, phi_deg=90.0)
    err_sym = check_symmetry(blocks)
    print(f"  max |C_{{-d}} - C_{{d}}^T| = {err_sym:.3e}")
    assert err_sym < 1e-12, "Coupling blocks fail symmetry check"

    # 2. Stencil sums to zero (rigid-translation null space)
    print("\n--- 2. Zero row sum ---")
    err_zero = check_zero_row_sum(blocks)
    print(f"  max |sum C_{{delta}}| = {err_zero:.3e}")
    assert err_zero < 1e-12, "Coupling blocks fail zero-sum check"

    # 3. The big test: build K_0 by direct assembly and via inverse DFT
    print("\n--- 3. K_0 via direct assembly vs inverse DFT of symbol ---")
    from qhomogenize.fea.full_K import assemble_K

    for M in [2, 4]:
        a_h = 1.0 / (2 * M)
        K_direct = assemble_K(M, p=np.ones(M * M), lam=1.0, mu=1.0, phi_deg=90.0).toarray()
        K_dft = K0_from_symbol(M, lam=1.0, mu=1.0, a=a_h, b=a_h, phi_deg=90.0)

        # K_dft is complex due to DFT roundoff; should be near-real
        max_imag = float(np.max(np.abs(K_dft.imag)))
        err_real = float(np.max(np.abs(K_dft.real - K_direct)))
        print(f"  M={M}: shape {K_direct.shape}, max imag part = {max_imag:.3e}, "
              f"max |K_dft - K_direct| = {err_real:.3e}")
        assert max_imag < 1e-10, f"K_dft has nontrivial imaginary part at M={M}"
        assert err_real < 1e-10, f"K_dft does not match K_direct at M={M}"

    # 4. Symbol inverse on zero-mean subspace
    print("\n--- 4. Khat0_inv: zero mode handling ---")
    M = 4
    Khat_inv = Khat0_inv(M, lam=1.0, mu=1.0)
    print(f"  Khat0_inv[0, 0] = \n{Khat_inv[0, 0]}")
    print(f"  ||Khat0_inv[0, 0]|| = {np.linalg.norm(Khat_inv[0, 0]):.3e}  (should be 0)")
    assert np.linalg.norm(Khat_inv[0, 0]) < 1e-12, "Zero mode not zeroed out"

    # 5. Khat0 * Khat0_inv = I on nonzero modes
    print("\n--- 5. Khat0 * Khat0_inv = I on nonzero modes ---")
    Khat = Khat0(M, lam=1.0, mu=1.0)
    max_err = 0.0
    for kx in range(M):
        for ky in range(M):
            if (kx, ky) == (0, 0):
                continue
            err = float(np.max(np.abs(Khat[kx, ky] @ Khat_inv[kx, ky] - np.eye(2))))
            max_err = max(max_err, err)
    print(f"  max |Khat * Khat_inv - I| over nonzero modes: {max_err:.3e}")
    assert max_err < 1e-10, "Symbol inverse failed"

    print("\nAll symbol derivation tests passed.")

"""
Full assembled stiffness operator K = A^T D_p A.

Numpy/scipy reference for the operator we will block-encode in tasks 1-3.
Used as ground truth in `verify_against` calls.

The decomposition K = A^T D_p A with the Lamé split

    K = lambda * (A^T D_p^lambda A) + mu * (A^T D_p^mu A),

    D_p^lambda = diag_e(p_e * Ke_lambda),
    D_p^mu     = diag_e(p_e * Ke_mu),

is the structure that the quantum side will mirror.  This module assembles
each piece individually so the quantum implementation can validate against
each layer (D_K^lambda alone, D_p^lambda alone, A^T D_p^lambda A alone, full K).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, block_diag, diags

from qhomogenize.fea.assembly_pbc import assembly_matrix, n_dofs
from qhomogenize.fea.element_template import element_template_for_mesh
from qhomogenize.fea.microstructure import p_vector


def D_K(L: int, Ke: np.ndarray) -> csr_matrix:
    """Block-diagonal D_K = I_{L^2} ⊗ Ke (L^2 copies of the 8x8 element template).

    Parameters
    ----------
    L : int
    Ke : ndarray, shape (8, 8)

    Returns
    -------
    D_K : csr_matrix, shape (8 L^2, 8 L^2)
    """
    blocks = [csr_matrix(Ke)] * (L * L)
    return block_diag(blocks, format="csr")


def M_p(L: int, p: np.ndarray) -> csr_matrix:
    """Diagonal mask M_p = diag(p) ⊗ I_8.

    p[e] in {0, 1} indicates whether element e is solid.
    """
    if p.shape != (L * L,):
        raise ValueError(f"p must have shape ({L*L},), got {p.shape}")
    diag_vals = np.repeat(p.astype(float), 8)
    return diags(diag_vals, format="csr")


def D_p(L: int, p: np.ndarray, Ke: np.ndarray) -> csr_matrix:
    """D_p = D_K · M_p = M_p · D_K = block-diag(p_e * Ke).

    The mask and block-diagonal commute since both are diagonal in the
    block-element index.
    """
    return D_K(L, Ke) @ M_p(L, p)


def assemble_K_layer(L: int, p: np.ndarray, Ke: np.ndarray) -> csr_matrix:
    """Assemble A^T D_p A for one Lamé layer, given the layer-specific Ke."""
    A = assembly_matrix(L)  # shape (8 L^2, 2 L^2)
    Dp = D_p(L, p, Ke)
    K = A.T @ Dp @ A
    return K


def assemble_K(
    L: int,
    p: np.ndarray,
    lam: float,
    mu: float,
    phi_deg: float = 90.0,
) -> csr_matrix:
    """Full assembled stiffness K = lam * A^T D_p^lambda A + mu * A^T D_p^mu A.

    Parameters
    ----------
    L : int
    p : ndarray of shape (L^2,)
        Element-presence indicator p_e in {0, 1}.
    lam, mu : float
        Lamé parameters of the solid phase.  (Void phase has Lamé = 0.)
    phi_deg : float, default 90.0
        Cell-wall angle.

    Returns
    -------
    K : csr_matrix, shape (2 L^2, 2 L^2)
    """
    Kl, Km, _, _ = element_template_for_mesh(L, phi_deg)
    Kl_layer = assemble_K_layer(L, p, Kl)
    Km_layer = assemble_K_layer(L, p, Km)
    return (lam * Kl_layer + mu * Km_layer).tocsr()


# ---------------------------------------------------------------------------
# Quick test: shape, symmetry, PSD, and rigid-translation null space under PBC
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    L = 4
    p = p_vector(L, cx=0.5, cy=0.5, r=0.3)
    print(f"L={L}, vol fraction = {p.sum() / (L*L):.3f}")

    K = assemble_K(L, p, lam=1.0, mu=1.0)
    N_global, _ = n_dofs(L)
    assert K.shape == (N_global, N_global)

    # symmetry
    asymm = (K - K.T)
    assert np.max(np.abs(asymm.toarray())) < 1e-12, "K not symmetric"

    # PSD
    eigs = np.sort(np.linalg.eigvalsh(K.toarray()))
    assert eigs[0] > -1e-9, f"K not PSD: smallest eig = {eigs[0]}"

    # Rigid-translation null space under PBC: u_x = const everywhere is a zero mode,
    # u_y = const everywhere is a zero mode.  So at least 2 zero eigenvalues.
    n_zero = int((np.abs(eigs) < 1e-9).sum())
    print(f"K shape {K.shape}, rank deficiency = {n_zero}, smallest 5 eigs = {np.round(eigs[:5], 6)}")
    assert n_zero >= 2, f"expected >=2 zero modes from rigid-translation null space, got {n_zero}"

    # Verify by direct construction of the rigid-translation modes
    e_x = np.zeros(N_global)
    e_x[0::2] = 1.0  # u_x = 1 on every node
    e_x /= np.linalg.norm(e_x)
    e_y = np.zeros(N_global)
    e_y[1::2] = 1.0
    e_y /= np.linalg.norm(e_y)
    res_x = K @ e_x
    res_y = K @ e_y
    assert np.linalg.norm(res_x) < 1e-12, f"rigid-x not null: {np.linalg.norm(res_x)}"
    assert np.linalg.norm(res_y) < 1e-12, f"rigid-y not null: {np.linalg.norm(res_y)}"
    print("Rigid-translation null space verified  OK")

    # Sanity check the layer decomposition vs. classical FEA driver from the project
    # (skipped here -- it is exercised in tests/test_against_classical.py instead).
    print(f"||K||_2 ~ {eigs[-1]:.4f}")
    print("OK")

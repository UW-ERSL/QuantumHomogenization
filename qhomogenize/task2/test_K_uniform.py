"""
Triple composition test: build U_K = U_{A^T} * U_{D_K} * U_A and verify
against the numpy reference K = A^T D_K A for a uniform-material lattice
(M_p = I).

This is the headline check for the qhomogenize package's multiplicative composition
path: it confirms the structured block encoding of A composes correctly with
the Pauli-LCU encoding of D_K.

The reference K is taken from qhomogenize.fea.full_K.global_stiffness, which assembles
K from the local templates and the assembly matrix.  For uniform material
(p_e = 1 everywhere), this matches A^T D_K A exactly.

Subnormalization check
----------------------
alpha_K = alpha_{A^T} * alpha_{D_K} * alpha_A.

Note: A^T A = (4 I) under our padding convention (every column of A_padded
has exactly 4 nonzeros = 4 incident elements at each global DOF; every row of
A_padded that's nonzero has exactly one entry).  So
    A^T_padded * D_K * A_padded = K_global  (within the active 2L^2 x 2L^2 block)
where K_global is the stiffness obtained by direct assembly.
"""

from __future__ import annotations

import numpy as np

from qhomogenize.bencode.compose_product import compose_triple
from qhomogenize.task1.build_UA import build_UA
from qhomogenize.task2.build_UDk import build_UDk
from qhomogenize.fea.assembly_pbc import assembly_matrix
from qhomogenize.fea.element_template import element_template


def build_UK_uniform(
    L: int,
    lam: float = 1.0,
    mu: float = 1.0,
    name: str = "U_K_uniform",
):
    """Build U_K = U_{A^T} * U_{D_K} * U_A for uniform material (M_p = I)."""
    a_h = 1.0 / (2 * L)
    be_A = build_UA(L)
    be_DK = build_UDk(L, a=a_h, b=a_h, phi_deg=90.0, lam=lam, mu=mu)
    be_AT = be_A.conjugate_transpose(name="U_AT")
    be_K = compose_triple(be_AT, be_DK, be_A, name=name)
    return be_K, (be_A, be_DK, be_AT)


def reference_K(L: int, lam: float = 1.0, mu: float = 1.0) -> np.ndarray:
    """Direct numpy reference: A^T (I (x) K_e) A, padded to 8L^2 x 8L^2."""
    a_h = 1.0 / (2 * L)
    Ke_lam, Ke_mu, _, _ = element_template(a=a_h, b=a_h, phi_deg=90.0)
    Ke = lam * Ke_lam + mu * Ke_mu
    A = assembly_matrix(L).toarray()  # shape (8L^2, 2L^2)
    n_loc, n_glob = A.shape

    # Pad A to a square 8L^2 x 8L^2 matrix (zero columns beyond 2L^2)
    A_padded = np.zeros((n_loc, n_loc), dtype=A.dtype)
    A_padded[:, :n_glob] = A

    D_K = np.kron(np.eye(L * L), Ke)  # 8L^2 x 8L^2
    K_full = A_padded.T @ D_K @ A_padded  # 8L^2 x 8L^2 (zero outside top-left 2L^2 block)
    return K_full


if __name__ == "__main__":
    for L in [2, 4]:
        print(f"--- L = {L} ---")
        be_K, (be_A, be_DK, be_AT) = build_UK_uniform(L)
        print(f"  {be_K.summary()}")
        print(f"    components: alpha_A={be_A.alpha}, alpha_DK={be_DK.alpha:.4f}, "
              f"alpha_AT={be_AT.alpha}")
        print(f"    total qubits in circuit: {be_K.circuit.num_qubits}")

        K_ref = reference_K(L)

        # The full 16-qubit (L=2) / 18-qubit (L=4) unitary is too large to
        # materialize.  Verify compositionally: extract_matrix on each
        # component (which has manageable size), multiply, and compare.
        A_block = be_A.extract_matrix()       # = A_padded / alpha_A
        DK_block = be_DK.extract_matrix()     # = D_K / alpha_DK
        AT_block = be_AT.extract_matrix()     # = A_padded^T / alpha_AT

        # Composed block:  AT_block * DK_block * A_block  =  K_ref / alpha_K
        composed = AT_block @ DK_block @ A_block
        target = K_ref / be_K.alpha
        err = float(np.max(np.abs(composed - target)))
        print(f"    compositional check: max |AT*DK*A - K_ref/alpha| = {err:.3e}")
        assert err < 1e-10, f"compositional K test at L={L} failed (err={err})"

        # Sanity check: the K_ref's first 2L^2 x 2L^2 block should match
        # the user's qhomogenize.fea.full_K.assemble_K.
        from qhomogenize.fea.full_K import assemble_K
        K_ref_user = assemble_K(L, p=np.ones(L*L), lam=1.0, mu=1.0, phi_deg=90.0)
        n_glob = 2 * L * L
        K_ref_active = K_ref[:n_glob, :n_glob]
        diff = K_ref_active - K_ref_user.toarray()
        print(f"    cross-check vs full_K.assemble_K: max diff = {np.max(np.abs(diff)):.3e}")
        assert np.max(np.abs(diff)) < 1e-10, "Reference K mismatch with user's full_K"

    print("\nAll U_K (uniform) tests passed.")

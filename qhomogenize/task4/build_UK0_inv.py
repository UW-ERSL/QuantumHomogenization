"""
Composed block encoding of K_0^{-1}:

    U_{K_0^{-1}}  =  U_F^dagger  *  U_{Khat0^{-1}}  *  U_F

where
    U_F                  : spatial QFT (2D QFT on the node register, identity on dim)
    U_{Khat0^{-1}}       : block-diagonal in (kx, ky), applies the 2x2 inverse symbol
    U_F^dagger           : inverse spatial QFT

The composition is done via the existing product-of-block-encodings calculus.

Caveats and notes
-----------------
1. U_F does not naturally fit the BlockEncoding type because it has no ancilla
   (it's a pure unitary on the system register).  We wrap it as a BlockEncoding
   with n_ancilla=0, alpha=1.  This is the trivial case where the encoded
   "block" is the entire unitary.

2. The composition U_F^dag * U_Khat * U_F should reproduce K_0^{-1} on the
   zero-mean subspace.  At the zero mode, the Khat^{-1} entry is zero, so
   the composition projects out rigid translations.

3. The integration test (U_K0_inv * U_K = I on zero-mean subspace) is the
   key correctness check; it lives in tests/test_aim2.py.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from qhomogenize.bencode.types import BlockEncoding
from qhomogenize.bencode.compose_product import compose_triple
from qhomogenize.task4.spatial_qft import spatial_qft, spatial_qft_matrix
from qhomogenize.task4.build_USymbolInv import build_USymbolInv, Khat0_inv_diagonal_matrix
from qhomogenize.task4.symbol import Khat0_inv


def _wrap_unitary_as_block_encoding(
    qc: QuantumCircuit,
    name: str,
    target_matrix: np.ndarray | None = None,
) -> BlockEncoding:
    """Wrap a no-ancilla unitary circuit as a trivial BlockEncoding.

    For the QFT, the encoded "block" is the full unitary itself; n_ancilla=0,
    alpha=1.

    Returns a BlockEncoding with n_system = qc.num_qubits.
    """
    n_sys = qc.num_qubits
    target_shape = target_matrix.shape if target_matrix is not None else (2 ** n_sys, 2 ** n_sys)
    return BlockEncoding(
        circuit=qc,
        n_system=n_sys,
        n_ancilla=0,
        alpha=1.0,
        name=name,
        target_shape=target_shape,
    )


def build_UF_block(M: int) -> BlockEncoding:
    """Wrap the spatial QFT as a BlockEncoding with no ancilla."""
    qc = spatial_qft(M, name="U_F")
    target = spatial_qft_matrix(M)
    return _wrap_unitary_as_block_encoding(qc, name="U_F", target_matrix=target)


def build_UF_dag_block(M: int) -> BlockEncoding:
    """Wrap the inverse spatial QFT as a BlockEncoding with no ancilla."""
    qc = spatial_qft(M, name="_F").inverse()
    qc.name = "U_Fdag"
    target = spatial_qft_matrix(M).conj().T
    return _wrap_unitary_as_block_encoding(qc, name="U_Fdag", target_matrix=target)


def build_UK0_inv(M: int, name: str = "U_K0inv", **symbol_kwargs) -> BlockEncoding:
    """Build U_{K_0^{-1}} as the triple product U_F^dag * U_{Khat^{-1}} * U_F.

    Parameters
    ----------
    M : int
    name : str
    **symbol_kwargs : passed to Khat0_inv (lam, mu, a, b, phi_deg).

    Returns
    -------
    BlockEncoding with n_system = 1 + 2*log2(M), n_ancilla = (whatever
    U_{Khat0^{-1}} needed), target_shape = (2 M^2, 2 M^2).

    Subnormalization: alpha_F * alpha_Khat * alpha_Fdag = 1 * alpha_Khat * 1
                   = alpha_Khat.
    """
    be_F = build_UF_block(M)
    be_Fdag = build_UF_dag_block(M)
    be_Khat_inv = build_USymbolInv(M, name="U_Khat_inv", **symbol_kwargs)

    # Verify register widths match before composing
    assert be_F.n_system == be_Khat_inv.n_system, (
        f"register width mismatch: U_F has {be_F.n_system} sys qubits, "
        f"U_Khat_inv has {be_Khat_inv.n_system}"
    )

    # Triple product: A * B * C  with A = U_F^dag, B = U_Khat_inv, C = U_F
    return compose_triple(be_Fdag, be_Khat_inv, be_F, name=name)


# ---------------------------------------------------------------------------
# Numpy reference for the encoded operator
# ---------------------------------------------------------------------------

def K0_inv_reference(M: int, **symbol_kwargs) -> np.ndarray:
    """Numpy reference: K_0^{-1} on the zero-mean subspace, of size (2 M^2, 2 M^2).

    Uses the same convention as elsewhere: j = 2*node + dim, node = nx + M*ny.
    """
    F = spatial_qft_matrix(M)
    Khat_inv_op = Khat0_inv_diagonal_matrix(M, **symbol_kwargs).real
    K0_inv = F.conj().T @ Khat_inv_op @ F
    # Should be Hermitian and real (or near-real) up to numerics
    return K0_inv


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Composed preconditioner U_{K_0^{-1}} = U_F^dag U_{Khat^{-1}} U_F")
    print("=" * 72)

    for M in [2, 4]:
        print(f"\n--- M = {M} ---")
        be = build_UK0_inv(M)
        print(f"  {be.summary()}")
        print(f"  total qubits in circuit: {be.circuit.num_qubits}")

        # Reference
        K0_inv_ref = K0_inv_reference(M)
        K0_inv_real = K0_inv_ref.real
        print(f"  K0_inv ref: shape {K0_inv_ref.shape}, "
              f"max imag = {np.max(np.abs(K0_inv_ref.imag)):.3e}")

        if M <= 2:
            # Full extract is fast (6 qubits): use extract_matrix.
            U = be.extract_matrix()
            err = float(np.max(np.abs(U - K0_inv_real / be.alpha)))
            print(f"  full extract check: max |U - K0_inv/alpha| = {err:.3e}")
            assert err < 1e-10
        else:
            # 10+ qubits: spot-check several columns via Statevector simulation.
            from qiskit.quantum_info import Statevector
            from qiskit import QuantumCircuit

            n_sys = be.n_system
            n_anc = be.n_ancilla
            N = 2 ** n_sys
            errors = []
            test_indices = [0, 1, 5, 17, N - 1]
            for j in test_indices:
                qc_test = QuantumCircuit(n_sys + n_anc)
                for bit in range(n_sys):
                    if (j >> bit) & 1:
                        qc_test.x(bit)
                qc_test.compose(be.circuit, inplace=True)
                sv = Statevector.from_instruction(qc_test).data
                column = sv[:N]  # ancilla = 0 occupies the first N entries
                expected = K0_inv_real[:, j] / be.alpha
                err_col = float(np.max(np.abs(column - expected)))
                errors.append(err_col)
            print(f"  spot-check on {len(test_indices)} columns: max err = {max(errors):.3e}")
            assert max(errors) < 1e-10

    print("\nAll build_UK0_inv tests passed.")

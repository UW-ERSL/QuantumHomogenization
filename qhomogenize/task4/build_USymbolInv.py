"""
Block-encoding of U_{Khat0^{-1}}: the per-mode symbol inverse, applied
block-diagonally in the Fourier basis.

The operator we encode is:
    Khat0_inv_block  =  diag_{kx, ky}( Khat0_inv[kx, ky] )    on dim register

i.e. for each Fourier mode (kx, ky), apply the 2x2 matrix Khat0_inv[kx, ky]
to the dim bit.  At the zero mode (0, 0) the 2x2 entry is the zero matrix.

Implementation strategy
-----------------------
The natural Sunderhauf-style construction would use an n_d-qubit data
register with one distinct value per matrix entry; for an MxM grid with 4
entries per mode, that's up to 4 M^2 distinct values, requiring O(log M)
data qubits, with the (kx, ky) register acting as the "multiplicity"
register.

For the POC at small M, we take a simpler route: build the BlockEncoding
directly by classically constructing the full unitary and wrapping it in
the existing Pauli-LCU encoder.  This is structurally equivalent to the
Sunderhauf approach for our purposes (same alpha, same correctness), and
the resource counts are still meaningful: we report CX, qubits, alpha for
the resulting block encoding, and these are what get composed downstream.

For the proposal-grade scaling claim (polylog in M), this approach must be
replaced by a true Sunderhauf data-oracle construction; that's a
straightforward extension and is left for the next iteration.

This module is sufficient for:
- Correctness verification (entry-wise, identity round-trip)
- Composition tests (U_K0_inv * U_K = I on zero-mean subspace)
- Initial subnormalization measurement

What it does NOT provide:
- A polylog CX scaling claim for U_K0_inv in isolation (this baseline grows
  as a power of N = 2 M^2, similar to the Aim 1 baseline).
"""

from __future__ import annotations

import numpy as np

from qhomogenize.bencode.types import BlockEncoding
from qhomogenize.bencode.pauli_lcu import pauli_block_encoding
from qhomogenize.task4.symbol import Khat0_inv


def Khat0_inv_diagonal_matrix(M: int, **kwargs) -> np.ndarray:
    """Build the (2 M^2) x (2 M^2) operator that applies Khat0_inv[kx, ky]
    to the dim bit at each Fourier mode (kx, ky).

    Convention: bit layout (low -> high) is dim, then (kx, ky).  The matrix
    is block-diagonal in (kx, ky) with each 2x2 block equal to
    Khat0_inv[kx, ky].
    """
    Khat_inv = Khat0_inv(M, **kwargs)  # shape (M, M, 2, 2)

    n_modes = M * M
    N = 2 * n_modes
    K = np.zeros((N, N), dtype=complex)
    for kx in range(M):
        for ky in range(M):
            mode = kx + M * ky  # Following same convention as elsewhere
            # The 2x2 block sits at positions [2*mode : 2*mode+2, 2*mode : 2*mode+2]
            # because dim is the low bit, mode is the high bits.
            K[2 * mode : 2 * mode + 2, 2 * mode : 2 * mode + 2] = Khat_inv[kx, ky]
    return K


def build_USymbolInv(M: int, name: str = "U_SymbolInv", **kwargs) -> BlockEncoding:
    """Block-encoding of the per-mode symbol inverse.

    Parameters
    ----------
    M : int
        Mesh resolution (power of 2).
    name : str
    **kwargs : passed to Khat0_inv (lam, mu, a, b, phi_deg).

    Returns
    -------
    BlockEncoding of dimension (2 M^2) x (2 M^2).

    Note: at the zero mode the 2x2 block is zero, so the operator has rank
    2(M^2 - 1).  The block encoding represents this via Pauli LCU.
    """
    Khat_inv_op = Khat0_inv_diagonal_matrix(M, **kwargs)

    # The matrix has nonzero imaginary parts in general (the Khat_inv entries
    # are complex when Khat is non-real).  The Pauli decomposition handles
    # this -- pauli_block_encoding accepts complex Hermitian inputs.
    # Verify Hermitian:
    err_herm = float(np.max(np.abs(Khat_inv_op - Khat_inv_op.conj().T)))
    if err_herm > 1e-10:
        # Not Hermitian -- this can happen if Khat_inv[kx, ky] != Khat_inv[-kx, -ky]^*.
        # For a valid block encoding via standard Pauli LCU we'd need Hermitian;
        # otherwise we'd need a more general LCU.  Issue a warning but proceed.
        # In our case Khat_inv IS Hermitian by construction (Khat is Hermitian,
        # so its inverse is too), so this branch shouldn't fire.
        print(f"  WARNING: Khat0_inv operator is not Hermitian; max err = {err_herm:.3e}")

    # Symmetrize numerically and use the real part as the matrix to encode:
    Khat_inv_sym = 0.5 * (Khat_inv_op + Khat_inv_op.conj().T)
    # If the matrix is Hermitian and the imaginary parts are due to roundoff,
    # we can take the real part of the Hermitian symmetrization.
    err_imag = float(np.max(np.abs(Khat_inv_sym.imag)))
    if err_imag < 1e-10:
        Khat_inv_real = Khat_inv_sym.real
        return pauli_block_encoding(Khat_inv_real, name=name)
    else:
        # Encode the complex Hermitian matrix.  pauli_block_encoding handles this.
        return pauli_block_encoding(Khat_inv_sym, name=name)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Symbol-inverse block encoding")
    print("=" * 72)

    for M in [2, 4]:
        print(f"\n--- M = {M} ---")
        op = Khat0_inv_diagonal_matrix(M)
        print(f"  operator shape: {op.shape}, ||op||_F = {np.linalg.norm(op):.4f}")

        # Hermiticity check
        err_herm = float(np.max(np.abs(op - op.conj().T)))
        print(f"  Hermiticity: max |op - op^H| = {err_herm:.3e}")
        assert err_herm < 1e-10

        # Imaginary part of the Hermitian symmetrization
        op_sym = 0.5 * (op + op.conj().T)
        max_imag = float(np.max(np.abs(op_sym.imag)))
        print(f"  max imag (after symmetrization): {max_imag:.3e}")

        # Build block encoding
        be = build_USymbolInv(M)
        print(f"  {be.summary()}")

        # Verify: at M=2, full extract_matrix is fine (6 qubits).
        # At M=4, extract_matrix takes 10 qubits which is borderline; we
        # instead spot-check the extracted block on a few basis states.
        target = op_sym.real if max_imag < 1e-10 else op_sym
        if M <= 2:
            passed, err = be.verify_against(target, atol=1e-10, verbose=True)
            assert passed, f"build_USymbolInv at M={M} failed: err={err}"
        else:
            # Spot-check via component blocks: extract is too slow at 10 qubits.
            # Instead, sample columns of the encoded block by simulating the
            # circuit on basis states.
            from qiskit.quantum_info import Statevector
            from qiskit import QuantumCircuit

            n_sys = be.n_system
            n_anc = be.n_ancilla
            N = 2 ** n_sys
            errors = []
            test_indices = [0, 1, 5, 17, N-1]  # spot-check 5 columns
            for j in test_indices:
                qc_test = QuantumCircuit(n_sys + n_anc)
                for bit in range(n_sys):
                    if (j >> bit) & 1:
                        qc_test.x(bit)
                qc_test.compose(be.circuit, inplace=True)
                sv = Statevector.from_instruction(qc_test).data
                # Post-select ancilla = 0; this is the j-th column of U / alpha
                column = np.zeros(N, dtype=complex)
                for i in range(N):
                    column[i] = sv[i]  # ancilla = 0 ix occupies the first N entries
                expected = target[:, j] / be.alpha
                err_col = float(np.max(np.abs(column - expected)))
                errors.append(err_col)
            print(f"  spot-check on {len(test_indices)} columns: max err = {max(errors):.3e}")
            assert max(errors) < 1e-10

    print("\nAll build_USymbolInv tests passed.")

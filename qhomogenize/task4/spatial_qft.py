"""
Vector-valued spatial QFT for the Aim 2 preconditioner.

Acts as a 2D QFT on the node register (n_log_M qubits per axis, 2 n_log_M
qubits total) while leaving the displacement-component register (the dim
bit, 1 qubit) untouched.

Convention (matching qhomogenize's bit layout for the global DOF register):
    bit 0:                              dim (in [0, 2))
    bits 1 to n_log_M:                  nx (in [0, M))
    bits (n_log_M + 1) to (2 n_log_M):  ny (in [0, M))

The spatial QFT applies QFT to the 2 n_log_M-qubit (nx, ny) sub-register
exactly as a tensor product of two 1D QFTs (each on n_log_M qubits).  The dim
bit is left untouched.

Convention for the QFT itself: we use the standard Qiskit definition,
    QFT |x> = (1/sqrt(M)) sum_{k=0}^{M-1} exp(2 pi i k x / M) |k>

i.e. the inverse of `np.fft.fft` (which uses exp(-2 pi i k x / M)).  We make
this explicit in tests against numpy.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT


def spatial_qft(M: int, name: str = "U_F") -> QuantumCircuit:
    """Build the vector-valued spatial QFT on (1 + 2 n_log_M) qubits.

    Parameters
    ----------
    M : int
        Mesh resolution (must be a power of 2).

    Returns
    -------
    qc : QuantumCircuit
        Circuit on (1 + 2 n_log_M) qubits.  Bit 0 is dim (untouched);
        bits 1..n_log_M are the nx register (1D QFT applied);
        bits (n_log_M+1)..(2 n_log_M) are the ny register (1D QFT applied).
    """
    n_log_M = int(np.log2(M))
    if 2 ** n_log_M != M:
        raise NotImplementedError(f"M must be a power of 2; got M={M}")

    n_total = 1 + 2 * n_log_M
    qc = QuantumCircuit(n_total, name=name)

    # Build the 1D QFT and apply to nx, then ny.
    # Note: Qiskit's QFT class applies a SWAP at the end by default to put
    # the result in conventional bit-ordering.  We use do_swaps=True (default).
    qft1d = QFT(n_log_M, do_swaps=True)

    # Apply 1D QFT to nx register (bits 1..n_log_M)
    qc.append(qft1d, list(range(1, 1 + n_log_M)))
    # Apply 1D QFT to ny register (bits n_log_M+1..2*n_log_M)
    qc.append(qft1d, list(range(1 + n_log_M, 1 + 2 * n_log_M)))

    return qc


def spatial_qft_inverse(M: int, name: str = "U_Fdag") -> QuantumCircuit:
    """Inverse spatial QFT.  Returns the QFT^dagger circuit."""
    return spatial_qft(M, name="_temp").inverse().to_gate(label=name).definition


# ---------------------------------------------------------------------------
# Numpy reference for verification
# ---------------------------------------------------------------------------

def spatial_qft_matrix(M: int) -> np.ndarray:
    """Numpy reference matrix for spatial_qft, shape (2 M^2, 2 M^2).

    Convention (matching the quantum circuit's bit layout):
        global DOF index j = dim + 2 * nx + 2 * M * ny
        QFT acts as F_{kx, nx} (x) F_{ky, ny} on the spatial part,
        identity on the dim bit.

    F is the M x M DFT matrix with entries F[k, n] = exp(+2 pi i k n / M) / sqrt(M),
    matching Qiskit's QFT convention.
    """
    n_log_M = int(np.log2(M))
    if 2 ** n_log_M != M:
        raise NotImplementedError(f"M must be a power of 2; got M={M}")

    # 1D DFT matrix (Qiskit convention: + sign in exponent)
    n_grid = np.arange(M)
    F = np.exp(2j * np.pi * np.outer(n_grid, n_grid) / M) / np.sqrt(M)

    # Spatial QFT: F (x) F on the (nx, ny) part
    F2D = np.kron(F, F)  # shape (M^2, M^2)

    # Tensor with identity on the dim bit (low bit -> right factor in Qiskit)
    # In Qiskit's convention, the operator on the full register is (high (x) low):
    #   full = F2D_on_node (x) I_dim
    # because dim is the LOW bit and node is HIGH.
    U = np.kron(F2D, np.eye(2))
    return U


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from qiskit.quantum_info import Operator

    print("=" * 72)
    print("Spatial QFT: vector-valued 2D QFT on the node register")
    print("=" * 72)

    for M in [2, 4]:
        n_log_M = int(np.log2(M))
        n_total = 1 + 2 * n_log_M

        qc = spatial_qft(M)
        U_qiskit = Operator(qc).data

        U_ref = spatial_qft_matrix(M)
        err = float(np.max(np.abs(U_qiskit - U_ref)))
        print(f"  M={M}: n_qubits={n_total}, dim={U_qiskit.shape}, "
              f"max |U_circuit - U_ref| = {err:.3e}")
        assert err < 1e-10, f"spatial_qft mismatch at M={M}: {err}"

    # Also confirm that QFT applied twice (once forward, once inverse) is the identity.
    print("\n--- QFT * QFT^dagger = I ---")
    for M in [2, 4]:
        qc_f = spatial_qft(M)
        qc_finv = qc_f.inverse()
        full = QuantumCircuit(qc_f.num_qubits)
        full.compose(qc_f, inplace=True)
        full.compose(qc_finv, inplace=True)
        U = Operator(full).data
        err = float(np.max(np.abs(U - np.eye(U.shape[0]))))
        print(f"  M={M}: max |U_F U_F^dag - I| = {err:.3e}")
        assert err < 1e-10

    print("\nAll spatial_qft tests passed.")

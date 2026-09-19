"""
Baseline block encoding of the preconditioner K_0^{-1} via Pauli LCU.

This is the Aim-2 analogue of task3.baseline_shende: classically invert
K_0 (on the zero-mean subspace, using the Moore-Penrose pseudoinverse),
decompose into Paulis, and block-encode via PREP-SELECT-UNPREP.

The structured Aim-2 construction is

    U_{K_0^{-1}} = U_F^dagger * U_{Khat0^{-1}} * U_F

This baseline applies Pauli-LCU directly to the classical K_0^{-1} matrix
without exploiting its block-circulant / Fourier-diagonalizable structure.
For larger M, the baseline grows as a power of N = 2 M^2 (typically the
matrix is dense after inversion, so almost all 4^n Paulis can be present).
"""

from __future__ import annotations

import numpy as np

from qhomogenize.bencode.types import BlockEncoding
from qhomogenize.bencode.pauli_lcu import pauli_block_encoding
from qhomogenize.task4.build_UK0_inv import K0_inv_reference


def build_K0inv_baseline(
    M: int,
    lam: float = 1.0,
    mu: float = 1.0,
    a: float | None = None,
    b: float | None = None,
    phi_deg: float = 90.0,
    name: str | None = None,
) -> BlockEncoding:
    """Build the Pauli-LCU block encoding of the classical K_0^{-1}.

    Note: K_0 is rank-deficient (rigid-translation null space).  The
    pseudoinverse used here is the same one defined by Khat0_inv (zero
    entry at the zero Fourier mode).
    """
    if name is None:
        name = f"U_K0inv_baseline_M{M}"

    K0_inv = K0_inv_reference(M, lam=lam, mu=mu, a=a, b=b, phi_deg=phi_deg).real
    return pauli_block_encoding(K0_inv, name=name)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for M in [2, 4]:
        print(f"\n--- baseline at M = {M} ---")
        be = build_K0inv_baseline(M)
        print(f"  {be.summary()}")
        K_ref = K0_inv_reference(M).real
        # Spot-check at small M:
        if M <= 2:
            passed, err = be.verify_against(K_ref, atol=1e-10, verbose=True)
            assert passed, f"baseline at M={M} failed (err={err})"
        else:
            # Spot-check via Statevector simulation
            from qiskit.quantum_info import Statevector
            from qiskit import QuantumCircuit
            n_sys = be.n_system
            n_anc = be.n_ancilla
            N = 2 ** n_sys
            errors = []
            for j in [0, 1, 5, 17, N - 1]:
                qc_test = QuantumCircuit(n_sys + n_anc)
                for bit in range(n_sys):
                    if (j >> bit) & 1:
                        qc_test.x(bit)
                qc_test.compose(be.circuit, inplace=True)
                sv = Statevector.from_instruction(qc_test).data
                col = sv[:N]
                expected = K_ref[:, j] / be.alpha
                errors.append(float(np.max(np.abs(col - expected))))
            print(f"  spot-check on 5 columns: max err = {max(errors):.3e}")
            assert max(errors) < 1e-10
    print("\nBaseline tests passed.")

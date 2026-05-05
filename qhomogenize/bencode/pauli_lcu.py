"""
Pauli-LCU block encoding for small Hermitian matrices.

For a Hermitian operator A on n qubits, decompose A = sum_k c_k P_k where
{P_k} are n-qubit Pauli operators (I, X, Y, Z tensored) and c_k are real
coefficients.  Then a block encoding of A with subnormalization
alpha = sum_k |c_k| can be built via the standard PREP-SELECT-UNPREP scheme:

    PREP |0>_anc = sum_k sqrt(|c_k|/alpha) |k>_anc
    SELECT = sum_k |k><k|_anc (x) sgn(c_k) P_k
    UNPREP = PREP^dagger

so that  (<0|_anc (x) I_sys) UNPREP * SELECT * PREP (|0>_anc (x) I_sys)
        = sum_k (|c_k|/alpha) sgn(c_k) P_k = A / alpha.

Convention
----------
This module follows the qhomogenize/bencode convention: system qubits at LOW
positions, ancilla qubits at HIGH positions.  This is the OPPOSITE of the
user's existing `Pauli_Block_Encoding` in `Chapter13_MatrixEncoding_functions.py`
(which puts ancilla low and system high).  We re-implement it cleanly here to
keep the convention consistent across the package.

Cost
----
Number of ancilla qubits: ceil(log_2(L)) where L = number of non-zero Pauli
coefficients.  For an n-qubit matrix, L <= 4^n.

The PREP step uses Qiskit's `StatePreparation` which scales as O(L) gates
(Shende-Bullock-Markov synthesis).  The SELECT step uses L multi-controlled
Paulis, each requiring O(log L) ancillas but our use cases are small (n=3
for K_e gives at most 64 Paulis, anc <= 6).

This is the BASELINE we benchmark against in Task 3.  For block-encoding very
large dense matrices, this scales poorly (alpha grows as ||A||_1 and the
number of Paulis is exponential), but for small dense building blocks like
the 8x8 K_e it's appropriate.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp

from qhomogenize.bencode.types import BlockEncoding


def pauli_block_encoding(
    A: np.ndarray,
    name: str = "U_pauli",
    atol: float = 1e-12,
) -> BlockEncoding:
    """Build a block encoding of a Hermitian matrix A via Pauli-LCU.

    Parameters
    ----------
    A : ndarray, shape (2^n, 2^n)
        Hermitian matrix.  Non-Hermitian inputs are accepted; the routine
        decomposes them into Paulis with possibly complex coefficients, and
        the resulting block encoding represents A faithfully but is not
        Hermitian-style.
    name : str
    atol : float
        Coefficients with magnitude below `atol` are dropped from the
        decomposition.

    Returns
    -------
    BlockEncoding
        n_system = log_2(A.shape[0]),
        n_ancilla = ceil(log_2(L_nonzero)),
        alpha = sum |c_k| over the nonzero Paulis.
    """
    A = np.asarray(A, dtype=complex)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    n_sys = int(np.log2(A.shape[0]))
    if 2 ** n_sys != A.shape[0]:
        raise ValueError(f"A.shape[0] must be a power of 2, got {A.shape[0]}")

    # Decompose A into Paulis (real coefficients for Hermitian A).
    sparse_op = SparsePauliOp.from_operator(A).simplify(atol=atol)
    coeffs = np.asarray(sparse_op.coeffs)
    paulis = list(sparse_op.paulis)
    L = len(coeffs)
    if L == 0:
        # A = 0; return a trivial block encoding (the zero operator).
        # Just embed identity with alpha = infinity so the post-select gives 0.
        # Simpler: build a trivial 1-ancilla circuit that always sets ancilla = 1.
        qc_zero = QuantumCircuit(QuantumRegister(n_sys, "sys"), QuantumRegister(1, "anc"), name=name)
        qc_zero.x(n_sys)  # ancilla = 1 always -> post-select fails
        return BlockEncoding(qc_zero, n_sys, 1, alpha=1.0, name=name, target_shape=A.shape)

    alpha = float(np.sum(np.abs(coeffs)))
    n_anc = max(1, int(np.ceil(np.log2(L))))
    M = 2 ** n_anc

    # PREP vector:  amp[k] = sqrt(|c_k|/alpha) for k in [0, L); 0 for padding.
    prep_vec = np.zeros(M, dtype=complex)
    for k in range(L):
        prep_vec[k] = np.sqrt(np.abs(coeffs[k]) / alpha)
    # Normalize to absorb numerical drift
    prep_vec /= np.linalg.norm(prep_vec)

    # Layout (low -> high):  sys (n_sys) | anc (n_anc)
    qr_sys = QuantumRegister(n_sys, "sys")
    qr_anc = QuantumRegister(n_anc, "anc")
    qc = QuantumCircuit(qr_sys, qr_anc, name=name)

    # PREP on ancilla
    prep_gate = StatePreparation(prep_vec, label="PREP")
    qc.append(prep_gate, list(qr_anc))

    # SELECT: for each k, apply sign(c_k) * P_k on system, controlled on anc=k.
    for k in range(L):
        coeff = coeffs[k]
        phase = float(np.angle(coeff))  # sgn(c_k) -> phase 0 or pi for real Hermitian A.
        pauli = paulis[k]
        # Build a small circuit that applies P_k with global phase = phase, then
        # controlled on the ancilla register being |k>.
        sub = QuantumCircuit(n_sys, global_phase=phase)
        sub.append(pauli.to_instruction(), range(n_sys))
        ctrl_state = format(k, f"0{n_anc}b")
        ctrl_gate = sub.to_gate(label=f"P_{k}").control(n_anc, ctrl_state=ctrl_state)
        # ctrl_gate's qubit order: [controls..., targets...]
        # Controls = ancilla qubits, targets = system qubits.
        qc.append(ctrl_gate, list(qr_anc) + list(qr_sys))

    # UNPREP
    qc.append(prep_gate.inverse(), list(qr_anc))

    return BlockEncoding(
        circuit=qc,
        n_system=n_sys,
        n_ancilla=n_anc,
        alpha=alpha,
        name=name,
        target_shape=A.shape,
    )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Test 1: 2x2 Hermitian matrix [[1, 0.5], [0.5, -0.3]] ---")
    A = np.array([[1.0, 0.5], [0.5, -0.3]])
    be = pauli_block_encoding(A)
    print(f"  {be.summary()}")
    passed, err = be.verify_against(A, atol=1e-10, verbose=True)
    assert passed, f"2x2 test failed (err={err})"

    print("\n--- Test 2: 4x4 random Hermitian ---")
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    A = (M + M.conj().T) / 2
    be = pauli_block_encoding(A)
    print(f"  {be.summary()}")
    passed, err = be.verify_against(A, atol=1e-10, verbose=True)
    assert passed, f"4x4 test failed (err={err})"

    print("\n--- Test 3: 8x8 K_e^lambda from element template ---")
    from qhomogenize.fea.element_template import element_template
    Ke_lam, Ke_mu, _, _ = element_template(a=0.5, b=0.5, phi_deg=90.0)
    print(f"  ||Ke_lambda||_max = {np.max(np.abs(Ke_lam)):.4f}")
    print(f"  ||Ke_lambda||_F = {np.linalg.norm(Ke_lam):.4f}")
    be = pauli_block_encoding(Ke_lam, name="U_Ke_lam")
    print(f"  {be.summary()}")
    passed, err = be.verify_against(Ke_lam, atol=1e-10, verbose=True)
    assert passed, f"K_e^lambda test failed (err={err})"

    print("\n--- Test 4: 8x8 K_e^mu ---")
    be = pauli_block_encoding(Ke_mu, name="U_Ke_mu")
    print(f"  {be.summary()}")
    passed, err = be.verify_against(Ke_mu, atol=1e-10, verbose=True)
    assert passed, f"K_e^mu test failed (err={err})"

    print("\n--- Test 5: 8x8 K_e (combined: lambda=mu=1) ---")
    Ke = Ke_lam + Ke_mu
    be = pauli_block_encoding(Ke, name="U_Ke")
    print(f"  {be.summary()}")
    passed, err = be.verify_against(Ke, atol=1e-10, verbose=True)
    assert passed, f"K_e combined test failed (err={err})"

    print("\nAll pauli_lcu tests passed.")

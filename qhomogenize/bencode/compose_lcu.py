"""
LCU sum of block encodings.

Given block encodings U_1, ..., U_K of matrices A_1, ..., A_K with
subnormalizations alpha_1, ..., alpha_K, and signed weights w_1, ..., w_K, the
LCU-of-block-encodings composition yields a block encoding U of

    A = sum_k w_k A_k

with subnormalization

    alpha = sum_k |w_k| alpha_k

and ancilla count = max_k a_k + ceil(log2 K) (the K-side PREP register).

Implementation: standard PREP-SELECT-UNPREP.  Define coefficients
    beta_k = |w_k| * alpha_k,    alpha = sum_k beta_k.
PREP loads the state  sum_k sqrt(beta_k / alpha) |k>  on a ceil(log2 K)-qubit
"sel" register.  SELECT applies, for each k, the sign-corrected component
circuit U_k (times sign(w_k)) on (sys, anc) controlled on |k>.  UNPREP is
PREP^dagger.  The top-left block (all ancilla = |0>) is then

    (1/alpha) sum_k beta_k * sign(w_k) * (A_k / alpha_k)
        = (1/alpha) sum_k w_k A_k,

as required.

Each component's own ancilla register (width a_k) is mapped onto the FIRST
a_k qubits of a shared anc register of width max_k a_k.  Because the
post-selection requires ALL anc qubits to return to |0>, and each U_k starts
and ends its own ancillas at |0>, the shared register is consistent: unused
high ancilla qubits for a given U_k are simply idle (identity) and remain |0>.

The optional sign of w_k is applied as a global phase on the controlled
block via a Z-type phase on the sel register state |k> (implemented by
appending a single-qubit phase through a diagonal on the sel register during
PREP).  For real symmetric FEM operators all weights are positive in our use
(Lame split with lam, mu, and the masked corrections handled by separate
non-negative-weight terms), but we support signs for generality.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation

from qhomogenize.bencode.types import BlockEncoding


def compose_lcu(
    components: Sequence[BlockEncoding],
    weights: Sequence[float],
    name: str = "U_LCU",
) -> BlockEncoding:
    """Compose a weighted LCU of K block encodings.

    Parameters
    ----------
    components : sequence of BlockEncoding
        The K block encodings to sum.  All must share the same n_system.
    weights : sequence of float
        K signed weights.
    name : str

    Returns
    -------
    BlockEncoding of sum_k weights[k] * components[k].A with
    alpha = sum_k |weights[k]| * components[k].alpha and
    n_ancilla = max_k components[k].n_ancilla + ceil(log2 K).
    """
    K = len(components)
    if K != len(weights):
        raise ValueError("components and weights must have equal length")
    if K == 0:
        raise ValueError("need at least one component")
    n_sys = components[0].n_system
    for be in components:
        if be.n_system != n_sys:
            raise ValueError("all components must share n_system")

    # K = 1 is just a (possibly signed/scaled) single encoding.
    weights = [float(w) for w in weights]
    alphas = [be.alpha for be in components]
    betas = [abs(w) * a for w, a in zip(weights, alphas)]
    alpha_total = float(sum(betas))
    if alpha_total <= 0:
        raise ValueError("all-zero LCU is not a valid block encoding")

    n_sel = max(1, int(np.ceil(np.log2(K)))) if K > 1 else 0
    n_anc_max = max(be.n_ancilla for be in components)

    # PREP amplitudes over 2^n_sel states (pad with zeros beyond K).
    dim_sel = 2 ** n_sel if n_sel > 0 else 1
    amps = np.zeros(dim_sel, dtype=float)
    for k in range(K):
        amps[k] = np.sqrt(betas[k] / alpha_total)
    # Normalize defensively (should already be unit norm).
    nrm = np.linalg.norm(amps)
    amps = amps / nrm

    # Registers (low -> high): sys | anc(shared) | sel
    qr_sys = QuantumRegister(n_sys, "sys")
    qr_anc = QuantumRegister(n_anc_max, "anc") if n_anc_max > 0 else None
    qr_sel = QuantumRegister(n_sel, "sel") if n_sel > 0 else None

    regs = [qr_sys]
    if qr_anc is not None:
        regs.append(qr_anc)
    if qr_sel is not None:
        regs.append(qr_sel)
    qc = QuantumCircuit(*regs, name=name)

    # ---- PREP ----
    if n_sel > 0:
        prep = StatePreparation(amps)
        qc.append(prep, list(qr_sel))

    # ---- SELECT ----
    # For each k, apply sign(w_k) * U_k on (sys, anc[:a_k]) controlled on sel == k.
    for k, be in enumerate(components):
        a_k = be.n_ancilla
        sub_qubits = list(qr_sys) + (list(qr_anc[:a_k]) if a_k > 0 else [])
        comp_circ = be.circuit
        # Sign handling: apply a global phase of pi on the |k> branch if w_k < 0.
        if n_sel > 0:
            # Build controlled version of the component circuit on sel==k.
            # Decompose first to lower any StatePreparation (which carries a
            # reset and blocks to_gate) into plain unitary gates.
            comp_dec = comp_circ.decompose(reps=8)
            gate = comp_dec.to_gate(label=f"U_{k}")
            # ctrl_state as integer k: qiskit maps this little-endian over the
            # control qubits (q0=bit0, ...), matching our sel register.
            cgate = gate.control(n_sel, ctrl_state=k)
            qc.append(cgate, list(qr_sel) + sub_qubits)
            if weights[k] < 0:
                # phase flip on the sel==k branch
                _mc_phase(qc, qr_sel, k, n_sel, np.pi)
        else:
            comp_dec = comp_circ.decompose(reps=8)
            qc.append(comp_dec.to_gate(label=f"U_{k}"), sub_qubits)
            if weights[k] < 0:
                qc.global_phase += np.pi

    # ---- UNPREP ----
    if n_sel > 0:
        qc.append(StatePreparation(amps).inverse(), list(qr_sel))

    target_shape = components[0].target_shape
    return BlockEncoding(
        circuit=qc,
        n_system=n_sys,
        n_ancilla=n_anc_max + n_sel,
        alpha=alpha_total,
        name=name,
        target_shape=target_shape,
    )


def _mc_phase(qc: QuantumCircuit, qr_sel: QuantumRegister, k: int, n_sel: int, theta: float) -> None:
    """Apply phase exp(i*theta) on the |k> branch of the sel register."""
    # Flip zeros of k so that |k> -> |11...1>, apply multi-controlled phase, flip back.
    bits = [(k >> j) & 1 for j in range(n_sel)]
    for j in range(n_sel):
        if bits[j] == 0:
            qc.x(qr_sel[j])
    if n_sel == 1:
        qc.p(theta, qr_sel[0])
    else:
        from qiskit.circuit.library import MCPhaseGate
        qc.append(MCPhaseGate(theta, n_sel - 1), list(qr_sel))
    for j in range(n_sel):
        if bits[j] == 0:
            qc.x(qr_sel[j])


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from qhomogenize.bencode.pauli_lcu import pauli_block_encoding

    print("--- Test 1: LCU of two 2x2 matrices, equal weights ---")
    A1 = np.array([[1.0, 0.0], [0.0, -1.0]])  # Z
    A2 = np.array([[0.0, 1.0], [1.0, 0.0]])   # X
    be1 = pauli_block_encoding(A1, name="U_Z")
    be2 = pauli_block_encoding(A2, name="U_X")
    be = compose_lcu([be1, be2], [1.0, 1.0], name="U_ZX")
    print(f"  {be.summary()}")
    passed, err = be.verify_against(A1 + A2, atol=1e-9, verbose=True)
    assert passed, f"LCU Z+X failed: {err}"

    print("\n--- Test 2: signed weights, 0.5*Z - 2*X ---")
    be = compose_lcu([be1, be2], [0.5, -2.0], name="U_signed")
    print(f"  {be.summary()}")
    passed, err = be.verify_against(0.5 * A1 - 2.0 * A2, atol=1e-9, verbose=True)
    assert passed, f"signed LCU failed: {err}"

    print("\n--- Test 3: three terms, 4x4 ---")
    rng = np.random.default_rng(0)
    M = [rng.standard_normal((4, 4)) for _ in range(3)]
    M = [0.5 * (m + m.T) for m in M]  # symmetric
    bes = [pauli_block_encoding(m, name=f"U_{i}") for i, m in enumerate(M)]
    w = [1.3, -0.7, 0.4]
    be = compose_lcu(bes, w, name="U_three")
    print(f"  {be.summary()}")
    ref = sum(wi * mi for wi, mi in zip(w, M))
    passed, err = be.verify_against(ref, atol=1e-9, verbose=True)
    assert passed, f"three-term LCU failed: {err}"

    print("\nAll compose_lcu tests passed.")

"""
Diffusion gate over a sparsity register.

Sünderhauf's base scheme uses uniform superposition over the size-S sparsity
register (where S = max(S_c, S_r) is the maximum row/column sparsity, padded
to a power of 2 if necessary):

    |0>_S -> (1 / sqrt(S)) sum_{s=0}^{S-1} |s>_S.

Cost: log2(S) Hadamard gates if S is a power of 2; otherwise an
amplitude-amplification or controlled-rotation construction is used.

For our application: S_c = 4 (in 2D), so the diffusion register has 2 qubits
and the diffusion is just H ⊗ H.  Trivial.

TODO: Generalize to arbitrary S using PyEncode's STEP pattern (which already
supports general-S uniform superposition in O(log S) gates).
"""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister


def diffusion(qreg: QuantumRegister) -> QuantumCircuit:
    """Build a uniform-superposition diffusion gate on `qreg`.

    For now: assumes 2^qreg.size is the desired uniform-distribution size.
    """
    qc = QuantumCircuit(qreg, name="diff")
    for q in qreg:
        qc.h(q)
    return qc

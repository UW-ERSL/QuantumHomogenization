"""
Fast block extraction for verification.

The default BlockEncoding.extract_matrix() builds the full dense unitary via
Operator(circuit), which costs O(4^n_qubits) time and memory and OOMs beyond
~13 qubits.  For verification we only need the top-left (ancilla=|0>) system
block A/alpha of size 2^n_system x 2^n_system.

This module extracts that block column-by-column with statevector evolution:
for each system basis state |j> (ancillas in |0>), simulate the circuit once
and read the amplitudes on the ancilla=|0> system subspace.  Cost is
2^n_system statevector simulations of length 2^n_qubits, i.e.
O(2^n_system * 2^n_qubits) -- dramatically cheaper than O(4^n_qubits) when
there are many ancillas (the usual case after composition).

Convention matches BlockEncoding.extract_matrix(): system = least-significant
qubits, ancilla = |0> selects the top-left block.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qhomogenize.bencode.types import BlockEncoding


def extract_block(be: BlockEncoding, columns: int | None = None) -> np.ndarray:
    """Extract A/alpha (2^n_sys x 2^n_sys) via batched statevector evolution.

    For each system basis state |j> (ancillas in |0>), the circuit is simulated
    and the amplitudes on the ancilla=|0> system subspace give column j.  All
    columns are batched into a single Aer run to amortize setup overhead; if
    Aer is unavailable the method falls back to per-column Statevector.

    Parameters
    ----------
    be : BlockEncoding
    columns : int, optional
        Extract only the first `columns` columns (rest zero).

    Returns
    -------
    A_normalized : ndarray, shape (2^n_sys, 2^n_sys), equal to A/alpha.
    """
    n_sys = be.n_system
    n_q = be.circuit.num_qubits
    d = 2 ** n_sys
    ncol = d if columns is None else min(columns, d)
    out = np.zeros((d, d), dtype=complex)

    try:
        from qiskit_aer import AerSimulator

        sim = AerSimulator(method="statevector")
        base = be.circuit
        circs = []
        for j in range(ncol):
            qc = QuantumCircuit(n_q)
            # Prepare |j>_sys (x) |0>_anc : set system bits of j (qubits 0..n_sys-1).
            for b in range(n_sys):
                if (j >> b) & 1:
                    qc.x(b)
            qc.compose(base, inplace=True)
            qc.save_statevector()
            circs.append(qc)

        # Transpile once (structure identical across columns up to the prep X's).
        result = sim.run(circs).result()
        for j in range(ncol):
            amps = np.asarray(result.get_statevector(j).data)
            out[:, j] = amps[:d]
        return out
    except Exception:
        # Fallback: exact per-column Statevector (no Aer).
        for j in range(ncol):
            sv = Statevector.from_int(j, dims=2 ** n_q).evolve(be.circuit)
            out[:, j] = np.asarray(sv.data)[:d]
        return out


def verify_fast(
    be: BlockEncoding,
    A_reference: np.ndarray,
    atol: float = 1e-10,
    verbose: bool = False,
    columns: int | None = None,
) -> tuple[bool, float]:
    """Like BlockEncoding.verify_against but using the fast column extractor."""
    A_blk = extract_block(be, columns=columns)
    m, n = be.target_shape if be.target_shape is not None else (2 ** be.n_system,) * 2
    A_extracted = A_blk[:m, :n]
    expected = A_reference / be.alpha
    err = float(np.max(np.abs(A_extracted - expected)))
    passed = bool(err < atol)
    if verbose:
        print(f"  {be.name}: max abs error = {err:.3e}, passed = {passed}  (fast)")
    return passed, err


# ---------------------------------------------------------------------------
# Self-test: agree with the slow extractor on small cases, and time both.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from qhomogenize.task1.build_UA import build_UA
    from qhomogenize.task2.build_UDk import build_UDk

    for L in [2, 4]:
        print(f"--- L={L} ---")
        for label, be in [("U_A", build_UA(L)),
                          ("U_DK", build_UDk(L, a=1/(2*L), b=1/(2*L)))]:
            nq = be.circuit.num_qubits
            t0 = time.time()
            fast = extract_block(be)
            t_fast = time.time() - t0

            t0 = time.time()
            slow = be.extract_matrix()
            t_slow = time.time() - t0

            err = np.max(np.abs(fast - slow))
            print(f"  {label}: n_qubits={nq}  fast={t_fast:.3f}s  slow={t_slow:.3f}s  "
                  f"speedup={t_slow/max(t_fast,1e-9):.1f}x  max|fast-slow|={err:.2e}")

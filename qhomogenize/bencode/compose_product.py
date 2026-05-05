"""
Product of block encodings:  given U_A encoding A and U_B encoding B with the
same system register width, build U_{AB} encoding A B.

Construction (Gilyén-Su-Low-Wiebe, Quantum 2019, Lemma 53):

    U_{AB} = (U_A on (sys, anc_A))  followed by  (U_B on (sys, anc_B))

The two ancilla registers are disjoint; only the system register is shared.
Subnormalization composes multiplicatively:  alpha_{AB} = alpha_A * alpha_B.

Note on operation order
-----------------------
For matrix multiplication AB applied to a vector |x>, B acts first (B|x>) and
A acts second (A B|x>).  So in the quantum circuit we apply U_B first (left in
time) and U_A second (right in time).  In Qiskit, this means appending U_B
before U_A.

References
----------
Gilyén, Su, Low, Wiebe, "Quantum singular value transformation and beyond:
exponential improvements for quantum matrix arithmetics," Proc. STOC 2019,
arXiv:1806.01838 (Lemma 53 / Lemma 30 in the journal version).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from qhomogenize.bencode.types import BlockEncoding


def compose_product(
    be_A: BlockEncoding,
    be_B: BlockEncoding,
    name: str = "U_AB",
) -> BlockEncoding:
    """Build a block encoding of A * B from block encodings of A and B.

    Both block encodings must share the same system register width.  The ancilla
    registers are stacked: ancilla_combined = ancilla_B (low) ++ ancilla_A (high).

    Parameters
    ----------
    be_A : BlockEncoding
    be_B : BlockEncoding
    name : str

    Returns
    -------
    BlockEncoding
        Encodes A * B / (alpha_A * alpha_B).  Subnormalization = alpha_A * alpha_B.
    """
    if be_A.n_system != be_B.n_system:
        raise ValueError(
            f"system registers must match: A has n_system={be_A.n_system}, "
            f"B has n_system={be_B.n_system}"
        )
    n_sys = be_A.n_system
    n_anc_A = be_A.n_ancilla
    n_anc_B = be_B.n_ancilla

    # Layout (low -> high):  sys (n_sys) | anc_B (n_anc_B) | anc_A (n_anc_A)
    qr_sys = QuantumRegister(n_sys, "sys")
    qr_anc_B = QuantumRegister(n_anc_B, "anc_B") if n_anc_B > 0 else None
    qr_anc_A = QuantumRegister(n_anc_A, "anc_A") if n_anc_A > 0 else None

    regs = [qr_sys]
    if qr_anc_B is not None:
        regs.append(qr_anc_B)
    if qr_anc_A is not None:
        regs.append(qr_anc_A)
    qc = QuantumCircuit(*regs, name=name)

    # Apply U_B first (right factor in matrix product), on (sys, anc_B).
    qubits_B = list(qr_sys) + (list(qr_anc_B) if qr_anc_B is not None else [])
    qc.append(be_B.circuit, qubits_B)

    # Apply U_A second (left factor), on (sys, anc_A).
    qubits_A = list(qr_sys) + (list(qr_anc_A) if qr_anc_A is not None else [])
    qc.append(be_A.circuit, qubits_A)

    # Determine target shape if specified.
    if be_A.target_shape is not None and be_B.target_shape is not None:
        target_shape = (be_A.target_shape[0], be_B.target_shape[1])
    else:
        target_shape = None

    return BlockEncoding(
        circuit=qc,
        n_system=n_sys,
        n_ancilla=n_anc_A + n_anc_B,
        alpha=be_A.alpha * be_B.alpha,
        name=name,
        target_shape=target_shape,
    )


def compose_triple(
    be_A: BlockEncoding,
    be_B: BlockEncoding,
    be_C: BlockEncoding,
    name: str = "U_ABC",
) -> BlockEncoding:
    """Build a block encoding of A * B * C.  Convenience wrapper around compose_product."""
    return compose_product(be_A, compose_product(be_B, be_C), name=name)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from qhomogenize.bencode.builder import BlockEncodingSpec, build_block_encoding, make_data_oracle

    print("--- Test 1: compose two anti-diagonals X * X = I ---")
    O_c = QuantumCircuit(1, name="O_c"); O_r = QuantumCircuit(1, name="O_r"); O_r.x(0)
    O_data, _ = make_data_oracle([1.0], A_max=1.0)
    be_X = build_block_encoding(BlockEncodingSpec(
        n_idx=1, n_s=0, n_d=0, A_max=1.0, O_c=O_c, O_r=O_r, O_data=O_data, name="U_X"
    ))
    be_XX = compose_product(be_X, be_X, name="U_XX")
    print(f"  {be_XX.summary()}")
    A_ref = np.eye(2)  # X*X = I
    passed, err = be_XX.verify_against(A_ref, atol=1e-10, verbose=True)
    assert passed, f"X*X test failed (err={err})"

    print("\n--- Test 2: compose diag(2,1,2,1) with itself = diag(4,1,4,1) ---")
    O_c = QuantumCircuit(2, name="O_c"); O_r = QuantumCircuit(2, name="O_r")
    O_data, _ = make_data_oracle([2.0, 1.0], A_max=2.0)
    be_D = build_block_encoding(BlockEncodingSpec(
        n_idx=2, n_s=0, n_d=1, A_max=2.0, O_c=O_c, O_r=O_r, O_data=O_data, name="U_D"
    ))
    print(f"  U_D: {be_D.summary()}")
    be_DD = compose_product(be_D, be_D, name="U_DD")
    print(f"  {be_DD.summary()}")
    A_ref = np.diag([4.0, 1.0, 4.0, 1.0])  # D^2
    passed, err = be_DD.verify_against(A_ref, atol=1e-10, verbose=True)
    assert passed, f"D*D test failed (err={err})"

    print("\n--- Test 3: triple X * X * X = X ---")
    be_XXX = compose_triple(be_X, be_X, be_X, name="U_XXX")
    print(f"  {be_XXX.summary()}")
    A_ref = np.array([[0.0, 1.0], [1.0, 0.0]])  # X
    passed, err = be_XXX.verify_against(A_ref, atol=1e-10, verbose=True)
    assert passed, f"X*X*X test failed (err={err})"

    print("\nAll compose_product tests passed.")

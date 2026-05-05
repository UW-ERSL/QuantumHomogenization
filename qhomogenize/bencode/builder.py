"""
Sünderhauf base-scheme block-encoding builder.

Implements the base-scheme circuit of Sünderhauf, Campbell & Camps (Quantum 8,
1226 (2024)), eq. (11):

    flag (data + s) starts at |0>;
    apply H_S to s register;
    apply O_c^dagger on (s, j) register, reinterpreted as (d, m);
    apply O_data: multiplexed Y-rotation on data, controlled on d, encoding
        the amplitude A_d / ||A||_max as the |0>-component;
    apply O_r on (d, m) register, reinterpreted as (s, i);
    apply H_S^dagger on s register.

Subnormalization:  alpha = sqrt(S_c S_r) * ||A||_max = S * ||A||_max in the
symmetric case (S_c = S_r = S).

Flag qubits:  1 (data) + log_2(S) (s register)  =  1 + n_s qubits.

Bit-width consistency
---------------------
The framework requires NS = MD where N is the matrix dimension, S is the
sparsity register size, M is the multiplicity register size, and D is the
distinct-value register size.  In bit-width form:

    n_idx + n_s = n_d + n_m

The same physical qubits hold either (s, j) or (d, m); the oracles O_c, O_r
are bijective relabelings between these two views.

Convention
----------
Total system register layout (low bits to high), in the input view |0>|s>|j>:

    [ j register : n_idx ][ s register : n_s ][ data : 1 ]
       ^low qubits                                 ^high qubit

After O_c^dagger the same n_s + n_idx qubits are interpreted as (d, m) with
d in the low n_d bits and m in the high n_m bits.  O_data acts on the data
qubit controlled on the d sub-register (the low n_d bits).

In `BlockEncoding`'s convention (system = low qubits, ancilla = high), the
system is the n_idx-qubit |j>/|i> register; the ancilla is the (n_s + 1)
register comprising s and data.  Post-selecting ancilla = |0> recovers
A_{ij} / alpha as the (i, j) entry of the post-selected unitary.

For the full scheme (with out-of-range flagging via a `del` qubit, per
Sünderhauf eq. (16) and beyond), pass `O_rg` to `build_block_encoding`.

References
----------
Sünderhauf, Campbell, Camps, "Block-encoding structured matrices for data
input in quantum computing," Quantum 8, 1226 (2024), section 2.1.

The data-loading rotation is implemented via UCRYGate (uniformly-controlled
Y rotation), the standard multiplexed-rotation primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UCRYGate

from qhomogenize.bencode.types import BlockEncoding


# ---------------------------------------------------------------------------
# Helper: build the data oracle as a multiplexed Y rotation
# ---------------------------------------------------------------------------

def make_data_oracle(values: Sequence[float], A_max: Optional[float] = None) -> tuple[QuantumCircuit, float]:
    """Build O_data as a multiplexed Y rotation encoding values[d] / A_max.

    The rotation is R_y(theta_d) with theta_d = 2 arccos(values[d] / A_max),
    applied to the data qubit conditional on the d register being in state |d>.

    The convention (Sünderhauf eq. (7)) is:
        R_y(theta_d) |0> = (values[d] / A_max) |0> + sqrt(1 - (values[d]/A_max)^2) |1>,
    so that post-selection on data = |0> recovers values[d] / A_max.

    Parameters
    ----------
    values : sequence of float
        Distinct values v_d for d in [0, D).  Length D = 2^n_d should be a
        power of 2 (pad with zeros if necessary).  Values may be signed --
        sign is encoded in the rotation angle but loses the sign on
        post-selection; for signed matrices a separate sign oracle is needed
        (Sünderhauf §2.4).  For Lamé element templates we use absolute values
        and track signs at the data level via a separate Z gate.
    A_max : float, optional
        Normalization.  Defaults to max(|values|).  Must satisfy A_max >= max(|values|).

    Returns
    -------
    qc : QuantumCircuit
        Circuit on (n_d + 1) qubits, with the data qubit as the LAST qubit
        (highest index).  When appended to a larger circuit, the d sub-register
        comes first.
    A_max : float
        The normalization used.
    """
    D = len(values)
    if D < 1:
        raise ValueError("values must be non-empty")
    n_d = int(np.ceil(np.log2(D))) if D > 1 else 0
    D_padded = 2 ** n_d if n_d > 0 else 1

    if A_max is None:
        A_max = max((abs(v) for v in values), default=1.0)
    if A_max <= 0:
        raise ValueError("A_max must be positive")

    angles = []
    for d in range(max(D_padded, 1)):
        v = values[d] if d < D else 0.0
        ratio = abs(v) / A_max
        if ratio > 1.0:
            if ratio > 1.0 + 1e-12:
                raise ValueError(f"|v_{d}|/A_max = {ratio} > 1; A_max too small")
            ratio = 1.0
        angle = 2 * np.arccos(ratio)
        angles.append(float(angle))

    qc = QuantumCircuit(n_d + 1, name="O_data")
    if n_d == 0:
        # D = 1: unconditional rotation on the data qubit
        qc.ry(angles[0], 0)
    else:
        # UCRYGate convention: qubit list is [target, control_0, control_1, ...].
        # The angle index is the integer formed by the controls (control_0 = LSB).
        ucry = UCRYGate(angles)
        qc.append(ucry, [n_d, *range(n_d)])
    return qc, A_max


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

@dataclass
class BlockEncodingSpec:
    """Inputs for `build_block_encoding`.

    Attributes
    ----------
    n_idx : int
        Number of qubits for the matrix-index register (|j> on input, |i> on output).
        Matrix dimension N = 2^n_idx.
    n_s : int
        Number of qubits for the sparsity register.  S = 2^n_s = max(S_c, S_r).
    n_d : int
        Number of qubits for the distinct-value register.  D = 2^n_d.
        Must satisfy n_d <= n_s + n_idx; the multiplicity register has
        n_m = n_s + n_idx - n_d qubits.
    A_max : float
        ||A||_max (largest absolute value of any matrix entry).
    O_c : QuantumCircuit
        Column oracle on (n_s + n_idx) qubits.  Implements the bijection
        |s_c, j> -> |d, m> with the convention that the input low n_s bits
        are s_c and the high n_idx bits are j; on output the low n_d bits
        are d and the high n_m bits are m.
    O_r : QuantumCircuit
        Row oracle on (n_s + n_idx) qubits = (n_d + n_m) qubits.  Implements
        |d, m> -> |s_r, i> with output low n_s bits = s_r and high n_idx = i.
    O_data : QuantumCircuit
        Data oracle on (n_d + 1) qubits, with the data qubit as the last.
        Constructed via `make_data_oracle`.
    O_rg : QuantumCircuit, optional
        Out-of-range oracle on (n_d + n_m + 1) qubits, with the del qubit as
        the last.  If None, the base scheme without del is used (assumes
        no padding is needed: NS = MD).
    name : str
    """
    n_idx: int
    n_s: int
    n_d: int
    A_max: float
    O_c: QuantumCircuit
    O_r: QuantumCircuit
    O_data: QuantumCircuit
    O_rg: Optional[QuantumCircuit] = None
    name: str = "U_A"

    def __post_init__(self):
        if self.n_d > self.n_s + self.n_idx:
            raise ValueError(
                f"n_d ({self.n_d}) must be <= n_s + n_idx "
                f"({self.n_s} + {self.n_idx} = {self.n_s + self.n_idx})"
            )
        if self.O_c.num_qubits != self.n_s + self.n_idx:
            raise ValueError(
                f"O_c expects {self.n_s + self.n_idx} qubits, got {self.O_c.num_qubits}"
            )
        if self.O_r.num_qubits != self.n_s + self.n_idx:
            raise ValueError(
                f"O_r expects {self.n_s + self.n_idx} qubits, got {self.O_r.num_qubits}"
            )
        if self.O_data.num_qubits != self.n_d + 1:
            raise ValueError(
                f"O_data expects {self.n_d + 1} qubits, got {self.O_data.num_qubits}"
            )

    @property
    def n_m(self) -> int:
        return self.n_s + self.n_idx - self.n_d


def build_block_encoding(spec: BlockEncodingSpec) -> BlockEncoding:
    """Assemble the Sünderhauf base-scheme block encoding from oracles.

    Returns a `BlockEncoding` with system register of width n_idx and ancilla
    register of width (n_s + 1) [+ 1 if O_rg given].  The subnormalization is

        alpha = 2^n_s * A_max  =  S * A_max (the symmetric case S_c = S_r = S)

    Parameters
    ----------
    spec : BlockEncodingSpec

    Returns
    -------
    BlockEncoding
    """
    n_idx = spec.n_idx
    n_s = spec.n_s
    n_d = spec.n_d
    use_del = spec.O_rg is not None
    n_anc = n_s + 1 + (1 if use_del else 0)

    # Registers (low to high)
    qr_sys = QuantumRegister(n_idx, "sys")     # |j> -> |i> register
    qr_s = QuantumRegister(n_s, "s") if n_s > 0 else None
    qr_data = QuantumRegister(1, "data")
    qr_del = QuantumRegister(1, "del") if use_del else None

    regs = [qr_sys]
    if qr_s is not None:
        regs.append(qr_s)
    regs.append(qr_data)
    if qr_del is not None:
        regs.append(qr_del)

    qc = QuantumCircuit(*regs, name=spec.name)

    # Convenience: full (s, j) qubit list, low to high.
    sj_qubits = list(qr_sys) + (list(qr_s) if qr_s is not None else [])
    # The d sub-register is the LOW n_d qubits of (s, j) after O_c^dagger.
    d_qubits = sj_qubits[:n_d]

    # 1. Diffusion: H on s register
    if qr_s is not None:
        for q in qr_s:
            qc.h(q)

    # 2. O_c^dagger on (s, j)  [maps (s_c, j) -> (d, m)]
    qc.append(spec.O_c.inverse(), sj_qubits)

    # 3. (Full scheme) Apply O_rg to flag out-of-range (d, m) into the del qubit.
    if use_del:
        qc.append(spec.O_rg, sj_qubits + [qr_del[0]])

    # 4. Data oracle: rotates data qubit controlled on d.
    qc.append(spec.O_data, d_qubits + [qr_data[0]])

    # 4a. (Full scheme) For out-of-range entries we need data = |1> at this
    # point so that post-selection on data = 0 fails.  For 0/1 matrices
    # (A_d == A_max for all in-range d), R = identity leaves data = |0>, and
    # a CX from del to data flips data to |1> for invalid entries.
    #
    # NOTE: This trick only works correctly when A_d == A_max for all in-range
    # d (as is the case for boolean assembly operators A and indicator masks
    # M_p).  For matrices with non-uniform values, the proper approach is to
    # include d_invalid as a data value with A_{d_invalid} = 0 (then the
    # multiplexed rotation handles invalid entries automatically), or to
    # anti-control the rotation on del.  See Sünderhauf eq. (61) and (63)
    # for the labelled-d-value approach.
    if use_del:
        qc.cx(qr_del[0], qr_data[0])

    # 5. (Full scheme) Uncompute O_rg
    if use_del:
        qc.append(spec.O_rg.inverse(), sj_qubits + [qr_del[0]])

    # 6. O_r on (d, m)  [maps (d, m) -> (s_r, i)]
    qc.append(spec.O_r, sj_qubits)

    # 7. H on s register (uncompute the diffusion)
    if qr_s is not None:
        for q in qr_s:
            qc.h(q)

    alpha = (2 ** n_s) * spec.A_max
    N = 2 ** n_idx
    be = BlockEncoding(
        circuit=qc,
        n_system=n_idx,
        n_ancilla=n_anc,
        alpha=alpha,
        name=spec.name,
        target_shape=(N, N),
    )
    return be


# ---------------------------------------------------------------------------
# Quick test: 4x4 diagonal matrix diag(2, 1, 2, 1)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # diag(2, 1, 2, 1) has D=2 distinct values, M=2 multiplicities, N=4, S=1.
    # n_idx = 2, n_s = 0, n_d = 1, n_m = 1.  NS = 4 = MD ✓.

    # Labelling: (d, m) -> (i, j) with i = j = d + 2*m.  So:
    #   (d=0, m=0) -> (i=0, j=0):  A_0 = 2
    #   (d=1, m=0) -> (i=1, j=1):  A_1 = 1
    #   (d=0, m=1) -> (i=2, j=2):  A_0 = 2
    #   (d=1, m=1) -> (i=3, j=3):  A_1 = 1
    # In our bit convention low=d, high=m for (d,m) and low=j_low, high=j_high for j.
    # Given j = j_low + 2*j_high and we need (d, m) = (j_low, j_high), O_c is identity.
    # Given (d, m) and i = d + 2*m, we need O_r to set i = (d, m) interpreted with low=d, high=m.
    # That's also identity in our bit convention.

    n_idx, n_s, n_d = 2, 0, 1
    A_max = 2.0
    O_c = QuantumCircuit(n_s + n_idx, name="O_c")  # identity
    O_r = QuantumCircuit(n_s + n_idx, name="O_r")  # identity
    O_data, A_max_used = make_data_oracle([2.0, 1.0], A_max=A_max)

    spec = BlockEncodingSpec(
        n_idx=n_idx, n_s=n_s, n_d=n_d, A_max=A_max,
        O_c=O_c, O_r=O_r, O_data=O_data, name="U_diag2121",
    )
    be = build_block_encoding(spec)
    print(be.summary())

    # Verify against numpy
    A_ref = np.diag([2.0, 1.0, 2.0, 1.0])
    passed, err = be.verify_against(A_ref, atol=1e-10, verbose=True)
    assert passed, f"diag test failed (err={err})"

    # ---------- Second test: 2x2 anti-diagonal [[0, 1], [1, 0]] ----------
    # D = 1 (only value 1), N = 2, M = 2, S = 1.
    # (d=0, m=0) -> (i=1, j=0)  => off-diagonal at (1,0)
    # (d=0, m=1) -> (i=0, j=1)  => off-diagonal at (0,1)
    # n_idx = 1, n_s = 0, n_d = 0, n_m = 1.
    # O_c: |s_c=*, j> -> |d=*, m> ;  here n_s=n_d=0, so the register is just j (1 bit) = m (1 bit).
    #   We need (d, m) = (-, j) = m, so O_c is identity.
    # O_r: |d, m> -> |s_r, i> = i (1 bit).  We need i = NOT m (since (m=0)->i=1, (m=1)->i=0): X gate.
    # O_data: rotate by R_y(0) since A_d/A_max = 1/1 = 1.

    print("\n--- 2x2 anti-diagonal ---")
    n_idx, n_s, n_d = 1, 0, 0
    A_max = 1.0
    O_c = QuantumCircuit(1, name="O_c")  # identity
    O_r = QuantumCircuit(1, name="O_r")
    O_r.x(0)                              # X on the m bit
    O_data, _ = make_data_oracle([1.0], A_max=A_max)

    spec = BlockEncodingSpec(
        n_idx=n_idx, n_s=n_s, n_d=n_d, A_max=A_max,
        O_c=O_c, O_r=O_r, O_data=O_data, name="U_antidiag",
    )
    be = build_block_encoding(spec)
    print(be.summary())
    A_ref = np.array([[0.0, 1.0], [1.0, 0.0]])
    passed, err = be.verify_against(A_ref, atol=1e-10, verbose=True)
    assert passed, f"anti-diagonal test failed (err={err})"

    # ---------- Third test: 2x2 diag(1, 0.6) -- non-trivial rotation angle ----------
    print("\n--- 2x2 diag(1, 0.6) ---")
    A_max = 1.0
    O_c = QuantumCircuit(1, name="O_c")
    O_r = QuantumCircuit(1, name="O_r")
    O_data, _ = make_data_oracle([1.0, 0.6], A_max=A_max)
    spec = BlockEncodingSpec(
        n_idx=1, n_s=0, n_d=1, A_max=A_max,
        O_c=O_c, O_r=O_r, O_data=O_data, name="U_diag10p6",
    )
    be = build_block_encoding(spec)
    print(be.summary())
    A_ref = np.diag([1.0, 0.6])
    passed, err = be.verify_against(A_ref, atol=1e-10, verbose=True)
    assert passed, f"diag(1, 0.6) test failed (err={err})"

    # ---------- Fourth test: 2x2 all-ones with non-trivial diffusion register ----------
    # All-ones matrix [[1,1],[1,1]].  S_c = S_r = 2, D = 1, M = 2, N = 2.
    # n_idx=1, n_s=1, n_d=0, n_m=2.  NS = 4 = MD ✓.  alpha = 2 * 1 = 2.
    # Labelling so that A_d at multiplicity m sits at row i, col j: m = s_c + 2*j,
    # i = m // 2 = s_c, s_r = m mod 2 = j.
    # In the (low=s_c, high=j) bit convention, that means O_c = identity and
    # O_r = SWAP (relabels bits as (low=j=s_r, high=s_c=i)).
    print("\n--- 2x2 all-ones (diffusion register exercised) ---")
    n_idx, n_s, n_d = 1, 1, 0
    A_max = 1.0
    O_c = QuantumCircuit(2, name="O_c")  # identity
    O_r = QuantumCircuit(2, name="O_r")
    O_r.swap(0, 1)
    O_data, _ = make_data_oracle([1.0], A_max=A_max)
    spec = BlockEncodingSpec(
        n_idx=n_idx, n_s=n_s, n_d=n_d, A_max=A_max,
        O_c=O_c, O_r=O_r, O_data=O_data, name="U_ones",
    )
    be = build_block_encoding(spec)
    print(be.summary())
    A_ref = np.ones((2, 2))
    passed, err = be.verify_against(A_ref, atol=1e-10, verbose=True)
    assert passed, f"all-ones test failed (err={err})"

    print("\nAll bencode.builder tests passed.")

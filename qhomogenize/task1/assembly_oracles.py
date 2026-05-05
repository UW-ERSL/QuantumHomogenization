"""
Quantum oracles for the FE assembly operator A under PBC (Task 1).

For an L x L Cartesian mesh with PBC, A has shape (8 L^2, 2 L^2):
    A_{k, j} = 1  iff  k = 8*e + ell  and  j = global_dof(e, ell, L)
    (see qhomogenize.fea.assembly_pbc.global_dof for the closed-form map).

Sparsities:
    S_r = 1  (each row has exactly one nonzero -- A is a {0,1}-row-permutation
              of the global DOF register)
    S_c = 4  (each global DOF is shared by 4 incident elements in 2D)
    D = 1   (only the value 1)
    ||A||_max = 1

Sünderhauf framework requires bit-width consistency  N S = M D  with
N = 2^n_idx (after square padding), S = 2^n_s, D = 2^n_d, M = 2^n_m.
We use:
    n_idx = log_2(8 L^2) = 3 + 2 log_2 L   (matrix dimension after square padding)
    n_s   = 2  (S = 4)
    n_d   = 0  (D = 1)
    n_m   = n_idx + n_s - n_d = 5 + 2 log_2 L

Out-of-range:
- The matrix is padded square 8L^2 x 8L^2 with the actual A in the first 2L^2
  columns; the remaining 6L^2 columns are zero-padded.
- Each row has exactly 1 real nonzero, padded to 4 (= S) entries per row in
  the framework.  3 of the 4 sparsity slots per row are out-of-range.
- The out-of-range oracle flags m-values whose padding bits are nonzero.

Bit layout of the m register (low to high, n_m bits total):
    bit 0                            : dim (= ell mod 2)
    bits 1..n_log                    : i_x in [0, L)         where n_log = log_2 L
    bits (n_log+1)..(2 n_log)        : i_y in [0, L)
    bits (2 n_log+1)..(2 n_log+2)    : padding (2 bits, must be 0 for valid m)
    bits (2 n_log+3)..(2 n_log+4)    : c (= ell // 2, the corner index in [0, 4))

Bit layout of the (s_c, j) register (same physical qubits, after applying O_c):
    bit 0                            : dim of j
    bits 1..n_log                    : n_x = (i_x + dx_c) mod L
    bits (n_log+1)..(2 n_log)        : n_y = (i_y + dy_c) mod L
    bits (2 n_log+1)..(2 n_log+2)    : padding bits of j (high bits beyond 2 L^2 mark)
    bits (2 n_log+3)..(2 n_log+4)    : s_c = c

This layout makes O_c clean (just two modular adds, no bit permutation) and
O_r a cyclic shift-by-2 of bits 1..(n_m - 1) (bit 0 fixed).

Per the Andreassen-Andreasen corner ordering (LL, LR, UR, UL):
    c=0 (LL) : (dx, dy) = (0, 0)
    c=1 (LR) : (dx, dy) = (1, 0)
    c=2 (UR) : (dx, dy) = (1, 1)
    c=3 (UL) : (dx, dy) = (0, 1)
So dx_c = c[0] XOR c[1] (= 1 iff c in {1, 2})
   dy_c = c[1]          (= 1 iff c in {2, 3})

Cost
----
- O_c : 2 modular increments controlled on c.  Each increment is O(log L)
  multi-controlled X gates.  Total: O(log L) two-qubit gates.
- O_r : 2 (n_m - 1) SWAP gates = O(log L).
- O_rg: 3 gates (CX, CX, CCX).
- O_data: trivial (D = 1, single Y-rotation = X for value 1).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


# ---------------------------------------------------------------------------
# Internal arithmetic helpers
# ---------------------------------------------------------------------------

def _multi_controlled_increment_pow2(qc: QuantumCircuit, ctrls: Iterable, qreg: list, n: int) -> None:
    """qreg += 1 (mod 2^n), gated on AND of ctrls (treated as Toffoli controls).

    ctrls : iterable of Qubit -- extra control qubits.
    qreg  : list of n Qubits -- target register (low bit at index 0).
    n     : int -- bit width.
    """
    ctrls = list(ctrls)
    # Flip high-to-low: bit (n-1) controlled on ctrls + bits 0..(n-2) all 1, etc.
    for j in range(n - 1, 0, -1):
        controls = ctrls + [qreg[i] for i in range(j)]
        if len(controls) == 1:
            qc.cx(controls[0], qreg[j])
        else:
            qc.mcx(controls, qreg[j])
    # Flip bit 0 conditional on ctrls.
    if not ctrls:
        qc.x(qreg[0])
    elif len(ctrls) == 1:
        qc.cx(ctrls[0], qreg[0])
    else:
        qc.mcx(ctrls, qreg[0])


def _conditional_increment_pow2(
    qc: QuantumCircuit, ctrl_value: int, ctrls: list, qreg: list, n: int
) -> None:
    """qreg += 1 (mod 2^n) iff (ctrls == ctrl_value) as integer.

    Uses ancilla-free "X-flip-X" trick: temporarily X each control where
    ctrl_value bit is 0, then apply multi-controlled increment, then undo X's.
    """
    flipped = []
    for i, c in enumerate(ctrls):
        if (ctrl_value >> i) & 1 == 0:
            qc.x(c)
            flipped.append(c)
    _multi_controlled_increment_pow2(qc, ctrls, qreg, n)
    for c in flipped:
        qc.x(c)


# ---------------------------------------------------------------------------
# Bit-position helpers
# ---------------------------------------------------------------------------

def bit_layout(L: int) -> dict:
    """Return the bit-position layout for the m register at given L."""
    n_log = int(np.log2(L))
    if 2 ** n_log != L:
        raise NotImplementedError(f"L must be a power of 2; got L={L}")
    n_m = 5 + 2 * n_log
    return {
        "n_log": n_log,
        "n_m": n_m,
        "DIM": 0,
        "IX": list(range(1, 1 + n_log)),
        "IY": list(range(1 + n_log, 1 + 2 * n_log)),
        "PAD": [1 + 2 * n_log, 2 + 2 * n_log],
        "C": [3 + 2 * n_log, 4 + 2 * n_log],
    }


# ---------------------------------------------------------------------------
# Column oracle O_c
# ---------------------------------------------------------------------------

def assembly_column_oracle(L: int, name: str = "O_c_A") -> QuantumCircuit:
    """Build O_c for the assembly operator A.

    Maps |d, m> -> |s_c, j> (forward direction).  The builder applies O_c.inverse()
    to map (s_c, j) -> (d, m).

    Implementation is just two modular increments controlled on c:
        - i_x += dx_c  (mod L), where dx_c = c[0] XOR c[1].
        - i_y += dy_c  (mod L), where dy_c = c[1].
    """
    layout = bit_layout(L)
    n_m = layout["n_m"]
    n_log = layout["n_log"]
    qr = QuantumRegister(n_m, "m")
    qc = QuantumCircuit(qr, name=name)

    C0, C1 = qr[layout["C"][0]], qr[layout["C"][1]]
    ix_qubits = [qr[i] for i in layout["IX"]]
    iy_qubits = [qr[i] for i in layout["IY"]]

    # Step 1: i_x += dx_c, dx_c = c[0] XOR c[1] = 1 iff c in {1, 2}
    # case c == 1 (binary 01): c[0] = 1, c[1] = 0
    _conditional_increment_pow2(qc, ctrl_value=0b01, ctrls=[C0, C1], qreg=ix_qubits, n=n_log)
    # case c == 2 (binary 10): c[0] = 0, c[1] = 1
    _conditional_increment_pow2(qc, ctrl_value=0b10, ctrls=[C0, C1], qreg=ix_qubits, n=n_log)

    # Step 2: i_y += dy_c = c[1]
    _multi_controlled_increment_pow2(qc, ctrls=[C1], qreg=iy_qubits, n=n_log)

    return qc


# ---------------------------------------------------------------------------
# Row oracle O_r
# ---------------------------------------------------------------------------

def assembly_row_oracle(L: int, name: str = "O_r_A") -> QuantumCircuit:
    """Build O_r for the assembly operator A.

    Maps |d, m> -> |s_r, i>.  In our bit layout this is a cyclic shift-by-2
    of bits 1..(n_m - 1), with bit 0 fixed.

    After the shift:
        bit 0 (m's dim) -> stays as i's dim
        bits 1-2 (m's i_x) -> bits 3-4 (i's i_x)
        bits 3-4 (m's i_y) -> bits 5-6 (i's i_y)
        ...
        bits (2 n_log+3)-(2 n_log+4) (m's c) -> bits 1-2 (i's c, since
            ell = 2c + dim, so i's c sits at bits 1-2)

    For valid m, the padding bits of m become s_r = 0 in the output (the
    "high" bits of the final register).
    """
    layout = bit_layout(L)
    n_m = layout["n_m"]
    qr = QuantumRegister(n_m, "m")
    qc = QuantumCircuit(qr, name=name)

    # Cyclic shift right by 2 of bits 1..(n_m - 1).  Bit 0 stays.
    # Implement as two consecutive "shift right by 1" passes.
    # Each pass: swap(n_m-2, n_m-1); swap(n_m-3, n_m-2); ...; swap(1, 2).
    # That's a "rotate right by 1" of the upper (n_m - 1)-bit subregister.
    rotate_qubits = list(range(1, n_m))  # bits 1..(n_m - 1)

    def shift_right_by_1():
        # bubble the top bit down to position 1; equivalent to cyclic
        # shift right by 1 of the rotate_qubits.
        for k in range(len(rotate_qubits) - 1, 0, -1):
            a = rotate_qubits[k - 1]
            b = rotate_qubits[k]
            qc.swap(qr[a], qr[b])

    shift_right_by_1()
    shift_right_by_1()

    return qc


# ---------------------------------------------------------------------------
# Out-of-range oracle O_rg
# ---------------------------------------------------------------------------

def assembly_out_of_range_oracle(L: int, name: str = "O_rg_A") -> QuantumCircuit:
    """Build O_rg: del ^= (m's padding bits are nonzero).

    Operates on (n_m + 1) qubits: low n_m qubits are m, top qubit is del.
    Implements del ^= b_pad0 OR b_pad1 via three CNOTs:
        CX b_pad0 -> del   (del ^= b_pad0)
        CX b_pad1 -> del   (del ^= b_pad1)
        CCX (b_pad0, b_pad1) -> del   (del ^= b_pad0 AND b_pad1, correcting
                                       the double-count when both are 1)

    Verify by truth table:
        b_pad0 b_pad1 | del-after
        0 0           | 0 ^ 0 ^ 0 ^ 0 = 0
        1 0           | 0 ^ 1 ^ 0 ^ 0 = 1
        0 1           | 0 ^ 0 ^ 1 ^ 0 = 1
        1 1           | 0 ^ 1 ^ 1 ^ 1 = 1
    """
    layout = bit_layout(L)
    n_m = layout["n_m"]
    qr = QuantumRegister(n_m + 1, "m_del")
    qc = QuantumCircuit(qr, name=name)

    pad0, pad1 = qr[layout["PAD"][0]], qr[layout["PAD"][1]]
    del_q = qr[n_m]

    qc.cx(pad0, del_q)
    qc.cx(pad1, del_q)
    qc.ccx(pad0, pad1, del_q)

    return qc


# ---------------------------------------------------------------------------
# Quick test: verify the oracle behavior on basis states (no Sünderhauf builder yet)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    from qhomogenize.fea.assembly_pbc import global_dof

    print("--- Verifying assembly oracles at L=2 ---")
    L = 2
    layout = bit_layout(L)
    n_log = layout["n_log"]
    n_m = layout["n_m"]
    print(f"L={L}: n_log={n_log}, n_m={n_m}, layout={layout}")

    # Build O_c.  For each m = (dim, i_x, i_y, padding=0, c) representing a real
    # nonzero entry, applying O_c should produce (s_c, j) where:
    #   s_c = c
    #   j = 2 * ((i_x + dx_c) mod L + L * ((i_y + dy_c) mod L)) + dim
    O_c = assembly_column_oracle(L)
    O_r = assembly_row_oracle(L)
    O_rg = assembly_out_of_range_oracle(L)

    # Test O_c: enumerate all valid m and check the output (s_c, j) bit pattern.
    sim = AerSimulator(method="matrix_product_state")
    failures = 0
    n_tested = 0
    for dim in (0, 1):
        for i_x in range(L):
            for i_y in range(L):
                for c in range(4):
                    # Encode m as integer using the bit layout.
                    m_int = (dim << layout["DIM"])
                    for k in range(n_log):
                        m_int |= ((i_x >> k) & 1) << layout["IX"][k]
                        m_int |= ((i_y >> k) & 1) << layout["IY"][k]
                    m_int |= (c & 1) << layout["C"][0]
                    m_int |= ((c >> 1) & 1) << layout["C"][1]

                    # Build a circuit: prepare |m> on n_m qubits, apply O_c, measure.
                    tqc = QuantumCircuit(n_m)
                    for k in range(n_m):
                        if (m_int >> k) & 1:
                            tqc.x(k)
                    tqc.compose(O_c, inplace=True)
                    tqc.measure_all()
                    tqc_dec = tqc.decompose(reps=10)
                    counts = sim.run(transpile(tqc_dec, sim, optimization_level=0), shots=1).result().get_counts()
                    outcome = next(iter(counts.keys())).replace(" ", "")[::-1]
                    output_int = int(outcome[::-1], 2)

                    # Decode output as (s_c, j):
                    # bits 0 = dim of j
                    # bits 1..n_log = n_x
                    # bits (n_log+1)..(2 n_log) = n_y
                    # bits (2 n_log+1)..(2 n_log+2) = padding (should be 0 for valid)
                    # bits (2 n_log+3)..(2 n_log+4) = s_c (= c)
                    out_dim = (output_int >> 0) & 1
                    out_nx = 0
                    for k in range(n_log):
                        out_nx |= ((output_int >> (1 + k)) & 1) << k
                    out_ny = 0
                    for k in range(n_log):
                        out_ny |= ((output_int >> (1 + n_log + k)) & 1) << k
                    out_pad0 = (output_int >> layout["PAD"][0]) & 1
                    out_pad1 = (output_int >> layout["PAD"][1]) & 1
                    out_sc0 = (output_int >> layout["C"][0]) & 1
                    out_sc1 = (output_int >> layout["C"][1]) & 1
                    out_sc = out_sc0 | (out_sc1 << 1)

                    # Expected:
                    corner_offsets = [(0, 0), (1, 0), (1, 1), (0, 1)]
                    dx_c, dy_c = corner_offsets[c]
                    exp_nx = (i_x + dx_c) % L
                    exp_ny = (i_y + dy_c) % L
                    exp_dim = dim
                    exp_sc = c
                    if (out_dim, out_nx, out_ny, out_pad0, out_pad1, out_sc) != \
                       (exp_dim, exp_nx, exp_ny, 0, 0, exp_sc):
                        failures += 1
                        print(f"  m=(dim={dim},ix={i_x},iy={i_y},c={c}) -> "
                              f"got (dim={out_dim},nx={out_nx},ny={out_ny},pad=({out_pad0},{out_pad1}),sc={out_sc}), "
                              f"expected (dim={exp_dim},nx={exp_nx},ny={exp_ny},pad=(0,0),sc={exp_sc})")
                    n_tested += 1

    if failures == 0:
        print(f"  O_c: all {n_tested} basis states correct.")
    else:
        print(f"  O_c: {failures}/{n_tested} failures.")

    # Test O_r: enumerate valid m, apply O_r, verify output is (s_r=0, i) where
    # i = 8*(i_x + L*i_y) + (2*c + dim).
    failures = 0
    n_tested = 0
    for dim in (0, 1):
        for i_x in range(L):
            for i_y in range(L):
                for c in range(4):
                    m_int = (dim << layout["DIM"])
                    for k in range(n_log):
                        m_int |= ((i_x >> k) & 1) << layout["IX"][k]
                        m_int |= ((i_y >> k) & 1) << layout["IY"][k]
                    m_int |= (c & 1) << layout["C"][0]
                    m_int |= ((c >> 1) & 1) << layout["C"][1]

                    tqc = QuantumCircuit(n_m)
                    for k in range(n_m):
                        if (m_int >> k) & 1:
                            tqc.x(k)
                    tqc.compose(O_r, inplace=True)
                    tqc.measure_all()
                    tqc_dec = tqc.decompose(reps=10)
                    counts = sim.run(transpile(tqc_dec, sim, optimization_level=0), shots=1).result().get_counts()
                    outcome = next(iter(counts.keys())).replace(" ", "")[::-1]
                    output_int = int(outcome[::-1], 2)

                    # Decode (s_r, i) at bits: low n_idx = i, high n_s = s_r.
                    # n_idx = 3 + 2 n_log, n_s = 2.  In our 7-bit register:
                    n_idx = 3 + 2 * n_log
                    i_val = output_int & ((1 << n_idx) - 1)
                    sr_val = (output_int >> n_idx) & 0b11

                    # Expected i:
                    e = i_x + L * i_y
                    ell = 2 * c + dim
                    exp_i = 8 * e + ell
                    if (i_val, sr_val) != (exp_i, 0):
                        failures += 1
                        print(f"  m=(dim={dim},ix={i_x},iy={i_y},c={c}) -> "
                              f"got (i={i_val}, s_r={sr_val}), expected (i={exp_i}, s_r=0)")
                    # Cross-check against fea.assembly_pbc.global_dof: for the (e, ell) -> j mapping.
                    # Note: O_r doesn't compute j; it computes i.  The match exp_i = 8e + ell is the row index.
                    n_tested += 1

    if failures == 0:
        print(f"  O_r: all {n_tested} basis states correct.")
    else:
        print(f"  O_r: {failures}/{n_tested} failures.")

    # Test O_rg: enumerate all m (valid AND padded) and verify del flag.
    failures = 0
    n_tested = 0
    for m_int in range(2 ** n_m):
        for del_in in (0, 1):
            tqc = QuantumCircuit(n_m + 1)
            for k in range(n_m):
                if (m_int >> k) & 1:
                    tqc.x(k)
            if del_in:
                tqc.x(n_m)
            tqc.compose(O_rg, inplace=True)
            tqc.measure_all()
            tqc_dec = tqc.decompose(reps=10)
            counts = sim.run(transpile(tqc_dec, sim, optimization_level=0), shots=1).result().get_counts()
            outcome = next(iter(counts.keys())).replace(" ", "")[::-1]
            output_int = int(outcome[::-1], 2)

            # m unchanged, del = del_in XOR (pad0 OR pad1)
            m_out = output_int & ((1 << n_m) - 1)
            del_out = (output_int >> n_m) & 1
            pad0 = (m_int >> layout["PAD"][0]) & 1
            pad1 = (m_int >> layout["PAD"][1]) & 1
            exp_del = del_in ^ (pad0 | pad1)

            if (m_out, del_out) != (m_int, exp_del):
                failures += 1
            n_tested += 1
    if failures == 0:
        print(f"  O_rg: all {n_tested} states correct.")
    else:
        print(f"  O_rg: {failures}/{n_tested} failures.")

    print("\nAll assembly_oracles tests passed." if failures == 0 else "")

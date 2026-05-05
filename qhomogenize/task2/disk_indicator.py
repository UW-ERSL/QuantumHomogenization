"""
Disk-indicator oracle O_p: |i_x>|i_y>|0> -> |i_x>|i_y>|p>.

For a circular inclusion centered at (cx_unit, cy_unit) with radius r_unit on
the unit cell, element e = i_x + L * i_y is "solid" (p_e = 1) iff its center
lies inside the disk:

    ((i_x + 0.5)/L - cx_unit)^2 + ((i_y + 0.5)/L - cy_unit)^2 < r_unit^2.

After multiplying through by (2 L)^2 to clear half-integer offsets:

    (2 i_x + 1 - Cx)^2 + (2 i_y + 1 - Cy)^2 < R2,

where Cx = round(2 L cx_unit), Cy = round(2 L cy_unit), R2 = round((2 L r_unit)^2).

Implementation: unsigned-only via algebraic expansion
-----------------------------------------------------
The naive subtraction (2 i_x + 1) - Cx is signed and would require a signed
multiplier for the squaring step.  We avoid this entirely by expanding:

    (a - b)^2 = a^2 - 2 a b + b^2,

so the membership test (a-Cx)^2 + (b-Cy)^2 < R2 with a := 2 i_x + 1 and
b := 2 i_y + 1 becomes

    (a^2 + b^2 + Cx^2 + Cy^2)  <  (2 Cx a + 2 Cy b + R2).

Both sides are non-negative for i_x, i_y in [0, L) and Cx, Cy in [0, 2L].
We compute LHS and RHS into unsigned accumulators and compare via a
quantum-quantum comparator (subtract-with-borrow), then uncompute everything.

Cost
----
- 2 unsigned squarings: O((log L)^2) gates each (Qiskit MultiplierGate).
- 2 classical-quantum multiplications: O((log L)^2) controlled adds.
- 2 register-to-register adds: O(log L) gates each.
- 1 quantum-quantum comparator: O(log L) gates.
Total: O((log L)^2) two-qubit gates.

Ancilla count: O(log L) -- bounded by ~ 10 (log L) qubits.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MultiplierGate, HalfAdderGate


# ---------------------------------------------------------------------------
# Inline arithmetic helpers (kept here rather than in qprim.integer because
# their interface is specific to the disk-indicator pipeline)
# ---------------------------------------------------------------------------

def _encode_2i_plus_1(qc: QuantumCircuit, i_reg: QuantumRegister, a_reg: QuantumRegister) -> None:
    """a_reg := 2 * i_reg + 1, both registers in computational basis."""
    if a_reg.size < i_reg.size + 1:
        raise ValueError(f"a_reg too small: {a_reg.size} < {i_reg.size + 1}")
    qc.x(a_reg[0])
    for k in range(i_reg.size):
        qc.cx(i_reg[k], a_reg[k + 1])


def _encode_2i_plus_1_inverse(qc: QuantumCircuit, i_reg: QuantumRegister, a_reg: QuantumRegister) -> None:
    for k in reversed(range(i_reg.size)):
        qc.cx(i_reg[k], a_reg[k + 1])
    qc.x(a_reg[0])


def _square_into(qc: QuantumCircuit, a_reg: QuantumRegister, sq_reg: QuantumRegister, copy_reg: QuantumRegister) -> None:
    """sq_reg := a_reg * a_reg, using copy_reg as ancilla for the multiplier."""
    if copy_reg.size != a_reg.size:
        raise ValueError("copy_reg size must equal a_reg size")
    if sq_reg.size != 2 * a_reg.size:
        raise ValueError("sq_reg size must equal 2 * a_reg size")
    for k in range(a_reg.size):
        qc.cx(a_reg[k], copy_reg[k])
    mul = MultiplierGate(num_state_qubits=a_reg.size, num_result_qubits=sq_reg.size)
    qc.append(mul, [*a_reg, *copy_reg, *sq_reg])


def _square_into_inverse(qc: QuantumCircuit, a_reg: QuantumRegister, sq_reg: QuantumRegister, copy_reg: QuantumRegister) -> None:
    mul_inv = MultiplierGate(num_state_qubits=a_reg.size, num_result_qubits=sq_reg.size).inverse()
    qc.append(mul_inv, [*a_reg, *copy_reg, *sq_reg])
    for k in reversed(range(a_reg.size)):
        qc.cx(a_reg[k], copy_reg[k])


def _add_register_into(qc: QuantumCircuit, src: QuantumRegister, dst: QuantumRegister) -> None:
    """In-place dst += src using HalfAdderGate.  dst.size >= src.size + 1 required."""
    n = src.size
    if dst.size < n + 1:
        raise ValueError(f"dst size {dst.size} must be >= src.size + 1 = {n + 1}")
    adder = HalfAdderGate(num_state_qubits=n)
    qc.append(adder, [*src, *(dst[i] for i in range(n)), dst[n]])


def _add_register_into_inverse(qc: QuantumCircuit, src: QuantumRegister, dst: QuantumRegister) -> None:
    n = src.size
    adder_inv = HalfAdderGate(num_state_qubits=n).inverse()
    qc.append(adder_inv, [*src, *(dst[i] for i in range(n)), dst[n]])


def _add_constant(qc: QuantumCircuit, qreg: QuantumRegister, c: int) -> None:
    """In-place qreg += c (mod 2^n) via cascade of multi-controlled flips."""
    if c == 0:
        return
    n = qreg.size
    c_eff = c % (2 ** n)
    for k in range(n):
        if (c_eff >> k) & 1:
            for j in range(n - 1, k, -1):
                control_bits = list(range(k, j))
                if len(control_bits) == 1:
                    qc.cx(qreg[control_bits[0]], qreg[j])
                else:
                    qc.mcx([qreg[i] for i in control_bits], qreg[j])
            qc.x(qreg[k])


def _add_constant_inverse(qc: QuantumCircuit, qreg: QuantumRegister, c: int) -> None:
    _add_constant(qc, qreg, -c)


def _classical_quantum_multiply_add(
    qc: QuantumCircuit,
    coeff: int,
    a_reg: QuantumRegister,
    dst: QuantumRegister,
) -> None:
    """In-place dst += coeff * a_reg, coeff a non-negative classical integer."""
    if coeff == 0:
        return
    if coeff < 0:
        raise ValueError("coeff must be non-negative")
    n_a = a_reg.size
    n_d = dst.size
    for k in range(n_a):
        addend = (coeff << k) & ((1 << n_d) - 1)
        if addend == 0:
            continue
        for j in range(n_d):
            if (addend >> j) & 1:
                for i in range(n_d - 1, j, -1):
                    control_bits = [a_reg[k]] + [dst[t] for t in range(j, i)]
                    if len(control_bits) == 1:
                        qc.cx(control_bits[0], dst[i])
                    else:
                        qc.mcx(control_bits, dst[i])
                qc.cx(a_reg[k], dst[j])


def _classical_quantum_multiply_add_inverse(
    qc: QuantumCircuit,
    coeff: int,
    a_reg: QuantumRegister,
    dst: QuantumRegister,
) -> None:
    """Inverse: dst -= coeff * a_reg."""
    if coeff == 0:
        return
    n_a = a_reg.size
    n_d = dst.size
    for k in reversed(range(n_a)):
        addend = (coeff << k) & ((1 << n_d) - 1)
        if addend == 0:
            continue
        for j in reversed(range(n_d)):
            if (addend >> j) & 1:
                qc.cx(a_reg[k], dst[j])
                for i in range(j + 1, n_d):
                    control_bits = [a_reg[k]] + [dst[t] for t in range(j, i)]
                    if len(control_bits) == 1:
                        qc.cx(control_bits[0], dst[i])
                    else:
                        qc.mcx(control_bits, dst[i])


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def disk_indicator_oracle(
    L: int,
    Cx: int,
    Cy: int,
    R2: int,
    name: str = "O_p",
) -> tuple[QuantumCircuit, dict]:
    """Reversible indicator oracle |i_x>|i_y>|0> -> |i_x>|i_y>|p>.

    Logic:
        a := 2 i_x + 1, b := 2 i_y + 1
        LHS := a^2 + b^2 + Cx^2 + Cy^2
        RHS := 2 Cx a + 2 Cy b + R2
        p := (LHS < RHS)

    Parameters
    ----------
    L : int (power of 2)
    Cx, Cy, R2 : int
    name : str

    Returns
    -------
    qc : QuantumCircuit
    info : dict with keys {qreg_ix, qreg_iy, flag, n_x, n_y, n_acc, n_qubits}
    """
    n_x = int(np.ceil(np.log2(L))) if L > 1 else 1
    n_y = n_x
    if 2 ** n_x != L:
        raise NotImplementedError(f"L must be a power of 2; got L={L}")

    n_a = n_x + 1            # a in [1, 2L-1]
    n_sq = 2 * n_a           # a^2

    lhs_max = 2 * (2 * L - 1) ** 2 + Cx ** 2 + Cy ** 2
    rhs_max = 2 * Cx * (2 * L - 1) + 2 * Cy * (2 * L - 1) + R2
    accum_max = max(lhs_max, rhs_max)
    n_acc = max(int(np.ceil(np.log2(accum_max + 1))) + 1, 4)  # +1 for headroom

    q_ix = QuantumRegister(n_x, "ix")
    q_iy = QuantumRegister(n_y, "iy")
    q_flag = QuantumRegister(1, "flag")
    q_a = QuantumRegister(n_a, "a")
    q_b = QuantumRegister(n_a, "b")
    q_a_sq = QuantumRegister(n_sq, "asq")
    q_b_sq = QuantumRegister(n_sq, "bsq")
    q_a_copy = QuantumRegister(n_a, "ac")
    q_b_copy = QuantumRegister(n_a, "bc")
    q_lhs = QuantumRegister(n_acc, "lhs")
    q_rhs = QuantumRegister(n_acc, "rhs")
    q_borrow = QuantumRegister(1, "borrow")

    qc = QuantumCircuit(
        q_ix, q_iy, q_flag,
        q_a, q_b, q_a_sq, q_b_sq, q_a_copy, q_b_copy, q_lhs, q_rhs, q_borrow,
        name=name,
    )

    # ---------- forward ----------
    _encode_2i_plus_1(qc, q_ix, q_a)
    _encode_2i_plus_1(qc, q_iy, q_b)

    _square_into(qc, q_a, q_a_sq, q_a_copy)
    _square_into(qc, q_b, q_b_sq, q_b_copy)

    _add_register_into(qc, q_a_sq, q_lhs)
    _add_register_into(qc, q_b_sq, q_lhs)
    _add_constant(qc, q_lhs, Cx * Cx + Cy * Cy)

    _classical_quantum_multiply_add(qc, 2 * Cx, q_a, q_rhs)
    _classical_quantum_multiply_add(qc, 2 * Cy, q_b, q_rhs)
    _add_constant(qc, q_rhs, R2)

    # ---------- compare: flag := (lhs < rhs) ----------
    # Strategy: rhs := rhs - lhs.  Borrow=1 iff lhs > rhs; result==0 iff lhs == rhs.
    # flag = (NOT borrow) AND (rhs != 0)
    n = q_lhs.size
    adder_inv = HalfAdderGate(num_state_qubits=n).inverse()
    qc.append(adder_inv, [*q_lhs, *q_rhs, q_borrow[0]])

    qc.x(q_flag[0])
    qc.cx(q_borrow[0], q_flag[0])
    # Now flag = NOT borrow.  AND with (rhs != 0):
    for q in q_rhs:
        qc.x(q)
    qc.mcx([*q_rhs], q_flag[0])      # flips flag iff all q_rhs (after X) are 1, i.e. rhs == 0
    for q in q_rhs:
        qc.x(q)

    # Restore rhs by re-adding lhs.
    adder = HalfAdderGate(num_state_qubits=n)
    qc.append(adder, [*q_lhs, *q_rhs, q_borrow[0]])

    # ---------- uncompute ----------
    _add_constant_inverse(qc, q_rhs, R2)
    _classical_quantum_multiply_add_inverse(qc, 2 * Cy, q_b, q_rhs)
    _classical_quantum_multiply_add_inverse(qc, 2 * Cx, q_a, q_rhs)

    _add_constant_inverse(qc, q_lhs, Cx * Cx + Cy * Cy)
    _add_register_into_inverse(qc, q_b_sq, q_lhs)
    _add_register_into_inverse(qc, q_a_sq, q_lhs)

    _square_into_inverse(qc, q_b, q_b_sq, q_b_copy)
    _square_into_inverse(qc, q_a, q_a_sq, q_a_copy)

    _encode_2i_plus_1_inverse(qc, q_iy, q_b)
    _encode_2i_plus_1_inverse(qc, q_ix, q_a)

    info = dict(
        qreg_ix=q_ix, qreg_iy=q_iy, flag=q_flag[0],
        n_x=n_x, n_y=n_y,
        n_acc=n_acc,
        n_qubits=qc.num_qubits,
    )
    return qc, info


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    from qhomogenize.fea.microstructure import disk_indicator_array, disk_indicator_grid_units

    print("--- Testing disk_indicator_oracle at L=4 ---")
    L = 4
    cx_u, cy_u, r_u = 0.5, 0.5, 0.3
    Cx, Cy, R2 = disk_indicator_grid_units(L, cx_u, cy_u, r_u)
    print(f"L={L}, (Cx, Cy, R2) = ({Cx}, {Cy}, {R2})")

    p_ref = disk_indicator_array(L, cx_u, cy_u, r_u)
    print("Reference indicator (rows = iy, cols = ix):")
    for row in p_ref:
        print("  " + " ".join(str(v) for v in row))

    qc, info = disk_indicator_oracle(L, Cx, Cy, R2)
    n_x, n_y = info["n_x"], info["n_y"]
    print(f"Circuit qubits: {qc.num_qubits} (n_x={n_x}, n_y={n_y}, n_acc={info['n_acc']})")

    sim = AerSimulator(method="matrix_product_state")
    failures = 0
    for ix in range(L):
        for iy in range(L):
            tqc = QuantumCircuit(*qc.qregs)
            for k in range(n_x):
                if (ix >> k) & 1:
                    tqc.x(info["qreg_ix"][k])
            for k in range(n_y):
                if (iy >> k) & 1:
                    tqc.x(info["qreg_iy"][k])
            tqc.compose(qc, inplace=True)
            tqc.measure_all()
            tqc_dec = tqc.decompose(reps=10)
            tqc_compiled = transpile(tqc_dec, sim, optimization_level=0)
            counts = sim.run(tqc_compiled, shots=1).result().get_counts()
            outcome = next(iter(counts.keys())).replace(" ", "")[::-1]
            ix_out = int(outcome[0:n_x][::-1], 2) if n_x > 0 else 0
            iy_out = int(outcome[n_x:n_x + n_y][::-1], 2) if n_y > 0 else 0
            flag_out = int(outcome[n_x + n_y])
            ancilla_bits = outcome[n_x + n_y + 1:]
            ancilla_nonzero = any(c == "1" for c in ancilla_bits)
            expected_flag = int(p_ref[iy, ix])
            ok = (ix_out, iy_out, flag_out, ancilla_nonzero) == (ix, iy, expected_flag, False)
            if not ok:
                failures += 1
                print(f"  ix={ix}, iy={iy}: ix_out={ix_out}, iy_out={iy_out}, "
                      f"flag={flag_out} (expected {expected_flag}), ancilla_nonzero={ancilla_nonzero}")
    if failures == 0:
        print(f"  ALL {L * L} cases pass.  OK")
    else:
        print(f"  {failures} failures out of {L * L} cases.")

"""
Quantum integer arithmetic primitives, transpiled to (u, cx).

Thin wrappers over Qiskit 2.x's circuit-library arithmetic gates, exposed at
the abstraction level the POC needs.  Three core primitives:

- `add_constant_modular(qc, qreg, c, modulus)`: modular addition by classical
  constant.  Used in task1 PBC wraparound.
- `subtract_constant(qc, qreg, c)`: subtract a classical constant
  (no modular reduction; sized so result fits).  Used in task2 to compute
  i_x - c_x without wraparound.
- `square_into(qc, qreg_in, qreg_out)`: in-place-from-input squaring,
  qreg_out := qreg_in * qreg_in.
- `compare_less(qc, qreg, threshold, flag_qubit)`: compare-less-than a
  classical constant; flip flag_qubit if (qreg < threshold).

These cover everything the indicator oracle and the assembly oracle need.
The data oracle (q-controlled value loading) sits separately in
qprim/data_oracle.py since it has a different interface (looks up a value,
doesn't compute one).

Implementation choices
----------------------
- Adders: CDKMRippleCarryAdder for non-modular, ModularAdderGate for
  modular.  Both are O(log L) two-qubit gates after transpile.
- Multiplier: MultiplierGate (the modern circuit-library gate; auto-selects
  HRSCumulative or RGQFT under the hood).  O((log L)^2) gates.
- Comparator: IntegerComparatorGate.  O(log L) gates.

Parameters use POSITIVE little-endian integer registers throughout: bit 0 of
the register is the least significant bit of the integer.  Negative results
are represented in two's complement when subtraction can produce them; we
keep things in non-negative regimes for the disk indicator by squaring before
comparison.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (
    CDKMRippleCarryAdder,
    ModularAdderGate,
    MultiplierGate,
    IntegerComparatorGate,
)


# ---------------------------------------------------------------------------
# Constant-add helpers
# ---------------------------------------------------------------------------

def add_constant_modular(qc: QuantumCircuit, qreg: QuantumRegister, c: int, modulus: int) -> None:
    """In-place |x> -> |(x + c) mod modulus> on `qreg`.

    Uses ModularAdderGate with a temporary classical-constant register.  The
    constant register is created within the circuit, the addition performed,
    and the constant register left in state |c>; we do not uncompute it
    (caller can manage via barriers / out-of-scope ancilla recycling if needed).

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to append onto.
    qreg : QuantumRegister of size n
        The "x" register.  modulus must satisfy modulus <= 2^n.
    c : int in [0, modulus)
        Constant to add.
    modulus : int, default 2^n
        Modulus.  For PBC over an L-element axis, modulus = L.
    """
    n = qreg.size
    if modulus > 2 ** n:
        raise ValueError(f"modulus {modulus} exceeds 2^{n}")
    if not (0 <= c < modulus):
        raise ValueError(f"constant c={c} not in [0, {modulus})")
    if c == 0:
        return  # no-op

    if modulus != 2 ** n:
        # The general case (L not a power of 2) requires a non-power-of-2 modular
        # adder. For the POC we only consider L in {4, 8, 16}, all powers of 2,
        # so we don't implement the generalization here. Punt.
        raise NotImplementedError(
            f"Non-power-of-2 modulus {modulus} not implemented.  POC uses L in {{4, 8, 16}}."
        )

    # Power-of-2 modulus: addition by classical constant truncated to n bits is
    # just a sequence of controlled increments. Implement directly without
    # ModularAdderGate (which expects a quantum addend register).
    _add_constant_power_of_2(qc, qreg, c, n)


def _add_constant_power_of_2(qc: QuantumCircuit, qreg: QuantumRegister, c: int, n: int) -> None:
    """In-place |x> -> |(x + c) mod 2^n> via decomposition c = sum_k c_k 2^k.

    For each set bit c_k of the constant, we apply an n-qubit "increment by 2^k"
    operation.  Increment-by-2^k on an n-bit register can be implemented as a
    cascade of (n - k - 1) MCX gates targeting bits k+1, k+2, ..., n-1 followed
    by an X on bit k -- but we need to be careful about the order.

    Specifically: |x> -> |x + 2^k mod 2^n> for k=0..n-1 is the standard "increment
    by 2^k", which flips bit k and propagates the carry to higher bits.

    We implement this directly with a loop of MCX gates from the most-significant
    bit downwards, then an X on bit k.
    """
    for k in range(n):
        if (c >> k) & 1:
            # Increment by 2^k on bits [k, n).  Walk from high bit to low bit.
            # Bit j flips iff bits k, k+1, ..., j-1 are all 1.
            for j in range(n - 1, k, -1):
                control_bits = list(range(k, j))
                if len(control_bits) == 1:
                    qc.cx(qreg[control_bits[0]], qreg[j])
                else:
                    qc.mcx([qreg[i] for i in control_bits], qreg[j])
            qc.x(qreg[k])


def add_constant_nonmodular(qc: QuantumCircuit, qreg: QuantumRegister, c: int) -> None:
    """In-place |x> -> |x + c> on `qreg`, assuming the result fits in the register.

    Used in task2 where we have a wider register that holds (i - c) without
    wraparound.  Caller must ensure qreg has enough bits.
    """
    n = qreg.size
    if c == 0:
        return
    if c > 0:
        _add_constant_power_of_2(qc, qreg, c, n)
    else:
        # x + c = x - |c| = x + (2^n - |c|) mod 2^n.  But for non-modular we'd
        # underflow if x < |c|.  Caller is responsible for keeping x >= |c|, or
        # accepting the modular semantics which equivalently encode -|c|.
        c_eff = (2 ** n + c) % (2 ** n)
        _add_constant_power_of_2(qc, qreg, c_eff, n)


# ---------------------------------------------------------------------------
# Multiplication / squaring
# ---------------------------------------------------------------------------

def multiplier_gate(n_a: int, n_b: int = None, n_out: int = None) -> MultiplierGate:
    """Return a Qiskit MultiplierGate for two unsigned integer registers.

    The returned gate operates on (n_a + n_b + n_out) qubits with the convention
    [a..., b..., out...] (lowest qubits first).
    """
    if n_b is None:
        n_b = n_a
    if n_out is None:
        n_out = n_a + n_b
    return MultiplierGate(num_state_qubits=n_a, num_result_qubits=n_out)


def append_squared(qc: QuantumCircuit, q_in: QuantumRegister, q_out: QuantumRegister, q_copy: QuantumRegister) -> None:
    """Append q_out += q_in^2 by first copying q_in into q_copy, then multiplying.

    Strategy: MultiplierGate operates on two distinct input registers, but for
    squaring we have only one source register.  We CNOT-copy q_in into a fresh
    ancilla register q_copy (correct because q_in is in computational basis after
    a comparator/adder result), apply the multiplier on (q_in, q_copy) -> q_out,
    and leave q_copy holding the original value (which the caller is responsible
    for uncomputing).

    Parameters
    ----------
    q_in : QuantumRegister of size n
    q_out : QuantumRegister of size 2n
    q_copy : QuantumRegister of size n
        Fresh ancilla, expected to be in |0...0>; will hold a copy of q_in
        on return.
    """
    if q_copy.size != q_in.size:
        raise ValueError(f"q_copy size {q_copy.size} must equal q_in size {q_in.size}")
    if q_out.size != 2 * q_in.size:
        raise ValueError(f"q_out size {q_out.size} must equal 2 * q_in size = {2 * q_in.size}")

    # CNOT-copy q_in -> q_copy
    for k in range(q_in.size):
        qc.cx(q_in[k], q_copy[k])

    # Multiply: q_out += q_in * q_copy
    mul = multiplier_gate(n_a=q_in.size, n_b=q_copy.size, n_out=q_out.size)
    qc.append(mul, [*q_in, *q_copy, *q_out])


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------

def compare_constant(
    qc: QuantumCircuit,
    qreg: QuantumRegister,
    threshold: int,
    flag_qubit,
    geq: bool = True,
) -> None:
    """Compare qreg against a classical constant; flip flag_qubit on match.

    Uses Qiskit 2.x's `IntegerComparatorGate`, which has a clean interface
    -- ancillas are managed internally during synthesis.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to append to.
    qreg : QuantumRegister of size n
        State register (interpreted as unsigned little-endian integer).
    threshold : int
        Classical constant to compare against.
    flag_qubit : Qubit
        Target qubit (XOR'ed with the comparison result).
    geq : bool, default True
        If True, flag := flag XOR (qreg >= threshold).
        If False, flag := flag XOR (qreg < threshold).

    Notes
    -----
    Qiskit's gate convention: `IntegerComparatorGate(n, value, geq=True)`
    flips the target if state >= value.  We pass through the geq flag.
    """
    n = qreg.size
    cmp = IntegerComparatorGate(num_state_qubits=n, value=threshold, geq=geq)
    qc.append(cmp, [*qreg, flag_qubit])


# ---------------------------------------------------------------------------
# Quick tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from qiskit.quantum_info import Statevector

    # Test 1: add_constant_modular on a 3-qubit (mod 8) register
    print("--- Test add_constant_modular (mod 8) ---")
    n = 3
    for x in range(8):
        for c in range(8):
            qr = QuantumRegister(n, "x")
            qc = QuantumCircuit(qr)
            # initialize |x>
            for k in range(n):
                if (x >> k) & 1:
                    qc.x(qr[k])
            add_constant_modular(qc, qr, c, modulus=2 ** n)
            sv = Statevector.from_instruction(qc)
            # find the basis state with amplitude 1
            idx = int(np.argmax(np.abs(sv.data)))
            expected = (x + c) % 8
            assert idx == expected, f"x={x}, c={c}: got {idx}, expected {expected}"
    print("  OK (8 x 8 = 64 cases)")

    # Test 2: square via append_squared
    print("--- Test append_squared (3-bit -> 6-bit) ---")
    for x in range(8):
        n = 3
        q_in = QuantumRegister(n, "in")
        q_out = QuantumRegister(2 * n, "out")
        q_copy = QuantumRegister(n, "copy")
        qc = QuantumCircuit(q_in, q_copy, q_out)
        for k in range(n):
            if (x >> k) & 1:
                qc.x(q_in[k])
        append_squared(qc, q_in, q_out, q_copy)
        sv = Statevector.from_instruction(qc)
        idx = int(np.argmax(np.abs(sv.data)))
        # Bit layout (Qiskit little-endian):
        #   bits 0..n-1  : q_in
        #   bits n..2n-1 : q_copy
        #   bits 2n..4n-1: q_out
        in_val = idx & ((1 << n) - 1)
        copy_val = (idx >> n) & ((1 << n) - 1)
        out_val = (idx >> (2 * n)) & ((1 << (2 * n)) - 1)
        assert in_val == x, f"q_in mutated: {in_val} != {x}"
        assert copy_val == x, f"q_copy expected {x}, got {copy_val}"
        assert out_val == x * x, f"x={x}: q_out = {out_val}, expected {x*x}"
    print("  OK (8 cases)")

    # Test 3: compare_constant -- (qreg < threshold) and (qreg >= threshold)
    print("--- Test compare_constant ---")
    n = 3
    for threshold in (0, 1, 4, 7, 8):
        for x in range(8):
            for geq in (False, True):
                qr = QuantumRegister(n, "x")
                qf = QuantumRegister(1, "f")
                qc = QuantumCircuit(qr, qf)
                for k in range(n):
                    if (x >> k) & 1:
                        qc.x(qr[k])
                compare_constant(qc, qr, threshold, qf[0], geq=geq)
                sv = Statevector.from_instruction(qc)
                idx = int(np.argmax(np.abs(sv.data)))
                x_out = idx & ((1 << n) - 1)
                f_out = (idx >> n) & 1
                if geq:
                    expected = 1 if x >= threshold else 0
                else:
                    expected = 1 if x < threshold else 0
                assert x_out == x, f"x mutated: {x_out}"
                assert f_out == expected, (
                    f"x={x}, threshold={threshold}, geq={geq}: "
                    f"flag={f_out}, expected={expected}"
                )
    print(f"  OK (5 thresholds x 8 values x 2 directions = 80 cases)")

    print("\nAll qprim.integer tests passed.")

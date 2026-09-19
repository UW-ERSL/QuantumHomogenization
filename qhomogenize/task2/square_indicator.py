"""
Square/box-indicator oracle  O_p : |i_x>|i_y>|0> -> |i_x>|i_y>|p>.

For an axis-aligned square (or rectangular) inclusion occupying

    sx <= i_x < sx + side_x   AND   sy <= i_y < sy + side_y,

element e = i_x + L * i_y is "inclusion" (p_e = 1) iff (i_x, i_y) lies in the
box.  This is the cheapest possible microstructure indicator: two independent
range tests on the integer grid coordinates, ANDed into the flag qubit.  No
multiplication or squaring is required (contrast disk_indicator), so the gate
cost is O(log L) two-qubit gates and the ancilla count is O(1) beyond the
two range-comparison work qubits.

Range test
----------
For a register i in [0, L) and integer bounds [lo, hi), the predicate
lo <= i < hi is computed by two constant comparators:
    ge := (i >= lo),   lt := (i < hi),   in := ge AND lt.
We implement i >= lo and i < hi via constant-comparator circuits built from
the same modular-add trick used elsewhere (add a constant, inspect carry),
but because L is a power of two and the bounds are compile-time constants, we
take the simplest robust route: a reversible comparator that writes the
boolean into a work qubit and is uncomputed after the AND.

Convention
----------
Coordinates (i_x, i_y) are integer grid indices in [0, L), little-endian
registers, matching disk_indicator and microstructure.py (element index
e = i_x + L * i_y).  side_x = side_y = side for a square; the rectangular
case is supported by passing side_x, side_y separately.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def _const_geq(qc, q_in, lo, q_out, q_work):
    """q_out ^= (value(q_in) >= lo), reversibly, using q_work as borrow scratch.

    Implemented as: (i >= lo)  <=>  NOT (i < lo).  We compute (i < lo) by
    constant subtraction lo - i and reading the borrow, then negate.  For
    simplicity and robustness with compile-time lo, we use the comparator
    identity on n-bit unsigned integers:

        i < lo  <=>  high bit of (i + (2^n - lo)) is 0   (i.e. no carry out).

    Rather than hand-roll carry logic, we enumerate via an explicit
    multi-controlled marking over the (compile-time known) set of i values
    satisfying the predicate.  Since the predicate is a contiguous range and
    L is small in the regimes we verify (L <= 16, n <= 4), this is both exact
    and cheap; for larger L a carry-based comparator should replace it.
    """
    raise NotImplementedError  # replaced by _mark_range below


def _mark_range(qc, q_reg, lo, hi, q_flag, n):
    """Flip q_flag for every basis state of q_reg with lo <= value < hi.

    Exact, compile-time construction: for each value v in [lo, hi), apply a
    multi-controlled X onto q_flag with the control pattern equal to the bits
    of v.  Contiguous-range compression is possible but unnecessary at the
    mesh sizes used for verification; the uncompressed form is transparently
    correct and is O((hi-lo) * n) gates.
    """
    for v in range(lo, hi):
        bits = [(v >> j) & 1 for j in range(n)]
        zero_positions = [j for j in range(n) if bits[j] == 0]
        for j in zero_positions:
            qc.x(q_reg[j])
        if n == 1:
            qc.cx(q_reg[0], q_flag)
        else:
            qc.mcx([q_reg[j] for j in range(n)], q_flag)
        for j in zero_positions:
            qc.x(q_reg[j])


def square_indicator_oracle(
    L: int,
    sx: int,
    sy: int,
    side_x: int,
    side_y: int | None = None,
    name: str = "O_p_square",
):
    """Reversible indicator oracle for an axis-aligned box inclusion.

    Parameters
    ----------
    L : int (power of 2)
        Mesh size (elements per side).
    sx, sy : int
        Lower corner (inclusive) of the box, in grid units [0, L).
    side_x : int
        Box width in x.
    side_y : int, optional
        Box height in y; defaults to side_x (square).
    name : str

    Returns
    -------
    qc : QuantumCircuit
    info : dict with keys {qreg_ix, qreg_iy, flag, n_x, n_y, n_qubits}

    Logic
    -----
        in_x := (sx <= i_x < sx + side_x)
        in_y := (sy <= i_y < sy + side_y)
        p    := in_x AND in_y
    Computed into the flag, with in_x, in_y on work qubits that are uncomputed.
    """
    if side_y is None:
        side_y = side_x
    n_x = int(np.ceil(np.log2(L))) if L > 1 else 1
    n_y = n_x
    if 2 ** n_x != L:
        raise NotImplementedError(f"L must be a power of 2; got L={L}")

    q_ix = QuantumRegister(n_x, "ix")
    q_iy = QuantumRegister(n_y, "iy")
    q_flag = QuantumRegister(1, "flag")
    q_inx = QuantumRegister(1, "in_x")
    q_iny = QuantumRegister(1, "in_y")

    qc = QuantumCircuit(q_ix, q_iy, q_flag, q_inx, q_iny, name=name)

    hx_lo, hx_hi = sx, min(sx + side_x, L)
    hy_lo, hy_hi = sy, min(sy + side_y, L)

    # forward: mark in_x, in_y
    _mark_range(qc, q_ix, hx_lo, hx_hi, q_inx[0], n_x)
    _mark_range(qc, q_iy, hy_lo, hy_hi, q_iny[0], n_y)

    # flag := in_x AND in_y
    qc.ccx(q_inx[0], q_iny[0], q_flag[0])

    # uncompute in_x, in_y
    _mark_range(qc, q_iy, hy_lo, hy_hi, q_iny[0], n_y)
    _mark_range(qc, q_ix, hx_lo, hx_hi, q_inx[0], n_x)

    info = dict(
        qreg_ix=q_ix, qreg_iy=q_iy, flag=q_flag[0],
        n_x=n_x, n_y=n_y, n_qubits=qc.num_qubits,
    )
    return qc, info


def square_indicator_array(L: int, sx: int, sy: int, side_x: int, side_y: int | None = None) -> np.ndarray:
    """Classical reference: p[iy, ix] = 1 inside the box, else 0.  (rows=iy, cols=ix)"""
    if side_y is None:
        side_y = side_x
    p = np.zeros((L, L), dtype=int)
    for ix in range(sx, min(sx + side_x, L)):
        for iy in range(sy, min(sy + side_y, L)):
            p[iy, ix] = 1
    return p


# ---------------------------------------------------------------------------
# Quick test: exhaustive over all (i_x, i_y) at L=4 and L=8
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    for L, (sx, sy, side) in [(4, (1, 1, 2)), (8, (2, 3, 3))]:
        print(f"--- square_indicator L={L}, corner=({sx},{sy}), side={side} ---")
        p_ref = square_indicator_array(L, sx, sy, side)
        qc, info = square_indicator_oracle(L, sx, sy, side)
        n_x, n_y = info["n_x"], info["n_y"]
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
                tqc_c = transpile(tqc.decompose(reps=8), sim, optimization_level=0)
                counts = sim.run(tqc_c, shots=1).result().get_counts()
                outcome = next(iter(counts)).replace(" ", "")[::-1]
                ixo = int(outcome[0:n_x][::-1], 2)
                iyo = int(outcome[n_x:n_x + n_y][::-1], 2)
                flag = int(outcome[n_x + n_y])
                anc = outcome[n_x + n_y + 1:]
                anc_nz = any(c == "1" for c in anc)
                exp = int(p_ref[iy, ix])
                if (ixo, iyo, flag, anc_nz) != (ix, iy, exp, False):
                    failures += 1
                    print(f"  FAIL ix={ix} iy={iy}: flag={flag} exp={exp} anc_nz={anc_nz}")
        print(f"  {'ALL %d pass' % (L*L) if failures==0 else '%d FAILURES'%failures}")

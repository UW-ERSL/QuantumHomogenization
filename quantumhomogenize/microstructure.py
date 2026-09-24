"""microstructure -- centred square inclusions at any void fraction, with the
oracle cost that comes with them.

The block-encoding paper uses the dyadic square of side N/2^k, for which
membership is a test on the top k+1 bits: 4k CNOT and one multi-controlled Z,
flat in m. That family reaches only v_f = 4^-k, so it cannot thin a ligament
and cannot sweep the scope boundary.

A centred square of side N - 2t sweeps v_f = ((N-2t)/N)^2 continuously while
keeping the oracle cheap. The window t <= e < N - t is a union of maximal
dyadic blocks, each a prefix of the bit string, so it is one multi-controlled X
per block. The number of blocks is set by the binary weight of t relative to N,
not by m, so a family with t proportional to N costs the same at every mesh:

    v_f = 0.5625 costs 150 CX at m = 4, 5, 6 and 7 alike,
    v_f = 0.2500 costs  54 CX at every mesh.

chi is the inclusion indicator, 1 inside, so the compliant phase is the
inclusion and the stiff phase is the connected matrix.

    from .microstructure import square, square_t, oracle_cost
    chi = square(m=5, vf=0.5625)
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from functools import lru_cache
from qiskit.circuit.library import XGate

BASIS = ["cx", "u"]


# ==========================================================================
# 1. the geometry
# ==========================================================================
def square_t(m: int, t: int) -> np.ndarray:
    """Centred square of side N - 2t. chi[ex, ey], 1 in the inclusion."""
    N = 2 ** m
    c = np.zeros((N, N))
    c[t:N - t, t:N - t] = 1.0
    return c


def square(m: int, vf: float) -> np.ndarray:
    """Centred square at the achievable volume fraction nearest vf."""
    N = 2 ** m
    t = int(round((N - N * np.sqrt(vf)) / 2))
    return square_t(m, min(max(t, 0), N // 2 - 1))


def ligament(m: int, t: int) -> float:
    """Wall thickness between periodic images, in elements."""
    return 2.0 * t


# ==========================================================================
# 2. the oracle
# ==========================================================================
def dyadic_cover(m: int, lo: int, hi: int) -> list[tuple[int, int]]:
    """Maximal dyadic blocks tiling [lo, hi), as (prefix value, prefix bits)."""
    out, e = [], lo
    while e < hi:
        size = 1
        while e % (2 * size) == 0 and e + 2 * size <= hi:
            size *= 2
        k = int(np.log2(size))
        out.append((e >> k, m - k))
        e += size
    return out


def _mark(qc, bits, blocks, target) -> None:
    for val, plen in blocks:
        ctrl = bits[len(bits) - plen:]
        flips = [q for i, q in enumerate(ctrl)
                 if not (val >> (plen - 1 - i)) & 1]
        for q in flips:
            qc.x(q)
        qc.x(target) if plen == 0 else qc.append(
            XGate().control(plen), ctrl + [target])
        for q in flips:
            qc.x(q)


def oracle(m: int, t: int) -> QuantumCircuit:
    """R_chi for the centred square, as two window marks and one Toffoli."""
    blocks = dyadic_cover(m, t, 2 ** m - t)
    qc = QuantumCircuit(2 * m + 3)
    x, y = list(range(m)), list(range(m, 2 * m))
    wx, wy, flag = 2 * m, 2 * m + 1, 2 * m + 2
    _mark(qc, x, blocks, wx)
    _mark(qc, y, blocks, wy)
    qc.ccx(wx, wy, flag)
    _mark(qc, y, blocks, wy)
    _mark(qc, x, blocks, wx)
    return qc


@lru_cache(maxsize=None)
def oracle_cost(m: int, t: int) -> tuple[int, int]:
    """(dyadic blocks, transpiled two-qubit gates)."""
    qc = transpile(oracle(m, t), basis_gates=BASIS, optimization_level=3)
    return len(dyadic_cover(m, t, 2 ** m - t)), qc.count_ops().get("cx", 0)


# ==========================================================================
if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    print("Centred square: the oracle is flat in the mesh at fixed v_f")
    print(f"{'v_f':>8}" + "".join(f"{'m=' + str(m):>10}" for m in (4, 5, 6, 7)))
    for vf in (0.25, 0.3906, 0.5625, 0.7656):
        row = f"{vf:>8.4f}"
        for m in (4, 5, 6, 7):
            N = 2 ** m
            t = int(round((N - N * np.sqrt(vf)) / 2))
            nb, cx = oracle_cost(m, t)
            row += f"{str(cx) + ' CX':>10}"
        print(row)

    print("\nThe family the scope sweep uses, at m = 6")
    print(f"{'t':>4} {'side':>5} {'v_f':>8} {'ligament':>9} {'blocks':>7} "
          f"{'CX':>5}")
    for t in (4, 8, 12, 16, 20, 24):
        nb, cx = oracle_cost(6, t)
        print(f"{t:>4} {64 - 2 * t:>5} {((64 - 2 * t) / 64) ** 2:>8.4f} "
              f"{ligament(6, t):>9.0f} {nb:>7} {cx:>5}")
    print("---------------------------------------------------------------")

import numpy as np
from qiskit import transpile
from qhomogenize.fea.full_K import assemble_K
from qhomogenize.task2.test_K_uniform import build_UK_uniform
from qhomogenize.task3.baseline_shende import build_K_baseline

def cx_depth(be):
    t = transpile(be.circuit, basis_gates=["u","cx"], optimization_level=3)
    ops = t.count_ops()
    return ops.get("cx", 0), t.depth(), be.circuit.num_qubits

for L in [2, 4, 8, 16]:
    beK, (beA, beDK, beAT) = build_UK_uniform(L)
    s_cx, s_d, s_q = cx_depth(beK)
    beB = build_K_baseline(L)
    b_cx, b_d, b_q = cx_depth(beB)
    print(f"L={L} N={2*L*L} struct_cx={s_cx} struct_alpha={beK.alpha:.4f} "
          f"base_cx={b_cx} base_alpha={beB.alpha:.4f} speedup={b_cx/s_cx:.1f}x")
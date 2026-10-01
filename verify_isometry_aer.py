"""Independent check: run the exported U_V circuit in Qiskit Aer (MPS) on every basis input at N = 4."""
import time, numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from quantumhomogenize.isometry_circuit import IsometryCircuit

c = IsometryCircuit(2, nu=0.35, F=10)
Q, _ = c.simulate()
nw = c.n_wires; L = list(range(nw, nw + 4))          # local qubits, b0 first
gl = c.local_gates()
sim = AerSimulator(method="matrix_product_state", n_qubits=nw + 4, max_memory_mb=2**55)
print("qubits:", nw + 4, " arithmetic gates:", 2 * c.n_arith)

def build(k1, k2, init):
    qc = QuantumCircuit(nw + 4)
    for j in range(c.n):
        if (k1 >> j) & 1: qc.x(c.k1[j])
        if (k2 >> j) & 1: qc.x(c.k2[j])
    if init == 1: qc.x(L[0])
    if init == 2: qc.h(L[0]); qc.s(L[0])
    def emit(gs):
        for g in gs:
            if g[0] == "x": qc.x(g[1])
            elif g[0] == "cx": qc.cx(g[1], g[2])
            else: qc.ccx(g[1], g[2], g[3])
    emit(c.c.gates[:c.n_arith])
    for M, ctrl in gl:
        U = UnitaryGate(M)
        if ctrl:
            qc.append(U.control(len(ctrl), ctrl_state="".join(str(v) for _, v in ctrl[::-1])),
                      [w for w, _ in ctrl] + L)
        else:
            qc.append(U, L)
    for j in range(c.n):
        qc.p(np.pi * 2 ** j / c.N, c.k1[j]); qc.p(np.pi * 2 ** j / c.N, c.k2[j])
    emit(c.c.gates[c.n_arith:])
    return qc

worst = 0; t = time.time()
for k1 in range(c.N):
    for k2 in range(c.N):
        for init in (0, 1, 2):
            qc = transpile(build(k1, k2, init), basis_gates=["cx", "u", "x", "ccx", "p", "h", "s"], optimization_level=0); qc.save_density_matrix(L)
            rho = np.asarray(sim.run(qc).result().data()["density_matrix"])
            v = [np.eye(16)[0], np.eye(16)[1], (np.eye(16)[0] + 1j * np.eye(16)[1]) / np.sqrt(2)][init]; want = Q[k1 * c.N + k2] @ v
            worst = max(worst, np.abs(rho - np.outer(want, want.conj())).max())
print(f"Aer MPS vs bit simulator, all k at N=4, 3 inputs each: max |rho - psi psi^dag| = {worst:.2e}  ({time.time()-t:.0f}s)")

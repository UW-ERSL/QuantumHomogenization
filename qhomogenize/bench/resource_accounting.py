"""
Resource accounting for block-encoding circuits.

Transpiles a circuit (or BlockEncoding) to the (u, cx) basis at a fixed
optimization level and reports CX count, single-qubit-gate count, depth,
and qubit count.  CX count is the primary cost metric per the POC plan;
extension to Clifford+T T-count is left as a follow-on if needed.

Adapted from `estimateCircuitGates` in Chapter08_QuantumGates_functions.py,
with the basis explicitly pinned to ['u', 'cx'] (rather than relying on
AerSimulator's defaults) and consistent decomposition depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from qiskit import QuantumCircuit, transpile

from qhomogenize.bencode.types import BlockEncoding


# Fixed transpilation target for the entire project.
BASIS_GATES = ["u", "cx"]
DEFAULT_OPT_LEVEL = 3


@dataclass
class Resources:
    """Resource counts for a transpiled circuit."""
    n_qubits: int
    cx_count: int
    single_qubit_count: int
    total_gate_count: int
    depth: int
    name: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "cx_count": self.cx_count,
            "single_qubit_count": self.single_qubit_count,
            "total_gate_count": self.total_gate_count,
            "depth": self.depth,
        }

    def summary(self) -> str:
        return (
            f"{self.name}: n_qubits={self.n_qubits}, "
            f"CX={self.cx_count}, U={self.single_qubit_count}, "
            f"total={self.total_gate_count}, depth={self.depth}"
        )


def count_resources(
    obj: Union[QuantumCircuit, BlockEncoding],
    basis: list[str] = BASIS_GATES,
    opt_level: int = DEFAULT_OPT_LEVEL,
    name: str = "",
) -> Resources:
    """Transpile a circuit to the (u, cx) basis and return resource counts.

    Parameters
    ----------
    obj : QuantumCircuit | BlockEncoding
        Circuit (or BlockEncoding wrapping one) to count.
    basis : list[str], default ['u', 'cx']
        Target basis gates for transpilation.
    opt_level : int, default 3
        Qiskit optimization level (0..3).
    name : str
        Optional name for the returned Resources.

    Returns
    -------
    Resources
    """
    if isinstance(obj, BlockEncoding):
        qc = obj.circuit
        if not name:
            name = obj.name
    else:
        qc = obj

    # decompose first so high-level gates (StatePreparation, multi-controlled,
    # IntegerComparator, adders) get reduced before transpile.  reps=10 is
    # generous; the cost is just compile time.
    qc_decomposed = qc.decompose(reps=10)
    qc_transpiled = transpile(qc_decomposed, basis_gates=basis, optimization_level=opt_level)

    counts = qc_transpiled.count_ops()
    cx = counts.get("cx", 0)
    total = sum(counts.values())
    single = total - cx
    return Resources(
        n_qubits=qc_transpiled.num_qubits,
        cx_count=cx,
        single_qubit_count=single,
        total_gate_count=total,
        depth=qc_transpiled.depth(),
        name=name,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Two-qubit Bell-state preparer: 1 H + 1 CX.  Depth 2.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    res = count_resources(qc, name="bell")
    print(res.summary())
    assert res.cx_count == 1, f"expected 1 CX, got {res.cx_count}"
    # H decomposes to a single u3-class gate in the 'u' basis.
    assert res.single_qubit_count == 1, f"expected 1 single-qubit gate, got {res.single_qubit_count}"
    assert res.depth == 2
    print("OK")

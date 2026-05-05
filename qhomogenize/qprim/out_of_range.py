"""
Sünderhauf's out-of-range oracle O_rg.

Flags ancilla |1> when the (row, sparsity-index) pair would correspond to an
out-of-pattern matrix entry (i.e., a position the sparsity oracle says is zero).

For dense-rectangular matrices this oracle is trivial (always |0>); for
sparse matrices like A and D_p, it flags positions where the closed-form
sparsity-pattern arithmetic produces an invalid (out-of-range) index.

For the assembly operator A:
- Row sparsity S_r = 1; every row has exactly one entry; out-of-range is
  never triggered.
- Column sparsity S_c = 4 in 2D; for each global DOF j, exactly 4 elements
  share the corresponding node.  With our padded sparsity register of size
  S_c = 4 (power of 2), no padding is needed and the oracle is trivial.

For D_p with binary mask:
- Within a solid block, sparsity is dense (8 entries per row/column inside
  the block).  Within a void block, sparsity is zero -- but D_p's structure
  makes this irrelevant since the indicator already gates the entries.

TODO: For non-power-of-2 sparsity (e.g., 3D where S_c = 8 is fine, but other
sparsity counts may need padding), implement a comparator-based out-of-range
flag.
"""

from __future__ import annotations

from qiskit import QuantumCircuit


def trivial_out_of_range(n_qubits: int, name: str = "O_rg") -> QuantumCircuit:
    """A trivial out-of-range oracle (identity on all qubits).

    Used when the sparsity register size is exactly a power of 2 matching
    the actual sparsity, so no out-of-range case can occur.
    """
    qc = QuantumCircuit(n_qubits, name=name)
    return qc

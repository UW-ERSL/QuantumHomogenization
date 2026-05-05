"""
Sünderhauf-style data oracle.

Given a small set of D distinct values {v_0, ..., v_{D-1}} and an addressing
scheme for the matrix entries, the data oracle implements

    O_data : |d>|0>_amp -> |d> ⊗ ( v_d / ||v||_max  |0>_amp + sqrt(1 - ...) |1>_amp ),

where the "amp" register carries the value as a controlled rotation amplitude
on a single qubit.  This is the standard Sünderhauf encoding (cf. eq. (8) in
Sünderhauf, Campbell, Camps 2024).

Cost: O(D) two-qubit gates (multiplexed-rotation construction).  D is the
number of distinct values; for our applications D = O(1) (mesh-independent),
so the oracle is mesh-independent in cost.

The oracle takes a "data index" register of size ceil(log2 D) and an amplitude
qubit.  It reads the data index, writes the corresponding amplitude.

References
----------
Sünderhauf, Campbell, Camps, "Block-encoding structured matrices for data
input in quantum computing," Quantum 8, 1226 (2024), section 2.1.
Multiplexed-rotation construction: Möttönen et al., quant-ph/0407010.

TODO: Implement using qiskit.circuit.library.UCRYGate (uniformly-controlled
Y-rotation) which is exactly the multiplexed-rotation construction.  Each
amplitude v_d / ||v||_max maps to a rotation angle theta_d = 2 arcsin(v_d / ||v||_max).
"""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit


def data_oracle(values: Sequence[float], name: str = "O_data") -> tuple[QuantumCircuit, dict]:
    """Build a Sünderhauf data oracle for the given distinct values.

    Parameters
    ----------
    values : sequence of float
        The D distinct values to load.  Will be normalized by ||values||_max.
    name : str

    Returns
    -------
    qc : QuantumCircuit on ceil(log2 D) + 1 qubits
        The first ceil(log2 D) qubits are the data-index register; the last
        qubit is the amplitude qubit.
    info : dict
        {'D': int, 'norm': float, 'n_index': int, 'amp_qubit': index}
    """
    raise NotImplementedError("TODO: implement via UCRYGate multiplexed rotations")

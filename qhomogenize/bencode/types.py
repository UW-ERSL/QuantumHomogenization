"""
Core types for block encodings.

A `BlockEncoding` bundles a Qiskit circuit with the metadata needed to
interpret it as a block encoding of a target matrix:

    (<0|^a ⊗ I) U (|0>^a ⊗ I) = A / α,

where U is a unitary on (a + n) qubits, the first `a` qubits are the
ancilla/flag register, the last `n` qubits are the system register
encoding the matrix indices, and α >= ||A|| is the subnormalization.

This module is the central data type for the project; every Sünderhauf-
style construction (data oracle, indicator, assembly, composition) returns
a `BlockEncoding`.

Convention
----------
- System qubits = LEAST significant in Qiskit's little-endian ordering
  (declared first in QuantumCircuit, lowest indices).
- Ancilla qubits = MOST significant (declared second, highest indices).
- This matches the LCU_Ax convention in Chapter13_MatrixEncoding_functions.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit


@dataclass
class BlockEncoding:
    """A unitary circuit U that block-encodes a matrix A with subnormalization α.

    Parameters
    ----------
    circuit : QuantumCircuit
        The unitary circuit on (n_system + n_ancilla) qubits.  System qubits
        are declared first (low indices); ancilla qubits second (high indices).
    n_system : int
        Number of system qubits; matrix dimension is N = 2**n_system.
    n_ancilla : int
        Number of ancilla / flag qubits.
    alpha : float
        Subnormalization.  Must satisfy α >= ||A||_2 for the encoding to be valid.
    name : str
        Short identifier for debugging / logging (e.g. "U_A", "U_Dp_lambda").
    target_shape : tuple[int, int], optional
        Shape of the underlying matrix A.  Default is (2**n_system, 2**n_system),
        but rectangular matrices (e.g. the assembly operator A which maps
        N_global -> N_local) need an explicit shape.
    """

    circuit: QuantumCircuit
    n_system: int
    n_ancilla: int
    alpha: float
    name: str = "U"
    target_shape: Optional[tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.target_shape is None:
            d = 2 ** self.n_system
            self.target_shape = (d, d)
        if self.alpha <= 0:
            raise ValueError(f"subnormalization alpha must be positive, got {self.alpha}")
        if self.circuit.num_qubits != self.n_system + self.n_ancilla:
            raise ValueError(
                f"{self.name}: circuit has {self.circuit.num_qubits} qubits, "
                f"expected n_system + n_ancilla = {self.n_system + self.n_ancilla}"
            )

    @property
    def n_qubits(self) -> int:
        return self.n_system + self.n_ancilla

    def extract_matrix(self) -> np.ndarray:
        """Extract A/α from the simulated unitary by post-selecting on ancilla=|0>.

        Returns A_normalized of shape (2**n_system, 2**n_system); for rectangular
        targets the user is responsible for slicing to `target_shape`.

        For verification at small sizes only; cost is O(4**n_qubits).
        """
        from qiskit.quantum_info import Operator

        U = Operator(self.circuit).data  # shape (2**n_qubits, 2**n_qubits)
        d = 2 ** self.n_system
        # System = least significant => ancilla=|0> => take the d x d top-left block
        # (this matches the LCU_Ax convention used elsewhere in the project).
        A_normalized = U[:d, :d]
        return A_normalized

    def verify_against(
        self,
        A_reference: np.ndarray,
        atol: float = 1e-10,
        verbose: bool = False,
    ) -> tuple[bool, float]:
        """Verify that the extracted block matches A_reference / α to tolerance.

        Returns (passed, max_abs_error).
        """
        if A_reference.shape != self.target_shape:
            raise ValueError(
                f"{self.name}: reference shape {A_reference.shape} != "
                f"target_shape {self.target_shape}"
            )
        A_extracted_full = self.extract_matrix()
        # Slice to target shape (top-left for rectangular A).
        m, n = self.target_shape
        A_extracted = A_extracted_full[:m, :n]
        expected = A_reference / self.alpha
        err = np.max(np.abs(A_extracted - expected))
        passed = bool(err < atol)
        if verbose:
            print(f"  {self.name}: max abs error = {err:.3e}, passed = {passed}")
        return passed, float(err)

    def summary(self) -> str:
        return (
            f"BlockEncoding(name={self.name!r}, n_system={self.n_system}, "
            f"n_ancilla={self.n_ancilla}, alpha={self.alpha:.6g}, "
            f"target_shape={self.target_shape})"
        )

    def conjugate_transpose(self, name: Optional[str] = None) -> "BlockEncoding":
        """Return a block encoding of A^*T (conjugate transpose of A).

        Construction: invert the circuit.  Inverting U gives U^dagger, and the
        top-left block of U^dagger is (U[:N, :N])^dagger = M^dagger.  For real
        A this equals A^T.

        Subnormalization is unchanged.
        """
        new_name = name if name is not None else f"({self.name})_dag"
        new_target_shape = None
        if self.target_shape is not None:
            new_target_shape = (self.target_shape[1], self.target_shape[0])
        return BlockEncoding(
            circuit=self.circuit.inverse(),
            n_system=self.n_system,
            n_ancilla=self.n_ancilla,
            alpha=self.alpha,
            name=new_name,
            target_shape=new_target_shape,
        )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a trivial block encoding of the 2x2 identity with α=1: just the
    # identity circuit on 1 system qubit, no ancilla.
    qc = QuantumCircuit(1)
    be = BlockEncoding(circuit=qc, n_system=1, n_ancilla=0, alpha=1.0, name="U_I")
    print(be.summary())
    A_ref = np.eye(2)
    passed, err = be.verify_against(A_ref, verbose=True)
    assert passed, f"trivial identity BE failed verification (err={err})"
    print("OK")

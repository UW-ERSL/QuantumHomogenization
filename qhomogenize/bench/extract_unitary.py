"""
Full-unitary extraction at small mesh sizes.

For L = 4 and L = 8, the block-encoding circuit (system + ancilla qubits)
fits in memory for full-unitary simulation via `qiskit.quantum_info.Operator`.
This routine extracts the (post-selected on ancilla=|0>) top-left block of
the full unitary, which equals A / alpha for a valid block encoding.

Cost: O(4^n_qubits).  Used at L=4, L=8 only.

This is a thin wrapper around `BlockEncoding.extract_matrix()` and
`BlockEncoding.verify_against()` (both already in bencode.types), provided
here for symmetry with `extract_amplitude`.
"""

from __future__ import annotations

import numpy as np

from qhomogenize.bencode.types import BlockEncoding


def extract_unitary_block(be: BlockEncoding) -> np.ndarray:
    """Return the (2^n_system x 2^n_system) post-selected block, equal to A / alpha."""
    return be.extract_matrix()


def verify_at_small_L(be: BlockEncoding, A_reference: np.ndarray, atol: float = 1e-10) -> tuple[bool, float]:
    return be.verify_against(A_reference, atol=atol)

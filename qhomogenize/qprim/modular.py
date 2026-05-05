"""
Index encoding/decoding helpers.

Reversible primitives for converting between linear indices and tuple indices
used throughout the project.  For example:
- Element index e in [0, L^2) <-> (i_x, i_y) in [0, L) x [0, L)
- Local DOF ell in [0, 8) <-> (corner, dim) in [0, 4) x [0, 2)

When L is a power of 2, these decompositions are just bit-splittings of the
integer register; no actual quantum gates are required, just a relabeling.
The helpers below return register slices.

For non-power-of-2 L the situation is more involved (would require an Euclidean
division subroutine), but the POC restricts to L in {4, 8, 16}, all powers of 2.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumRegister


def split_element_index(e_reg: QuantumRegister, L: int) -> tuple[QuantumRegister, QuantumRegister]:
    """View e_reg as the concatenation (i_y || i_x) where each is log2(L) bits.

    Currently this is just register slicing; for L a power of 2, e = i_x + L * i_y.

    Returns
    -------
    ix_reg, iy_reg : the low / high halves of e_reg as separate aliases.
        Both are NEW QuantumRegister objects pointing at slices of e_reg via Qubit references.
    """
    n_xy = int(np.log2(L))
    if 2 ** n_xy != L:
        raise NotImplementedError(f"L must be a power of 2; got L={L}")
    if e_reg.size != 2 * n_xy:
        raise ValueError(f"e_reg size {e_reg.size} != 2 * log2(L) = {2 * n_xy}")
    # In Qiskit it's not idiomatic to construct a new QuantumRegister sharing the same qubits;
    # callers should pass slices [e_reg[0:n_xy]] (low = i_x) and [e_reg[n_xy:2*n_xy]] (high = i_y)
    # directly to gates as qubit lists.  This helper is a placeholder.
    raise NotImplementedError("Use [e_reg[0:n_xy]] and [e_reg[n_xy:2*n_xy]] directly as qubit lists.")

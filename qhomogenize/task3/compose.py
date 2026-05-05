"""
Task 3: compose K = A^T D_p A from Tasks 1 and 2.

Implements the LCU-first ordering recommended in the POC plan:

    1. LCU-combine D_p^lambda and D_p^mu into U_{D_p}.
    2. Apply triple product (U_{A^T}, U_{D_p}, U_A) to obtain U_K.

This ordering uses one fewer set of compositional ancillas and reuses U_A
(rather than building two copies, one per Lamé layer).

Total subnormalization (algebraically equal in either ordering):
    alpha_K = alpha_A^2 * (lambda * alpha_{D_p^lambda} + mu * alpha_{D_p^mu})
"""

from __future__ import annotations

from qhomogenize.bencode.types import BlockEncoding


def build_UK(L: int, Cx: int, Cy: int, R2: int, lam: float, mu: float) -> BlockEncoding:
    """Build U_K via LCU-first composition.

    TODO.
    """
    raise NotImplementedError("TODO: assemble via compose_lcu then compose_triple")

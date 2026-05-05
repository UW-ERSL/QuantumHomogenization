"""
LCU sum of block encodings.

Given block encodings U_1, ..., U_K of matrices A_1, ..., A_K with
subnormalizations alpha_1, ..., alpha_K, and weights w_1, ..., w_K, the
LCU-of-block-encodings composition yields a block encoding U of

    A = sum_k w_k A_k

with subnormalization

    alpha = sum_k |w_k| alpha_k

and ancilla count = max_k a_k + ceil(log2 K) (the K-side PREP register).

Implementation: standard PREP-SELECT-UNPREP with each A_k applied controlled
on the K-side ancilla index k.  This is essentially a generalization of
LCU_Ax (Chapter13_MatrixEncoding_functions.py), where the controlled
operation is the full block-encoding circuit rather than a Pauli gate.

TODO
----
- Adapt the LCU_Ax skeleton to take BlockEncoding inputs and produce a
  BlockEncoding output.  The PREP register loads the |w_k| amplitudes via
  StatePreparation; the SELECT applies each U_k controlled on |k>.
"""

from __future__ import annotations

from typing import Sequence

from qhomogenize.bencode.types import BlockEncoding


def compose_lcu(
    components: Sequence[BlockEncoding],
    weights: Sequence[float],
    name: str = "U_LCU",
) -> BlockEncoding:
    """Compose a weighted LCU of K block encodings.

    Parameters
    ----------
    components : sequence of BlockEncoding
        The K block encodings to sum.  All must encode same-shape matrices.
    weights : sequence of float
        K weights (signed).
    name : str

    Returns
    -------
    BlockEncoding of sum_k weights[k] * components[k].A with
    alpha = sum_k |weights[k]| * components[k].alpha and
    ancilla = max_k components[k].n_ancilla + ceil(log2 K).

    TODO: implement.
    """
    raise NotImplementedError("TODO: adapt PREP-SELECT-UNPREP from LCU_Ax")

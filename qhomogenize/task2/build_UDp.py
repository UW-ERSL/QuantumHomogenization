"""
Task 2 orchestration: build U_{D_p^lambda} and U_{D_p^mu}.

D_p = D_K . M_p with D_K = I_{L^2} ⊗ K_e and M_p = diag(p_e) ⊗ I_8.

Strategy
--------
1. Build U_{D_K^layer} via Sünderhauf for the block-diagonal element template.
   (Uses element_data_oracle for the small set of distinct values.)
2. Build U_{M_p} via the disk indicator (already implemented in
   task2.disk_indicator) packaged as a 1-qubit-amplitude block encoding.
3. Compose via product-of-block-encodings (bencode.compose_product).

Alternative: build a single combined oracle that gates the K_e data values
by p_e in one shot.  Slightly more efficient but harder to validate
incrementally.  Use the staged approach for now.
"""

from __future__ import annotations

from qhomogenize.bencode.types import BlockEncoding


def build_UDp_lambda(L: int, Cx: int, Cy: int, R2: int) -> BlockEncoding:
    """Block encoding of D_p^lambda = M_p . (I_{L^2} ⊗ K_e^lambda)."""
    raise NotImplementedError("TODO")


def build_UDp_mu(L: int, Cx: int, Cy: int, R2: int) -> BlockEncoding:
    raise NotImplementedError("TODO")

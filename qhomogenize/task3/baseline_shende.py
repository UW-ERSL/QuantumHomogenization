"""
Baseline block encoding of the assembled stiffness K via Pauli LCU.

This is the natural counterfactual to the structured composition
U_K = U_{A^T} U_{D_K} U_A: build K classically, decompose into Paulis, and
block-encode via PREP-SELECT-UNPREP.  PREP uses Qiskit's StatePreparation,
which uses Shende-Bullock-Markov synthesis under the hood (the "Shende baseline"
referred to in the POC plan).

For sparse K (our K is), the number of nonzero Paulis is smaller than N^2,
but still grows as a power of N.  We report both the actual nonzero Pauli
count and the resulting CX count.

Comparison metric: total CX count (two-qubit gate count after decomposing
into the {u, cx} basis).
"""

from __future__ import annotations

import numpy as np

from qhomogenize.bencode.types import BlockEncoding
from qhomogenize.bencode.pauli_lcu import pauli_block_encoding
from qhomogenize.fea.full_K import assemble_K


def build_K_baseline(
    L: int,
    p: np.ndarray | None = None,
    lam: float = 1.0,
    mu: float = 1.0,
    phi_deg: float = 90.0,
    name: str | None = None,
    use_padded: bool = False,
) -> BlockEncoding:
    """Build the Pauli-LCU block encoding of the assembled K.

    Parameters
    ----------
    L : int
    p : ndarray of shape (L^2,) or None
        Element-presence indicator p_e in {0, 1}.  Defaults to np.ones(L*L).
    lam, mu : float
        Lame parameters of the solid phase.
    phi_deg : float
        Cell-wall angle.
    name : str
    use_padded : bool
        If True, encode the 8L^2 x 8L^2 zero-padded K (matching the dimension
        of the structured U_K).  If False (default), encode the active
        2L^2 x 2L^2 block directly.

    Returns
    -------
    BlockEncoding
    """
    if p is None:
        p = np.ones(L * L)
    if name is None:
        name = f"U_K_baseline_L{L}"

    K = assemble_K(L, p=p, lam=lam, mu=mu, phi_deg=phi_deg).toarray()

    if use_padded:
        n_active = 2 * L * L
        n_padded = 8 * L * L
        K_pad = np.zeros((n_padded, n_padded), dtype=K.dtype)
        K_pad[:n_active, :n_active] = K
        K = K_pad

    return pauli_block_encoding(K, name=name)


if __name__ == "__main__":
    print("--- Baseline block encoding of K_uniform ---")
    for L in [2, 4]:
        be = build_K_baseline(L)
        print(f"  L={L}: {be.summary()}")
        K_ref = assemble_K(L, p=np.ones(L*L), lam=1.0, mu=1.0).toarray()
        passed, err = be.verify_against(K_ref, atol=1e-10, verbose=True)
        assert passed, f"baseline at L={L} failed (err={err})"

    print("\nAll baseline tests passed.")

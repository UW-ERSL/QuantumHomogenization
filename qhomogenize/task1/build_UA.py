"""
Build U_A: the block encoding of the FE assembly operator A using the
Sünderhauf base-scheme builder and the assembly oracles from
qhomogenize.task1.assembly_oracles.

This brings together:
- bencode.builder.build_block_encoding (the Sünderhauf scheme)
- task1.assembly_oracles (O_c, O_r, O_rg for A)
- a trivial O_data (D = 1 with value 1, so the rotation is the identity --
  but we still wrap it as a 1-qubit identity circuit so the builder is happy)

The matrix is padded to N x N = 8L^2 x 8L^2 with the actual A in the first
2L^2 columns; the remaining columns are zero.

Subnormalization
----------------
The builder uses alpha = 2^n_s * ||A||_max = 4 * 1 = 4.

This corresponds to the symmetric padded scheme where every row is padded to
S = 4 entries (only one of which is real, the others being out-of-range zeros)
and every column has S = 4 entries (all real, since each global DOF is shared
by 4 incident elements).  The resulting block encoding represents A / 4.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from qhomogenize.bencode.builder import BlockEncodingSpec, build_block_encoding, make_data_oracle
from qhomogenize.bencode.types import BlockEncoding
from qhomogenize.task1.assembly_oracles import (
    assembly_column_oracle,
    assembly_row_oracle,
    assembly_out_of_range_oracle,
    bit_layout,
)


def build_UA(L: int, name: str = "U_A") -> BlockEncoding:
    """Build U_A: block encoding of the (square-padded) FE assembly operator.

    Parameters
    ----------
    L : int
        Mesh resolution (must be a power of 2).

    Returns
    -------
    be : BlockEncoding
        n_system = 3 + 2 log_2 L, n_ancilla = 2 + 1 + 1 = 4
        (s_register + data + del).
        target_shape = (8 L^2, 8 L^2).
        alpha = 4.
    """
    layout = bit_layout(L)
    n_log = layout["n_log"]
    n_idx = 3 + 2 * n_log    # log_2(8 L^2)
    n_s = 2                  # log_2(S = 4)
    n_d = 0                  # D = 1
    A_max = 1.0

    O_c = assembly_column_oracle(L)
    O_r = assembly_row_oracle(L)
    O_rg = assembly_out_of_range_oracle(L)
    O_data, _ = make_data_oracle([1.0], A_max=1.0)

    spec = BlockEncodingSpec(
        n_idx=n_idx,
        n_s=n_s,
        n_d=n_d,
        A_max=A_max,
        O_c=O_c,
        O_r=O_r,
        O_data=O_data,
        O_rg=O_rg,
        name=name,
    )
    return build_block_encoding(spec)


# ---------------------------------------------------------------------------
# Verification against numpy reference (square-padded A)
# ---------------------------------------------------------------------------

def _padded_A_reference(L: int) -> np.ndarray:
    """Return the 8L^2 x 8L^2 zero-padded version of A."""
    from qhomogenize.fea.assembly_pbc import assembly_matrix
    A = assembly_matrix(L).toarray()  # shape (8 L^2, 2 L^2)
    n_rows = 8 * L * L
    n_cols_padded = 8 * L * L
    A_padded = np.zeros((n_rows, n_cols_padded), dtype=A.dtype)
    A_padded[:, :A.shape[1]] = A
    return A_padded


if __name__ == "__main__":
    for L in [2, 4]:
        print(f"--- L = {L} ---")
        be = build_UA(L)
        print(f"  {be.summary()}")

        A_pad = _padded_A_reference(L)
        passed, err = be.verify_against(A_pad, atol=1e-10, verbose=True)
        if not passed:
            # On failure, print a small diagnostic block
            U = be.extract_matrix()
            target = A_pad / be.alpha
            diff = U - target
            print(f"  max |U - A/alpha| = {np.max(np.abs(diff)):.4e}")
            mismatches = np.argwhere(np.abs(diff) > 1e-8)
            if len(mismatches) > 0:
                print(f"  number of mismatching entries: {len(mismatches)}")
                for ii, jj in mismatches[:5]:
                    print(f"    ({ii}, {jj}): U={U[ii,jj]:.4f}, target={target[ii,jj]:.4f}")

    print("\nDONE.")

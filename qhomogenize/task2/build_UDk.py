"""
Build U_{D_K}: block encoding of the block-diagonal element-stiffness operator

    D_K = I_{L^2}  (x)  K_e

acting on the 8 L^2-dimensional space of local DOFs.

Because every block is the same K_e, the block encoding has a clean tensor-
product structure: U_{D_K} = I  (x)  U_{K_e}, which means we can simply embed
the small (8 x 8) U_{K_e} circuit inside a larger system register, leaving
the L^2-dimensional element register as a passive identity.

Convention
----------
System register layout (low -> high):
    bits 0..2                : local DOF index ell in [0, 8)
    bits 3..(2 + 2 log_2 L)  : element index e = i_x + L * i_y in [0, L^2)

So D_K[j, j'] = K_e[ell, ell'] * delta(e, e'), which in tensor product form is
I_{L^2} (x) K_e under qiskit's "leftmost factor = high qubits" convention.

Subnormalization
----------------
alpha_{D_K} = alpha_{K_e} (passing identity on the element register doesn't
change subnorm).

For the standard element template (a = b = 0.5, phi = 90 deg) and lambda = mu = 1,
alpha_{K_e} ~ 4.67 (8 Paulis).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from qhomogenize.bencode.types import BlockEncoding
from qhomogenize.bencode.pauli_lcu import pauli_block_encoding
from qhomogenize.fea.element_template import element_template


def build_UKe(
    a: float = 0.5,
    b: float = 0.5,
    phi_deg: float = 90.0,
    lam: float = 1.0,
    mu: float = 1.0,
    name: str = "U_Ke",
) -> BlockEncoding:
    """Block encoding of K_e = lam * K_e^lambda + mu * K_e^mu via Pauli LCU.

    Returns
    -------
    BlockEncoding
        n_system = 3, n_ancilla = ceil(log_2 L), alpha = sum |c_k|.
    """
    Ke_lam, Ke_mu, _, _ = element_template(a=a, b=b, phi_deg=phi_deg)
    Ke = lam * Ke_lam + mu * Ke_mu
    return pauli_block_encoding(Ke, name=name)


def build_UDk(
    L: int,
    a: float | None = None,
    b: float | None = None,
    phi_deg: float = 90.0,
    lam: float = 1.0,
    mu: float = 1.0,
    name: str = "U_DK",
) -> BlockEncoding:
    """Block encoding of D_K = I_{L^2} (x) K_e.

    Default element half-widths a = b = 1/(2L) match a unit cell partitioned
    into L x L equal squares.

    Parameters
    ----------
    L : int (power of 2)
    a, b : float, optional
        Element half-widths.  Defaults to 1/(2L).
    phi_deg : float
        Element shape parameter (90 deg = orthogonal axes).
    lam, mu : float
        Lame parameters.
    name : str
    """
    n_log = int(np.log2(L))
    if 2 ** n_log != L:
        raise NotImplementedError(f"L must be a power of 2; got L={L}")
    if a is None:
        a = 1.0 / (2 * L)
    if b is None:
        b = 1.0 / (2 * L)

    # Build the small 8x8 K_e block encoding.
    be_Ke = build_UKe(a=a, b=b, phi_deg=phi_deg, lam=lam, mu=mu, name=f"{name}_Ke")

    n_dof = 3                  # local DOF index (ell)
    n_elem = 2 * n_log         # element index (i_x, i_y)
    n_sys = n_dof + n_elem
    n_anc = be_Ke.n_ancilla

    # Layout (low -> high): dof (3) | elem (2 n_log) | anc (n_anc).
    qr_dof = QuantumRegister(n_dof, "dof")
    qr_elem = QuantumRegister(n_elem, "elem") if n_elem > 0 else None
    qr_anc = QuantumRegister(n_anc, "anc") if n_anc > 0 else None

    regs = [qr_dof]
    if qr_elem is not None:
        regs.append(qr_elem)
    if qr_anc is not None:
        regs.append(qr_anc)
    qc = QuantumCircuit(*regs, name=name)

    # Apply U_{K_e} on (dof + anc) only.  The element register is left untouched
    # (=> identity on the L^2 block-diagonal structure).
    qubits = list(qr_dof) + (list(qr_anc) if qr_anc is not None else [])
    qc.append(be_Ke.circuit, qubits)

    return BlockEncoding(
        circuit=qc,
        n_system=n_sys,
        n_ancilla=n_anc,
        alpha=be_Ke.alpha,
        name=name,
        target_shape=(2 ** n_sys, 2 ** n_sys),
    )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def _Dk_reference(L: int, a: float, b: float, phi_deg: float, lam: float, mu: float) -> np.ndarray:
    """Numpy reference for D_K = I_{L^2} (x) K_e."""
    Ke_lam, Ke_mu, _, _ = element_template(a=a, b=b, phi_deg=phi_deg)
    Ke = lam * Ke_lam + mu * Ke_mu
    return np.kron(np.eye(L * L), Ke)  # element index high, dof index low


if __name__ == "__main__":
    for L in [2, 4]:
        print(f"--- L = {L} ---")
        a_h = 1.0 / (2 * L)
        be = build_UDk(L, a=a_h, b=a_h, phi_deg=90.0, lam=1.0, mu=1.0)
        print(f"  {be.summary()}")
        D_ref = _Dk_reference(L, a=a_h, b=a_h, phi_deg=90.0, lam=1.0, mu=1.0)
        passed, err = be.verify_against(D_ref, atol=1e-10, verbose=True)
        assert passed, f"D_K test at L={L} failed (err={err})"

    print("\nAll build_UDk tests passed.")

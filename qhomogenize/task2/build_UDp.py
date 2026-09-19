"""
Task 2 (variable material): build the two-phase element-constitutive operator
and, with the assembly operator, the full two-material stiffness encoding.

Operator
--------
With a Boolean indicator p_e in {0,1} (1 = inclusion / material 1, 0 = base /
material 0), the per-element Lame parameters are affine in p_e:

    lambda_e = lambda_0 (1 - p_e) + lambda_1 p_e = lambda_0 + (lambda_1 - lambda_0) p_e,
    mu_e     = mu_0 (1 - p_e)     + mu_1 p_e     = mu_0     + (mu_1 - mu_0) p_e.

The element-constitutive operator is block-diagonal,
    D_p = I_{L^2} (x) [lambda_e K_e^lambda + mu_e K_e^mu]   (block e),
and the global stiffness is K = A^T D_p A.  Substituting the affine split and
distributing over the (mesh-independent) Lame templates gives a four-term LCU:

    K = lambda_0          * A^T (I (x) K_e^lambda) A
      + mu_0              * A^T (I (x) K_e^mu)     A
      + (lambda_1-lambda_0) * A^T M_p (I (x) K_e^lambda) A
      + (mu_1-mu_0)         * A^T M_p (I (x) K_e^mu)     A,

where M_p = diag(p_e) (x) I_8 is the inclusion projector.  The first two terms
are exactly the verified uniform machinery (build_UDk with the base material);
the last two are the same with the mask M_p inserted.  Only M_p is new.

This module provides:
  - build_UMp  : block encoding of the projector M_p (alpha = 1).
  - build_UDp_masked_lambda / _mu : A^T M_p (I (x) K_e^layer) A pieces are
    assembled at the K level in build_UK below; here we expose the masked
    element operator M_p (I (x) K_e^layer).
  - build_UK   : the full four-term-LCU two-material stiffness encoding.

Verification target: the classical assemble_K(L, p, lam, mu, ...) from
fea.full_K, in the pilot's own element convention (e = i_x + L i_y).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from qhomogenize.bencode.types import BlockEncoding
from qhomogenize.bencode.compose_product import compose_product, compose_triple
from qhomogenize.bencode.compose_lcu import compose_lcu
from qhomogenize.task1.build_UA import build_UA
from qhomogenize.task2.build_UDk import build_UDk, build_UKe
from qhomogenize.task2.square_indicator import square_indicator_oracle
from qhomogenize.task2.disk_indicator import disk_indicator_oracle


# ---------------------------------------------------------------------------
# Projector M_p = diag(p_e) (x) I_8
# ---------------------------------------------------------------------------
def build_UMp(L: int, indicator: str, params: dict, name: str = "U_Mp") -> BlockEncoding:
    """Block encoding of the inclusion projector M_p = diag(p_e) (x) I_8.

    The system register is (dof: 3) | (elem: 2 log2 L), matching build_UDk.
    M_p acts as identity on the dof sub-register and as diag(p_e) on the elem
    sub-register.

    Construction
    ------------
    A diagonal 0/1 projector P is block-encoded with alpha = 1 by an oracle
    that computes the predicate into a single ancilla and post-selects it to
    |0>.  Concretely, if O_p|e>|0> = |e>|p_e>, then
        (<0|_anc (x) I) O_p (|0>_anc (x) I) = diag(1 - p_e),
    i.e. O_p directly block-encodes the COMPLEMENT projector I - M_p.  To
    encode M_p itself we flip the flag after the oracle (X on the flag), so the
    post-selected block becomes diag(p_e) = M_p.

    The indicator oracle acts on the elem register's (i_x, i_y) decomposition.
    Recall elem index e = i_x + L i_y, so the elem register splits as
    low n_log bits = i_x, high n_log bits = i_y.
    """
    n_log = int(np.log2(L))
    if 2 ** n_log != L:
        raise NotImplementedError(f"L must be a power of 2; got L={L}")
    n_dof = 3
    n_elem = 2 * n_log

    # Build the chosen indicator oracle on its own (ix, iy, flag, work) qubits.
    if indicator == "square":
        orc, info = square_indicator_oracle(
            L, params["sx"], params["sy"], params["side_x"],
            params.get("side_y"), name="O_p_sq",
        )
    elif indicator == "disk":
        orc, info = disk_indicator_oracle(
            L, params["Cx"], params["Cy"], params["R2"], name="O_p_disk",
        )
    else:
        raise ValueError(f"unknown indicator '{indicator}'")

    n_x, n_y = info["n_x"], info["n_y"]
    assert n_x == n_log and n_y == n_log

    # The oracle circuit has registers: ix, iy, flag, <work...>.
    # Its qubit 0..n_x-1 = ix, next n_y = iy, next 1 = flag, rest = work ancilla.
    n_orc = orc.num_qubits
    n_work = n_orc - (n_x + n_y + 1)   # work qubits (incl. flag's siblings)
    # Ancilla for the projector = flag (1) + work (n_work).  All start/end |0>.
    n_anc = 1 + n_work

    # System register layout (low->high): dof (3) | elem (2 n_log)
    qr_dof = QuantumRegister(n_dof, "dof")
    qr_elem = QuantumRegister(n_elem, "elem")
    qr_anc = QuantumRegister(n_anc, "anc")   # [0] = flag, [1:] = work
    qc = QuantumCircuit(qr_dof, qr_elem, qr_anc, name=name)

    # Map oracle qubits onto our registers:
    #   ix  -> elem[0:n_log]      (low half = i_x)
    #   iy  -> elem[n_log:2n_log] (high half = i_y)
    #   flag-> anc[0]
    #   work-> anc[1:]
    orc_qubits = (
        [qr_elem[j] for j in range(n_log)] +
        [qr_elem[n_log + j] for j in range(n_log)] +
        [qr_anc[0]] +
        [qr_anc[1 + j] for j in range(n_work)]
    )
    qc.append(orc.to_gate(label="O_p"), orc_qubits)
    # Flip flag so post-select |0> keeps p_e = 1 (M_p, not I - M_p).
    qc.x(qr_anc[0])

    n_sys = n_dof + n_elem
    return BlockEncoding(
        circuit=qc,
        n_system=n_sys,
        n_ancilla=n_anc,
        alpha=1.0,
        name=name,
        target_shape=(2 ** n_sys, 2 ** n_sys),
    )


# ---------------------------------------------------------------------------
# Full two-material stiffness encoding  U_K = U_{A^T} U_{D_p} U_A
# assembled as a four-term LCU over the Lame split.
# ---------------------------------------------------------------------------
def build_UK(
    L: int,
    indicator: str,
    params: dict,
    E0: float, nu0: float,
    E1: float, nu1: float,
    name: str = "U_K_twophase",
):
    """Two-material stiffness block encoding for a periodic unit cell.

    Parameters
    ----------
    L : int (power of 2)
    indicator : {'square', 'disk'}
    params : dict
        Geometry parameters for the indicator oracle.
        square: {sx, sy, side_x[, side_y]} ; disk: {Cx, Cy, R2}.
    E0, nu0 : base material (p_e = 0).
    E1, nu1 : inclusion material (p_e = 1).

    Returns
    -------
    be_K : BlockEncoding of K = A^T D_p A.
    parts : dict of the component block encodings (for inspection/verification).
    """
    a_h = 1.0 / (2 * L)

    def lame(E, nu):
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))   # plane strain
        mu = E / (2 * (1 + nu))
        return lam, mu

    lam0, mu0 = lame(E0, nu0)
    lam1, mu1 = lame(E1, nu1)
    dlam, dmu = lam1 - lam0, mu1 - mu0

    # Assembly operators.
    be_A = build_UA(L)
    be_AT = be_A.conjugate_transpose(name="U_AT")

    # Element operators for the two Lame templates (unit Lame coeff; weights
    # carry the actual lam/mu so we can reuse one template encoding per layer).
    be_Dlam = build_UDk(L, a=a_h, b=a_h, phi_deg=90.0, lam=1.0, mu=0.0, name="U_Dlam")
    be_Dmu = build_UDk(L, a=a_h, b=a_h, phi_deg=90.0, lam=0.0, mu=1.0, name="U_Dmu")

    # Mask projector.
    be_Mp = build_UMp(L, indicator, params, name="U_Mp")

    # Masked element operators: M_p . D_layer  (product of block encodings).
    be_MpDlam = compose_product(be_Mp, be_Dlam, name="U_Mp_Dlam")
    be_MpDmu = compose_product(be_Mp, be_Dmu, name="U_Mp_Dmu")

    # The four D-level terms (before sandwiching by A^T ... A):
    #   T0 = lam0 * D_lam
    #   T1 = mu0  * D_mu
    #   T2 = dlam * M_p D_lam
    #   T3 = dmu  * M_p D_mu
    # Assemble D_p = T0+T1+T2+T3 via LCU, then sandwich once: K = A^T D_p A.
    comps = [be_Dlam, be_Dmu, be_MpDlam, be_MpDmu]
    weights = [lam0, mu0, dlam, dmu]
    # Drop exactly-zero-weight terms to avoid a degenerate PREP amplitude.
    nz = [(c, w) for c, w in zip(comps, weights) if abs(w) > 0]
    comps_nz = [c for c, _ in nz]
    weights_nz = [w for _, w in nz]
    be_Dp = compose_lcu(comps_nz, weights_nz, name="U_Dp")

    be_K = compose_triple(be_AT, be_Dp, be_A, name=name)

    parts = dict(
        be_A=be_A, be_AT=be_AT, be_Dlam=be_Dlam, be_Dmu=be_Dmu,
        be_Mp=be_Mp, be_MpDlam=be_MpDlam, be_MpDmu=be_MpDmu, be_Dp=be_Dp,
        lam0=lam0, mu0=mu0, lam1=lam1, mu1=mu1,
    )
    return be_K, parts


if __name__ == "__main__":
    # Smoke test: build at L=4 and report shapes/alphas (verification is in
    # the dedicated harness test_K_twophase.py).
    be_K, parts = build_UK(
        4, "square", dict(sx=1, sy=1, side_x=2),
        E0=1.0, nu0=0.3, E1=0.1, nu1=0.3,
    )
    print(be_K.summary())
    print("alpha_Mp =", parts["be_Mp"].alpha, " alpha_Dp =", parts["be_Dp"].alpha)

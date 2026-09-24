"""
pyfastinvert -- fast-inversion preconditioning of periodic two-phase operators.

Companion to pyblockencode, which encodes the operator; this encodes its
preconditioner. The two-phase cell splits into a mean part and a contrast part
(Equation 42 there),

    K^chi = (E1+E2)/2 K - (E1-E2)/2 sum_(a,b) R_(a,b) T_(a,b),

and the mean part is block-circulant, hence diagonalised by the QFT with a 2x2
symbol per mode. That makes it fast invertible in the sense of Tong et al.

    from .fastinvert import PRECONDITIONER, encode
    circuit, info = encode(PRECONDITIONER(nu=0.3), N=16)

Design
------
Do NOT block-encode the inverse and compose. ||A^{-1/2}|| grows as N, so
composing it with an encoding of the contrast part multiplies subnormalizations
and inflates alpha by kappa(A) ~ N^2, cancelling the gain. Split the inverse
square root onto both sides instead and encode the ISOMETRY

    V = W^{1/2} G A^{-1/2},    M = V^dag E V,    V^dag V = E_0^{-1} P,

whose symbol Vhat(p,q) = Ghat(p,q) Khat^{-1/2}(p,q) has both singular values
equal to 1 at every mode, for every nu and every N. V therefore carries no N in
its norm, and spec(M) = [1/rho, 1] exactly with rho the contrast, so alpha_M = 1.

The symbol and its inverse square root are closed form. For a real symmetric
2x2 Khat with s = sqrt(det) and t = tr,

    Khat^{-1/2} = [[d+s, -b], [-b, a+s]] / (s sqrt(t + 2s)),

two square roots and no eigensolve, and the zero mode is handled by
construction because the symbol vanishes there.

Circuit
-------
A QFT pair around a mode-multiplexed rotation. Khat^{-1/2} is real symmetric,
so it diagonalises in ONE rotation per mode rather than a two-sided
factorisation: theta = atan2(2b, a-d)/2 with eigenvalues in closed form. The
displacement qubit carries R(theta), an ancilla carries the scale, and R(theta)
is undone. No dense unitary and no arithmetic.

    U_V = (QFT (x) QFT (x) I) . R(theta)^T . SCALE . R(theta) . (QFT^dag ...)

Cost note
---------
The multiplexers are exact but dense: one angle per mode gives O(N^2) gates.
The polylog claim requires a register-controlled preparation of the symbol,
which is stated as an open assumption in the paper and is NOT implemented here.
The counts reported below are therefore an upper bound and a correctness
witness, not the asymptotic claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import ClassVar

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UCRYGate
from qiskit.quantum_info import Statevector
from qiskit.synthesis import synth_qft_full

BASIS = ["cx", "u"]
OPT_LEVEL = 2
TOL = 1e-12

CORNERS = [(0, 0), (1, 0), (1, 1), (0, 1)]      # ccw from lower left
_GAUSS = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)),
          (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]


# ==========================================================================
# 1. symbol algebra  (no matrices)
# ==========================================================================
def _factors(m: int):
    """Eigenvalues of K1 = circ(-1,2,-1), M1 = circ(1,4,1)/6, and sin."""
    t = 2 * np.pi * np.arange(2 ** m) / 2 ** m
    return 4 * np.sin(t / 2) ** 2, (2 + np.cos(t)) / 3, np.sin(t)


def symbol(m: int, nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    """The 2x2 acoustic tensor at every mode, from Equations (15) to (17)."""
    k, mu, s = _factors(m)
    C = E / (1 - nu ** 2)
    kx, ky, mx, my = k[:, None], k[None, :], mu[:, None], mu[None, :]
    out = np.empty((2 ** m, 2 ** m, 2, 2))
    out[..., 0, 0] = C * (kx * my + (1 - nu) / 2 * mx * ky)
    out[..., 1, 1] = C * ((1 - nu) / 2 * kx * my + mx * ky)
    out[..., 0, 1] = out[..., 1, 0] = C * (1 + nu) / 2 * s[:, None] * s[None, :]
    return out


def symbol_inv_sqrt(m: int, nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    """Khat^{-1/2}, zero at the zero mode. Closed form, no eigensolve."""
    S = symbol(m, nu, E)
    a, b, d = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    det, tr = a * d - b * b, a + d
    ok = det > TOL * det.max()
    s = np.sqrt(np.where(ok, det, 1.0))
    out = np.zeros_like(S)
    out[..., 0, 0], out[..., 1, 1] = d + s, a + s
    out[..., 0, 1] = out[..., 1, 0] = -b
    return np.where(ok, 1.0, 0.0)[..., None, None] * out / (
        s * np.sqrt(tr + 2 * s))[..., None, None]


def eigen(m: int, nu: float = 0.3):
    """Khat^{-1/2} = R(theta) diag(e0, e1) R(theta)^T, mode by mode.

    Real symmetric, so one rotation suffices: theta = atan2(2b, a-d)/2 and the
    eigenvalues follow from the half-trace and the radius.
    """
    Sh = symbol_inv_sqrt(m, nu, 1.0)
    a, b, d = Sh[..., 0, 0], Sh[..., 0, 1], Sh[..., 1, 1]
    hlf, rad = (a + d) / 2, np.sqrt(((a - d) / 2) ** 2 + b ** 2)
    return 0.5 * np.arctan2(2 * b, a - d), hlf + rad, hlf - rad


def gradient_symbol(m: int, nu: float = 0.3) -> np.ndarray:
    """Ghat(p,q), stacked over the four quadrature points. Shape (N,N,12,2).

    Rows are D^{1/2} B weighted by the quadrature measure, so that
    Ghat^dag Ghat = Khat exactly.
    """
    N, h = 2 ** m, 1.0 / 2 ** m
    D = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu ** 2)
    L = np.linalg.cholesky(D).T
    p = np.arange(N)
    ph = np.exp(-2j * np.pi * np.multiply.outer(p, [o[0] for o in CORNERS]) / N)
    qh = np.exp(-2j * np.pi * np.multiply.outer(p, [o[1] for o in CORNERS]) / N)
    out = np.zeros((N, N, 12, 2), complex)
    for g, (xi, eta) in enumerate(_GAUSS):
        dN = np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                       [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]]) / 4
        dNx = np.linalg.solve((h / 2) * np.eye(2), dN)
        B = np.zeros((3, 4, 2))                  # strain, corner, component
        for i in range(4):
            B[0, i, 0], B[1, i, 1] = dNx[0, i], dNx[1, i]
            B[2, i, 0], B[2, i, 1] = dNx[1, i], dNx[0, i]
        phase = ph[:, None, :] * qh[None, :, :]                  # (N,N,4)
        Bh = np.einsum("aci,pqc->pqai", B, phase)                # (N,N,3,2)
        out[:, :, 3 * g:3 * g + 3, :] = np.einsum(
            "ab,pqbj->pqaj", L, Bh) * np.sqrt(h * h / 4)
    return out


def isometry_symbol(m: int, nu: float = 0.3) -> np.ndarray:
    """Vhat = Ghat Khat^{-1/2}. Unit singular values at every mode."""
    return np.einsum("pqab,pqbc->pqac", gradient_symbol(m, nu),
                     symbol_inv_sqrt(m, nu, 1.0))


# ==========================================================================
# 2. operator specification
# ==========================================================================
@dataclass(frozen=True)
class PRECONDITIONER:
    """A^{-1/2} for the periodic plane-stress Q4 cell, as an isometry factor."""
    nu: float = 0.3
    kind: ClassVar[str] = "elasticity2d_precond"

    def __post_init__(self):
        if not -1.0 < self.nu < 0.5:
            raise ValueError(f"nu must lie in (-1, 0.5), got {self.nu}")

    def alpha(self, m: int) -> float:
        """Subnormalization: the largest eigenvalue of Khat^{-1/2}."""
        _, e0, e1 = eigen(m, self.nu)
        return float(max(e0.max(), e1.max()))


# ==========================================================================
# 3. the circuit
# ==========================================================================
def _qft(m: int, inverse: bool = False) -> QuantumCircuit:
    q = synth_qft_full(m, do_swaps=True)
    return q.inverse() if inverse else q


def spatial_qft(m: int, inverse: bool = False) -> QuantumCircuit:
    """QFT on x and y, identity on the displacement qubit."""
    qc = QuantumCircuit(2 * m + 1, name="QFT2" + ("dg" if inverse else ""))
    g = _qft(m, inverse).to_gate()
    qc.append(g, range(m))
    qc.append(g, range(m, 2 * m))
    return qc


class FastInversion:
    """Block encoding of A^{-1/2}. Builds no matrix by default."""

    def __init__(self, operator: PRECONDITIONER, m: int):
        if m < 1:
            raise ValueError("m must be >= 1")
        self.operator, self.m, self.nu = operator, m, operator.nu
        self.n_system = 2 * m + 1
        self.num_ancilla = 1
        self.num_qubits = self.n_system + 1
        self.alpha = operator.alpha(m)

    def info(self) -> dict:
        return dict(kind=self.operator.kind, m=self.m, N=2 ** self.m,
                    dofs=2 ** self.n_system, alpha=self.alpha,
                    system=self.n_system, ancilla=self.num_ancilla,
                    total_qubits=self.num_qubits)

    def circuit(self) -> QuantumCircuit:
        m = self.m
        theta, e0, e1 = eigen(m, self.nu)
        ang = 2 * np.arccos(np.clip(
            np.stack([e0, e1], axis=-1) / self.alpha, -1.0, 1.0))
        order = lambda A: np.transpose(A).ravel().tolist()

        x, y = QuantumRegister(m, "x"), QuantumRegister(m, "y")
        d, a = QuantumRegister(1, "d"), QuantumRegister(1, "a")
        qc = QuantumCircuit(x, y, d, a, name=f"A^-1/2[m={m}]")
        modes = list(x) + list(y)

        qc.compose(spatial_qft(m), qubits=modes + list(d), inplace=True)
        qc.append(UCRYGate(order(-2 * theta)), [d[0]] + modes)
        qc.append(UCRYGate(order(ang)), [a[0]] + modes + [d[0]])
        qc.append(UCRYGate(order(2 * theta)), [d[0]] + modes)
        qc.compose(spatial_qft(m, inverse=True), qubits=modes + list(d),
                   inplace=True)
        return qc

    def resources(self, optimization_level: int = OPT_LEVEL) -> dict:
        t = transpile(self.circuit(), basis_gates=BASIS,
                      optimization_level=optimization_level)
        c = t.count_ops()
        return dict(qubits=self.num_qubits, alpha=self.alpha,
                    cx=c.get("cx", 0), u=c.get("u", 0), depth=t.depth())

    # -- materialisation (opt-in) -----------------------------------------
    def matrix(self) -> np.ndarray:
        """A^{-1/2} densely, from the symbol, for verification."""
        N = 2 ** self.m
        T = symbol_inv_sqrt(self.m, self.nu, 1.0)
        v = np.eye(2 * N * N, dtype=complex)
        w = np.swapaxes(v.reshape(2, N, N, -1), 1, 2)
        f = np.fft.fft2(w, axes=(1, 2))
        g = np.einsum("xyde,exyc->dxyc", T, f)
        out = np.fft.ifft2(g, axes=(1, 2))
        return np.swapaxes(out, 1, 2).reshape(2 * N * N, -1).real

    def verify(self, atol: float = 1e-9) -> dict:
        """Check alpha * <0|U|0> against the symbol, and V's unit norm."""
        qc = self.circuit()
        n = self.n_system
        blk = np.zeros((2 ** n, 2 ** n), complex)
        for j in range(2 ** n):
            blk[:, j] = Statevector.from_int(
                j, 2 ** qc.num_qubits).evolve(qc).data[:2 ** n]
        sv = np.linalg.svd(isometry_symbol(self.m, self.nu),
                           compute_uv=False)
        off0 = np.ones(sv.shape[:2], bool)
        off0[0, 0] = False
        out = {"block_err": float(np.abs(self.alpha * blk - self.matrix()).max()),
               "isometry_err": float(np.abs(sv[off0] - 1.0).max())}
        out["ok"] = all(v <= atol for k, v in out.items() if k != "ok")
        return out


# ==========================================================================
# 4. user-facing entry point
# ==========================================================================
@dataclass
class EncodingInfo:
    operator: object
    kind: str
    m: int
    N: int
    dofs: int
    alpha: float
    qubits: int
    system: int
    ancilla: int
    cx: int | None = None
    depth: int | None = None
    verification: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        w = [f"{self.operator!r}   N = {self.N} per direction   "
             f"m = {self.m} qubits/direction   dofs = {self.dofs:,}",
             f"  alpha  = {self.alpha:.6f}",
             f"  qubits = {self.qubits}  (system {self.system}, "
             f"ancilla {self.ancilla})"]
        if self.cx is not None:
            w.append(f"  CX = {self.cx}   depth = {self.depth}      "
                     f"[transpiled to {BASIS}]")
        if self.verification is not None:
            v = self.verification
            w.append(f"  verified: block vs symbol  {v['block_err']:.2e}")
            w.append(f"            V unit norm      {v['isometry_err']:.2e}")
            w.append(f"            OK = {v['ok']}")
        return "\n".join(w)


def _resolve_size(N, m) -> int:
    if (N is None) == (m is None):
        raise ValueError("give exactly one of N or m")
    if m is not None:
        return int(m)
    if N < 2 or (N & (N - 1)):
        raise ValueError(f"N must be a power of two and at least 2, got {N}")
    return int(N).bit_length() - 1


def encode(operator, N: int | None = None, *, m: int | None = None,
           materialize: bool = False, costs: bool = True, atol: float = 1e-9):
    """Block encode the fast inverse of the mean part.

    Returns (circuit, info).
    """
    m = _resolve_size(N, m)
    fi = FastInversion(operator, m)
    qc = fi.circuit()
    i = fi.info()
    info = EncodingInfo(operator=operator, kind=i["kind"], m=m, N=i["N"],
                        dofs=i["dofs"], alpha=i["alpha"], qubits=i["total_qubits"],
                        system=i["system"], ancilla=i["ancilla"])
    if costs:
        r = fi.resources()
        info.cx, info.depth = r["cx"], r["depth"]
    if materialize:
        info.verification = fi.verify(atol=atol)
    return qc, info


if __name__ == "__main__":
    for nu in (0.0, 0.3, 0.45):
        circuit, info = encode(PRECONDITIONER(nu=nu), N=8, materialize=True)
        print(info)
        print()

    print("scaling, no materialisation")
    print(f"{'nu':>6} {'m':>3} {'qubits':>7} {'alpha':>9} {'CX':>8} "
          f"{'depth':>8}")
    for nu in (0.3,):
        for m in (2, 3, 4, 5):
            _, info = encode(PRECONDITIONER(nu=nu), m=m)
            print(f"{nu:>6.2f} {m:>3} {info.qubits:>7} {info.alpha:>9.4f} "
                  f"{info.cx:>8} {info.depth:>8}")

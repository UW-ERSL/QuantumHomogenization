"""isometry_circuit -- a gate-level circuit U_V for the partial isometry V.

Replaces the tabulated multiplexers of `fastinvert` (O(N^2) gates) with
reversible arithmetic whose gate count is independent of N up to O(log N).

V = W^{1/2} G (K^+)^{1/2} = (F^dag (x) I_12) blockdiag_k Vhat(k) (F (x) I_2)
(Proposition 2). This module builds the wavenumber-controlled stage
sum_k |k><k| (x) Q(k) without tabulating anything, and checks it.

Factorization used
------------------
With phi_j = pi k_j / N, S_j = sin phi_j, C_j = cos phi_j and

    a = S1 C2,   b = C1 S2,   c = S1 S2,   r = sqrt(a^2 + b^2),

the symbol is Shat = 4 [ kappa1 (a,b)^T (a,b) + (kappa2 r^2 + (kappa1 + 2 kappa2) c^2/3) I ]
with kappa1 = lambda + mu, kappa2 = mu, and

    Vhat(k) = e^{i (phi1 + phi2)} T . Rpair(2 beta, beta) . F0 . Rpsi(psi1, psi2) . Rq0(-beta)

restricted to the first two local states, where

    beta = atan2(b, a),   psi_i = atan2(G_i c, r),
    G1 = sqrt((kappa1 + 2 kappa2) / (3 (kappa1 + kappa2))),
    G2 = sqrt((kappa1 + 2 kappa2) / (3 kappa2)),

and T, F0 are fixed 16 x 16 unitaries (independent of k and N). So the circuit
needs only three angles per wavenumber. They are computed by gate-level
reversible arithmetic: a, b, c from two CORDIC rotations of the angles
pi (k1 + k2)/N and pi (k1 - k2)/N, then three CORDIC vectorings. The CORDIC
decision bits control the rotations directly; no angle register is formed.
The arithmetic is then run in reverse, leaving every work wire at zero.
"""
from __future__ import annotations

import numpy as np

from .revcirc import (Circuit, BitSim, add_view, copy_view, shr, load_select,
                      const_mult, cordic_rotate, cordic_vector, cordic_gain, to_int)


# ==========================================================================
# 1. material constants and the exact symbol (independent reference)
# ==========================================================================
def lame(nu):
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1 / (2 * (1 + nu))
    return lam, mu


def D0(nu):
    lam, mu = lame(nu)
    return np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])


_CORNERS = [(0, 0), (1, 0), (1, 1), (0, 1)]
_XI = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
_Q = 1 / np.sqrt(3)
_GP = [(-_Q, -_Q), (_Q, -_Q), (_Q, _Q), (-_Q, _Q)]


def _B(g, h):
    r, s = _GP[g]
    Bm = np.zeros((3, 8))
    for a, (xa, ya) in enumerate(_XI):
        dx = xa * (1 + ya * s) / (2 * h)
        dy = ya * (1 + xa * r) / (2 * h)
        Bm[:, 2 * a:2 * a + 2] = [[dx, 0], [0, dy], [dy, dx]]
    return Bm


def vhat_exact(N, nu, k1, k2):
    """Vhat(k) = W_e^{1/2} Ghat(k) Shat(k)^{-1/2}, 12 x 2, paper conventions."""
    if k1 == 0 and k2 == 0:
        return np.zeros((12, 2), complex)
    h = 1 / N
    d, Qm = np.linalg.eigh(D0(nu))
    Weh = (h / 2) * np.kron(np.eye(4), (Qm * np.sqrt(d)) @ Qm.T)
    t1, t2 = 2 * np.pi * k1 / N, 2 * np.pi * k2 / N
    Gh = np.vstack([sum(_B(g, h)[:, 2 * a:2 * a + 2] * np.exp(1j * (t1 * cx + t2 * cy))
                        for a, (cx, cy) in enumerate(_CORNERS)) for g in range(4)])
    A = Weh @ Gh
    S = (A.conj().T @ A).real
    e, U = np.linalg.eigh(S)
    return A @ ((U / np.sqrt(e)) @ U.T)


def V_assembled(N, nu):
    """V = W^{1/2} G (K^+)^{1/2}, assembled element by element (no symbol)."""
    h = 1 / N
    d, Qm = np.linalg.eigh(D0(nu))
    Weh_e = (h / 2) * np.kron(np.eye(4), (Qm * np.sqrt(d)) @ Qm.T)
    idx = lambda i, j: (i % N) * N + (j % N)
    G = np.zeros((12 * N * N, 2 * N * N))
    Bs = [_B(g, h) for g in range(4)]
    for i in range(N):
        for j in range(N):
            e = idx(i, j)
            for g in range(4):
                for a, (cx, cy) in enumerate(_CORNERS):
                    n = idx(i + cx, j + cy)
                    G[12 * e + 3 * g:12 * e + 3 * g + 3, 2 * n:2 * n + 2] += Bs[g][:, 2 * a:2 * a + 2]
    Wh = np.kron(np.eye(N * N), Weh_e)
    K = G.T @ Wh @ Wh @ G
    ev, U = np.linalg.eigh(K)
    ih = np.where(ev > 1e-10 * ev.max(), 1 / np.sqrt(np.abs(ev)), 0)
    return Wh @ G @ ((U * ih) @ U.T)


# ==========================================================================
# 2. local (4-qubit) building blocks; local index l = 8 b3 + 4 b2 + 2 b1 + b0
# ==========================================================================
def _R2(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def Rq0(t):
    """Rotation on local qubit 0, all other local states."""
    return np.kron(np.eye(8), _R2(t))


def Rpsi(t0, t1):
    """Rotation on local qubit 1 by t0 when b0 = 0 and by t1 when b0 = 1."""
    M = np.zeros((16, 16))
    for hi in range(4):                     # b3 b2
        for b0, t in ((0, t0), (1, t1)):
            i0, i1 = 4 * hi + b0, 4 * hi + 2 + b0
            R = _R2(t)
            M[np.ix_([i0, i1], [i0, i1])] = R
    return M


def Rpair(t2, t1):
    """Rotation on qubit 0 by t2 on states (0, 1) and by t1 on (2, 3)."""
    M = np.eye(16)
    for (i0, i1), t in (((0, 1), t2), ((2, 3), t1)):
        M[np.ix_([i0, i1], [i0, i1])] = _R2(t)
    return M


def F0_matrix(nu):
    """Fixed rotation on the pair (0, 4): D0 -> sqrt(k2) D0 + sqrt(k1) V0."""
    lam, mu = lame(nu)
    k1, k2 = lam + mu, mu
    chi = np.arctan2(np.sqrt(k1), np.sqrt(k2))
    M = np.eye(16)
    M[np.ix_([0, 4], [0, 4])] = _R2(chi)
    return M


def G_constants(nu):
    lam, mu = lame(nu)
    k1, k2 = lam + mu, mu
    return (np.sqrt((k1 + 2 * k2) / (3 * (k1 + k2))),
            np.sqrt((k1 + 2 * k2) / (3 * k2)))


def exact_angles(N, nu, k1, k2):
    f1, f2 = np.pi * k1 / N, np.pi * k2 / N
    a, b, c = np.sin(f1) * np.cos(f2), np.cos(f1) * np.sin(f2), np.sin(f1) * np.sin(f2)
    r = np.hypot(a, b)
    G1, G2 = G_constants(nu)
    return np.arctan2(b, a), np.arctan2(G1 * c, r), np.arctan2(G2 * c, r)


def canonical(nu, beta, p1, p2):
    """Rpair(2 beta, beta) F0 Rpsi(p1, p2) Rq0(-beta), first two columns."""
    return (Rpair(2 * beta, beta) @ F0_matrix(nu) @ Rpsi(p1, p2) @ Rq0(-beta))[:, :2]


def T_matrix(nu):
    """The fixed unitary T, in closed form.

    Canonical local states and their images in the strain space
    (row 3 g + c, c = xx, yy, gamma; g the Gauss point (xi_g, eta_g)):

        0  D0 : (i/2) sum_g (e_xx - e_yy)/sqrt 2      deviatoric, Gauss average
        1  G0 : (i/2) sum_g e_gamma                   shear, Gauss average
        2  s1 : sum_g [-sgn(eta_g) D^{1/2} e_xx - sgn(xi_g) D^{1/2} e_gamma] / (2 sqrt(lam + 3 mu))
        3  s2 : sum_g [-sgn(xi_g) D^{1/2} e_yy - sgn(eta_g) D^{1/2} e_gamma] / (2 sqrt(lam + 3 mu))
        4  V0 : (i/2) sum_g (e_xx + e_yy)/sqrt 2      volumetric, Gauss average

    The five images are orthonormal; T is completed to a unitary on the strain
    block 0..11 and fixes the unused states 12..15.
    """
    lam, mu = lame(nu)
    d, Qm = np.linalg.eigh(D0(nu))
    Dh = (Qm * np.sqrt(d)) @ Qm.T
    img = np.zeros((12, 5), complex)
    for g, (xg, eg) in enumerate(_GP):
        sl = slice(3 * g, 3 * g + 3)
        img[sl, 0] = 0.5j * np.array([1, -1, 0]) / np.sqrt(2)
        img[sl, 1] = 0.5j * np.array([0, 0, 1])
        img[sl, 2] = (-np.sign(eg) * Dh[:, 0] - np.sign(xg) * Dh[:, 2]) / (2 * np.sqrt(lam + 3 * mu))
        img[sl, 3] = (-np.sign(xg) * Dh[:, 1] - np.sign(eg) * Dh[:, 2]) / (2 * np.sqrt(lam + 3 * mu))
        img[sl, 4] = 0.5j * np.array([1, 1, 0]) / np.sqrt(2)
    assert np.allclose(img.conj().T @ img, np.eye(5), atol=1e-13)
    # complete: columns 0..4 fixed, 5..11 an orthonormal basis of the complement
    P = np.eye(12) - img @ img.conj().T
    U, _, _ = np.linalg.svd(P)
    T = np.eye(16, dtype=complex)
    T[:12, :5] = img
    T[:12, 5:12] = U[:, :7]
    assert np.allclose(T.conj().T @ T, np.eye(16), atol=1e-12)
    return T


# ==========================================================================
# 3. the circuit
# ==========================================================================
class IsometryCircuit:
    """U_V for N = 2^n, Poisson ratio nu, F fractional bits, I CORDIC iterations."""

    def __init__(self, n, nu=0.35, F=20, I=None, guard=4):
        self.n, self.N, self.nu, self.F = n, 2 ** n, nu, F
        self.I = I if I is not None else F
        self.w = F + guard
        if F < n:
            raise ValueError("need F >= n")
        self._build()

    # -- arithmetic -----------------------------------------------------------
    def _build(self):
        n, F, I, w = self.n, self.F, self.I, self.w
        c = Circuit()
        self.c = c
        self.k1, self.k2 = c.alloc(n), c.alloc(n)
        n_in = 2 * n
        K = cordic_gain(I)
        Fz, wz = F, F + 1

        # angles pi (k1 + k2)/N and pi (k1 - k2)/N, in units of pi, mod 2
        def place(reg):
            v = [None] * wz
            for j in range(n):
                v[Fz - n + j] = reg[j]
            return v

        zp, zm = c.alloc(wz), c.alloc(wz)
        copy_view(c, place(self.k1), zp)
        add_view(c, zp, place(self.k2))
        copy_view(c, place(self.k1), zm)
        add_view(c, zm, place(self.k2), neg=True)

        cs = {}
        for name, z in (("p", zp), ("m", zm)):
            f = c.alloc(1)[0]
            c.cx(z[-1], f); c.cx(z[-2], f)       # |z| >= 1/2
            c.cx(f, z[-1])                       # z <- z + 1 (mod 2)
            x, y = c.alloc(w), c.alloc(w)
            load_select(c, x, to_int(1 / K, F, w), to_int(-1 / K, F, w), f)
            cordic_rotate(c, x, y, z, I, F, Fz)
            cs[name] = (x, y)                    # (cos, sin)
        (xp, yp), (xm, ym) = cs["p"], cs["m"]

        a, b, cc = c.alloc(w), c.alloc(w), c.alloc(w)
        copy_view(c, shr(yp, 1), a); add_view(c, a, shr(ym, 1))
        copy_view(c, shr(yp, 1), b); add_view(c, b, shr(ym, 1), neg=True)
        copy_view(c, shr(xm, 1), cc); add_view(c, cc, shr(xp, 1), neg=True)
        self.abc = (a, b, cc)

        # beta = atan2(b, a): quadrant flag, then vectoring
        fb = c.alloc(1)[0]
        c.cx(a[-1], fb)
        for j in range(w):
            c.cx(fb, a[j]); c.cx(fb, b[j])       # (a, b) <- ~(a, b) ~ -(a, b)
        dbeta = cordic_vector(c, a, b, I)        # a <- K r
        G1, G2 = G_constants(self.nu)
        y1 = const_mult(c, cc, G1 * K, F)
        y2 = const_mult(c, cc, G2 * K, F)
        x1, x2 = c.alloc(w), c.alloc(w)
        copy_view(c, a, x1); copy_view(c, a, x2)
        dpsi1 = cordic_vector(c, x1, y1, I)
        dpsi2 = cordic_vector(c, x2, y2, I)

        # zero-mode flag
        kk = self.k1 + self.k2
        for q in kk:
            c.x(q)
        acc = kk[0]
        for q in kk[1:]:
            t = c.alloc(1)[0]
            c.ccx(acc, q, t)
            acc = t
        for q in kk:
            c.x(q)
        self.flag0 = acc
        self.fbeta, self.dbeta, self.dpsi1, self.dpsi2 = fb, dbeta, dpsi1, dpsi2
        self.n_arith = len(c.gates)
        self.arith_counts = c.counts()
        # uncompute: the exact reverse (all gates are self-inverse)
        c.gates += c.gates[::-1]
        self.n_wires = c.n

    # -- local gate list (controlled on work bits) ----------------------------
    def local_gates(self):
        """[(matrix16, [(wire, value), ...]), ...] in application order."""
        al = np.arctan(2.0 ** -np.arange(self.I))
        A = al.sum()
        g = []
        X2 = np.array([[0, 1], [1, 0]])
        P = np.kron(np.kron(X2, X2), np.eye(4))         # X on b3 and b2
        g.append((P, [(self.flag0, 1)]))                       # k = 0 -> unused states
        # Rq0(-beta), beta = pi fb + A - 2 sum d_i al_i
        g.append((Rq0(-A), []))
        for i, d in enumerate(self.dbeta):
            g.append((Rq0(2 * al[i]), [(d, 1)]))
        g.append((Rq0(-np.pi), [(self.fbeta, 1)]))
        # Rpsi(psi1, psi2), psi = A - 2 sum d_i al_i
        g.append((Rpsi(A, A), []))
        for i, d in enumerate(self.dpsi1):
            g.append((Rpsi(-2 * al[i], 0), [(d, 1)]))
        for i, d in enumerate(self.dpsi2):
            g.append((Rpsi(0, -2 * al[i]), [(d, 1)]))
        g.append((F0_matrix(self.nu).astype(complex), []))
        # Rpair(2 beta, beta)
        g.append((Rpair(2 * A, A), []))
        for i, d in enumerate(self.dbeta):
            g.append((Rpair(-4 * al[i], -2 * al[i]), [(d, 1)]))
        g.append((Rpair(2 * np.pi, np.pi), [(self.fbeta, 1)]))
        g.append((T_matrix(self.nu), []))
        return g

    # -- simulation -----------------------------------------------------------
    def simulate(self):
        """Run compute, read controls, apply local gates, run uncompute.

        Returns Q (N^2, 16, 16) including the phase e^{i(phi1+phi2)}, and a
        dict of checks.
        """
        N, n = self.N, self.n
        kk = np.arange(N * N)
        k1, k2 = kk // N, kk % N
        sim = BitSim(self.n_wires, N * N)
        sim.set_int(self.k1, k1)
        sim.set_int(self.k2, k2)
        sim.run(self.c.gates[:self.n_arith])
        Q = np.broadcast_to(np.eye(16, dtype=complex), (N * N, 16, 16)).copy()
        bits = {}
        for M, ctrl in self.local_gates():
            mask = np.ones(N * N, bool)
            for wire, val in ctrl:
                if wire not in bits:
                    bits[wire] = sim.get_bits(wire)
                mask &= bits[wire] == val
            Q[mask] = M @ Q[mask]
        # phase gates on the position register: e^{i pi k1/N} e^{i pi k2/N}
        ph = np.ones(N * N, complex)
        for j in range(n):
            ph *= np.where((k1 >> j) & 1, np.exp(1j * np.pi * 2 ** j / N), 1)
            ph *= np.where((k2 >> j) & 1, np.exp(1j * np.pi * 2 ** j / N), 1)
        Q *= ph[:, None, None]
        self.decoded = dict(
            beta=np.pi * sim.get_bits(self.fbeta) + sum(
                (1 - 2 * sim.get_bits(d)) * a for d, a in zip(self.dbeta, np.arctan(2.0 ** -np.arange(self.I)))),
            psi1=sum((1 - 2 * sim.get_bits(d)) * a for d, a in zip(self.dpsi1, np.arctan(2.0 ** -np.arange(self.I)))),
            psi2=sum((1 - 2 * sim.get_bits(d)) * a for d, a in zip(self.dpsi2, np.arctan(2.0 ** -np.arange(self.I)))))
        sim.run(self.c.gates[self.n_arith:])
        work = [q for q in range(self.n_wires) if q not in self.k1 + self.k2]
        dirty = int(sum(sim.get_bits(q).sum() for q in work))
        restored = bool(np.all(sim.get_int(self.k1, False) == k1) and
                        np.all(sim.get_int(self.k2, False) == k2))
        return Q, dict(work_bits_left_set=dirty, inputs_restored=restored)

    # -- resources -------------------------------------------------------------
    def resources(self):
        ar = self.arith_counts
        nI = self.I
        # controlled rotations on the local register, by number of controls
        # (work bit + local qubits); a k-local-control gate is counted by its
        # control count, before any decomposition.
        # controlled rotations on the local register, by control pattern
        rot = {"Ry on b0, ctrl d (input rotation by beta)": nI + 1,
               "Ry on b1, ctrl d and b0 (psi1, psi2)": 2 * nI,
               "Ry on b0, ctrl d and b3=b2=0 (pair (2,3) by beta)": nI + 1,
               "Ry on b0, ctrl d and b3=b2=b1=0 (pair (0,1), extra beta)": nI + 1}
        fixed = {"uncontrolled Ry": 4, "F0: Ry on b2, ctrl b3=b1=b0=0": 1,
                 "T: fixed 4-qubit unitary": 1, "zero mode: CX from flag to b3, b2": 2}
        return dict(N=self.N, F=self.F, I=self.I, width=self.w,
                    wires=self.n_wires + 4, peak_live_wires=self.c.peak_live + 4,
                    toffoli_compute=ar["ccx"], toffoli_total=2 * ar["ccx"],
                    cnot_total=2 * ar["cx"], x_total=2 * ar["x"],
                    controlled_rotations=rot, fixed_local_ops=fixed,
                    phase_gates=2 * self.n)


# ==========================================================================
# 4. QFT circuit and the dense end-to-end check
# ==========================================================================
def qft_matrix_from_gates(n):
    """Standard QFT (e^{+2 pi i jk/2^n}) built from H, controlled phases, swaps."""
    D = 2 ** n
    U = np.eye(D, dtype=complex)
    idx = np.arange(D)
    bit = lambda q: (idx >> (n - 1 - q)) & 1          # q = 0 is the most significant
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def apply_1q(U, q, M):
        out = np.zeros_like(U)
        b = bit(q)
        partner = idx ^ (1 << (n - 1 - q))
        out += M[b, b][:, None] * U
        out += M[b, 1 - b][:, None] * U[partner]
        return out

    for q in range(n):
        U = apply_1q(U, q, H)
        for t in range(q + 1, n):
            ph = np.exp(2j * np.pi / 2 ** (t - q + 1))
            U = np.where((bit(q) & bit(t))[:, None] == 1, ph * U, U)
    perm = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(D)])
    return U[perm]


def end_to_end_error(circ, Q):
    """|| (I x Pi_out) U_V (I x Pi_in) - V ||_2 from the gates, densely."""
    N, n = circ.N, circ.n
    F1s = qft_matrix_from_gates(n)                     # standard QFT
    F1 = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
    qft_err = np.abs(F1s - F1.conj().T).max()          # paper's F is QFT^dag
    Fm = np.kron(F1, F1)
    V = V_assembled(N, circ.nu)
    # U_V restricted: columns (position, local in {0,1}) -> rows (position, local 0..11)
    BD = np.zeros((12 * N * N, 2 * N * N), complex)
    for k in range(N * N):
        BD[12 * k:12 * k + 12, 2 * k:2 * k + 2] = Q[k, :12, :2]
    UV = np.kron(Fm.conj().T, np.eye(12)) @ BD @ np.kron(Fm, np.eye(2))
    return qft_err, np.linalg.norm(UV - V, 2), np.abs(UV.imag).max()

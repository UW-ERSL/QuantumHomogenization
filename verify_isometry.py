"""verify_isometry -- checks for the gate-level circuit U_V (Sec. 3.3 of the paper).

    python verify_isometry.py          # about 1 minute
    python verify_isometry.py --aer    # adds the Qiskit Aer cross-check (about 5 minutes)

1. Reversible arithmetic: adder exhaustive, shifted add/subtract exhaustive,
   constant multiply, CORDIC rotation and vectoring.
2. Closed-form factorization Vhat(k) = e^{i(phi1+phi2)} T Rpair F0 Rpsi Rq0,
   exact angles, every k, several N and nu. No fitting.
3. The circuit, simulated gate by gate: work register returns to zero, every
   block against Vhat(k), and the dense U_V (with the QFT built from gates)
   against V = W^{1/2} G (K^+)^{1/2} assembled element by element.
4. Error and Toffoli count against N and bit width F.
"""
import itertools
import sys

import numpy as np

from quantumhomogenize.revcirc import (Circuit, BitSim, cuccaro, add_view, shr,
                                       const_mult, cordic_rotate, cordic_vector,
                                       cordic_gain, to_int, load_select)
from quantumhomogenize.isometry_circuit import (IsometryCircuit, T_matrix, canonical,
                                                exact_angles, vhat_exact,
                                                end_to_end_error)


def arithmetic():
    w = 5
    c = Circuit(); a, b, cin = c.alloc(w), c.alloc(w), c.alloc(1)
    cuccaro(c, a, b, cin[0])
    A, B, C = np.array(list(itertools.product(range(2 ** w), range(2 ** w), range(2)))).T
    s = BitSim(c.n, len(A)); s.set_int(a, A); s.set_int(b, B); s.set_int(cin, C); s.run(c.gates)
    ok1 = np.all(s.get_int(b, False) == (A + B + C) % 2 ** w) and np.all(s.get_int(a, False) == A)

    w = 6; ok2 = True
    for i in range(4):
        c = Circuit(); x, y, e = c.alloc(w), c.alloc(w), c.alloc(1)
        add_view(c, x, shr(y, i), neg=e[0])
        X, Y, E = np.array(list(itertools.product(range(2 ** w), range(2 ** w), range(2)))).T
        s = BitSim(c.n, len(X)); s.set_int(x, X); s.set_int(y, Y); s.set_int(e, E); s.run(c.gates)
        ys = np.where(Y >> (w - 1), Y - 2 ** w, Y) >> i
        ok2 &= bool(np.all(s.get_int(x, False) == np.where(E == 1, X - ys, X + ys) % 2 ** w))
        ok2 &= all(s.get_bits(q).max() == 0 for q in range(c.n) if q not in x + y + e)

    F, wd = 16, 20
    c = Circuit(); x, y, z = c.alloc(wd), c.alloc(wd), c.alloc(F + 1)
    K = cordic_gain(F)
    load_select(c, x, to_int(1 / K, F, wd), to_int(1 / K, F, wd), x[0])
    cordic_rotate(c, x, y, z, F, F, F)
    ang = np.linspace(-0.5, 0.5, 2001); zi = np.round(ang * 2 ** F).astype(int)
    s = BitSim(c.n, len(ang)); s.set_int(z, zi); s.run(c.gates)
    th = zi / 2 ** F * np.pi
    e_rot = max(np.abs(s.get_int(x) / 2 ** F - np.cos(th)).max(),
                np.abs(s.get_int(y) / 2 ** F - np.sin(th)).max())

    c = Circuit(); x, y = c.alloc(wd), c.alloc(wd)
    d = cordic_vector(c, x, y, F)
    rng = np.random.default_rng(1)
    Xi = np.round(rng.uniform(0, 1, 3000) * 2 ** F).astype(int)
    Yi = np.round(rng.uniform(-1, 1, 3000) * 2 ** F).astype(int)
    s = BitSim(c.n, 3000); s.set_int(x, Xi); s.set_int(y, Yi); s.run(c.gates)
    al = np.arctan(2.0 ** -np.arange(F))
    got = sum((1 - 2 * s.get_bits(di)) * al[i] for i, di in enumerate(d))
    e_vec = (np.abs(got - np.arctan2(Yi, Xi)) * np.hypot(Xi, Yi) / 2 ** F).max()
    return ok1, ok2, e_rot, e_vec


def factorization():
    out = []
    for nu in (-0.3, 0.0, 0.35, 0.45):
        T, worst = T_matrix(nu), 0.0
        for N in (8, 64):
            for k1 in range(N):
                for k2 in range(N):
                    if k1 == k2 == 0:
                        continue
                    Vt = np.exp(1j * np.pi * (k1 + k2) / N) * (T @ canonical(nu, *exact_angles(N, nu, k1, k2)))[:12]
                    worst = max(worst, np.abs(Vt - vhat_exact(N, nu, k1, k2)).max())
        out.append((nu, worst))
    return out


def circuit_checks():
    out = []
    for n in (3, 4):
        c = IsometryCircuit(n, nu=0.35, F=24)
        Q, chk = c.simulate()
        q, e, _ = end_to_end_error(c, Q)
        out.append((c.N, chk["work_bits_left_set"], chk["inputs_restored"], q, e))
    return out


def scaling(nu=0.35):
    rows = []
    for n in (4, 5, 6, 7):
        N = 2 ** n
        vh = np.array([vhat_exact(N, nu, k1, k2) for k1 in range(N) for k2 in range(N)])
        for F in (16, 24, 32):
            c = IsometryCircuit(n, nu=nu, F=F)
            Q, chk = c.simulate()
            d = np.linalg.norm(Q[:, :12, :2] - vh, ord=2, axis=(1, 2)).max()
            r = c.resources()
            rows.append((N, F, d, chk["work_bits_left_set"] == 0, r["toffoli_total"], r["wires"]))
    return rows


if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    ok1, ok2, er, ev = arithmetic()
    print(f"adder exhaustive (w=5)            {ok1}")
    print(f"shifted add/sub exhaustive (w=6)  {ok2}")
    print(f"CORDIC rotation  F=16  max err    {er:.1e}")
    print(f"CORDIC vectoring F=16  r*|dang|   {ev:.1e}")
    for nu, e in factorization():
        print(f"closed-form factorization nu={nu:+.2f}, N=8 and 64: {e:.1e}")
    for N, dirty, rest, q, e in circuit_checks():
        print(f"N={N:<3} F=24  work bits left set {dirty}  inputs restored {rest}  "
              f"QFT circuit {q:.1e}  ||Pi U_V Pi - V||_2 {e:.2e}")
    print(f"{'N':>5} {'F':>4} {'block err':>10} {'clean':>6} {'Toffoli':>9} {'wires':>7}")
    for N, F, d, cl, t, w in scaling():
        print(f"{N:>5} {F:>4} {d:>10.2e} {str(cl):>6} {t:>9,} {w:>7,}")
    if "--aer" in sys.argv:
        import runpy
        runpy.run_path("verify_isometry_aer.py", run_name="__main__")
    print("---------------------------------------------------------------")

"""qsvt -- the QSVT pieces this package needs, vendored.

Extracted from UW-ERSL/SpectralCorrectionQSVT so that `quantumhomogenize` has
no dependency on that repository. Only what the fast-inversion solve actually
uses is here: the ChebIter polynomial, the normalisation QSVT requires, the QSP
phase angles, a block encoding, and the real-part-extracting circuit.

Provenance
----------
`ChebIterPolynomial` is the closed-form optimal odd polynomial for the RELATIVE
residual max |x p(x) - 1| on [a, 1], due to Gribling, Kerenidis and Szilagyi
(arXiv:2109.04248, Corollary 8), in the odd form of Sunderhauf et al.
(arXiv:2507.15537, Eq. 21). The relative criterion is the one this package
needs: the readout is Q = f^T K^{-1} f = sum_k w_k / lambda_k, so the error
that propagates is sum_k w_k |p(lambda_k) - 1/lambda_k|, and weighting by
lambda_k gives exactly that criterion.

The solver follows the real-part extraction of Martyn, Rossi, Tan and Chuang,
PRX Quantum 2, 040203 (2021), Sec. II: reading the QSP ancilla in the |+>/|->
basis projects onto Re P by the circuit itself, at no extra qubit, gate or
query.

pyqsp
-----
Phase angles need `pyqsp`, which is imported lazily so that the polynomial and
the degree bounds work without it. Install with

    pip install "setuptools<60"
    pip install pyqsp --no-build-isolation

since pyqsp's legacy setup.py does not build against modern setuptools.

Caveat
------
`block_encoding` builds a DENSE (2N x 2N) unitary. That reintroduces the cost
this package exists to remove and caps the mesh at m = 4. It is a correctness
witness for the numerical chain, not a resource claim; `fastinvert` holds the
structured encoding.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import scipy.linalg
from numpy.polynomial.chebyshev import Chebyshev
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, Statevector


# ==========================================================================
# 1. the polynomial
# ==========================================================================
class ChebIterPolynomial:
    """Closed-form optimal odd polynomial for max |x p(x) - 1| on [a, 1].

        p(x) = (1 - (-1)^n T_n(u(x)) / T_n(z0)) / x,
        u(x) = (2x^2 - (1+a^2)) / (1 - a^2),   z0 = (1+a^2)/(1-a^2).

    The residual equioscillates, so the achieved error is exact rather than a
    bound: eps = 1 / cosh(n arccosh z0) with n = (d+1)/2, which inverts in
    closed form to give mindegree.
    """

    @staticmethod
    def _z0(a: float) -> float:
        return (1.0 + a * a) / (1.0 - a * a)

    @staticmethod
    def poly(d: int, a: float) -> Chebyshev:
        if d % 2 == 0:
            raise ValueError("d must be odd")
        n = (d + 1) // 2
        Tn_z0 = np.cosh(n * np.arccosh(ChebIterPolynomial._z0(a)))

        def f(x):
            u = (2.0 * x * x - (1.0 + a * a)) / (1.0 - a * a)
            inside = np.abs(u) <= 1.0
            Tn = np.where(inside,
                          np.cos(n * np.arccos(np.clip(u, -1.0, 1.0))),
                          np.cosh(n * np.arccosh(np.maximum(np.abs(u), 1.0)))
                          * np.sign(u) ** n)
            return (1.0 - ((-1) ** n) * Tn / Tn_z0) / np.where(x == 0, 1e-300, x)

        coef = np.polynomial.chebyshev.chebinterpolate(f, d)
        coef[0::2] = 0.0                      # enforce odd parity exactly
        return Chebyshev(coef)

    @staticmethod
    def error_for_degree(d: int, a: float) -> float:
        n = (d + 1) // 2
        return 1.0 / np.cosh(n * np.arccosh(ChebIterPolynomial._z0(a)))

    @staticmethod
    def mindegree(epsilon: float, a: float) -> int:
        n = int(np.ceil(np.arccosh(1.0 / epsilon)
                        / np.arccosh(ChebIterPolynomial._z0(a))))
        return 2 * n - 1


# ==========================================================================
# 2. normalisation and phases
# ==========================================================================
def normalise(poly: Chebyshev, degree: int) -> tuple[Chebyshev, float]:
    """Scale p so |p| <= 1 on ALL of [-1, 1], as QSVT requires.

    The maximum frequently falls in the central gap (-a, a) where no
    eigenvalue lies, so the global maximum is the right normaliser. Returns
    (p / tau, tau).
    """
    n_sample = 25 * degree
    x = np.linspace(-1, 1, n_sample)
    tau = np.max(np.abs(poly(x))) / np.cos(np.pi * degree / (2 * n_sample))
    p = Chebyshev(poly.coef / tau)

    peak = np.max(np.abs(p(np.linspace(-1, 1, 2000))))
    if peak > 0.999:
        s = 0.999 / peak
        p, tau = Chebyshev(p.coef * s), tau / s
    return p, float(tau)


def qsp_phases(poly_normalised: Chebyshev, crit: float = 1e-12) -> list[float]:
    """Phase angles in the Wx convention, target in Re <0|U|0>.

    The symmetric-QSP Newton solver of Dong et al. reaches machine precision
    where pyqsp's default Laurent completion stalls near 1e-4 at these degrees.
    It encodes the target in Im <0|U|0>, so the leading phase is shifted by
    -pi/2 to rotate it onto the real part the circuit reads. Falls back to the
    Laurent method if Newton does not converge.
    """
    from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    from pyqsp.sym_qsp_opt import newton_solver as _newton

    coef = np.asarray(poly_normalised.coef, float)
    parity = 1 if np.max(np.abs(coef[0::2])) <= 1e-8 else 0
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, _, seq = _newton(coef[parity::2], parity, crit=crit)
        ph = np.array(seq.full_phases, float)
        ph[0] -= np.pi / 2
        return [float(v) for v in ph]
    except Exception:
        return [float(v) for v in
                QuantumSignalProcessingPhases(poly_normalised,
                                              signal_operator="Wx")]


# ==========================================================================
# 3. the circuit
# ==========================================================================
def block_encoding(A: np.ndarray) -> Operator:
    """Dense (2N x 2N) block encoding matching pyqsp's Wx signal.

        U = [[ A,                i sqrt(I - A A^dag) ],
             [ i sqrt(I - A^dag A),        A^dag     ]]

    so the effective 2x2 sub-unitary at each singular value sigma is exactly
    W(sigma). Every singular value must lie strictly inside (0, 1): at
    sigma = 1 the square root is singular and the block is not encodable.
    """
    N = A.shape[0]
    Ad = A.conj().T
    U = np.block([[A, 1j * scipy.linalg.sqrtm(np.eye(N) - A @ Ad)],
                  [1j * scipy.linalg.sqrtm(np.eye(N) - Ad @ A), Ad]])
    err = np.max(np.abs(U @ U.conj().T - np.eye(2 * N)))
    if err > 1e-10:
        raise ValueError(f"block encoding not unitary, max error {err:.2e}")
    return Operator(U)


class QSVT:
    """QSVT solve for a quadratic form in the resolvent.

    The observable here is f^T A^{-1} f, not the solution state, so `solve`
    returns the contraction directly and no amplitude of A^{-1} f is ever read
    individually.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, kappa: float,
                 target_error: float = 1e-3):
        self.A = A
        self.b = b / np.linalg.norm(b)
        self.n = int(np.log2(len(b)))
        self.kappa = kappa
        d = ChebIterPolynomial.mindegree(target_error, 1.0 / kappa)
        self.degree = d
        self.achieved_error = ChebIterPolynomial.error_for_degree(d, 1.0 / kappa)
        p, self.tau = normalise(ChebIterPolynomial.poly(d, 1.0 / kappa), d)
        self.angles = qsp_phases(p)

    def circuit(self) -> QuantumCircuit:
        """Real-part extraction by reading the QSP ancilla in |+>/|->."""
        anc = QuantumRegister(1, "anc")
        dat = QuantumRegister(self.n, "b")
        qc = QuantumCircuit(anc, dat)
        qc.prepare_state(Statevector(self.b), dat)
        qc.h(anc[0])
        U = block_encoding(self.A).to_instruction()
        for phi in self.angles[:-1]:
            qc.rz(-2.0 * phi, anc[0])
            qc.append(U, list(dat) + list(anc))
        qc.rz(-2.0 * self.angles[-1], anc[0])
        qc.h(anc[0])
        return qc

    def solve(self) -> tuple[np.ndarray, float, float]:
        """Returns (unit solution direction, success probability, real norm).

        p(A)|b> has norm tau * norm_real, so a quadratic form follows as
        (b . u_dir) * tau * norm_real.
        """
        sv = Statevector.from_instruction(self.circuit())
        amp = sv.data[0::2]                       # qubit 0 is the ancilla
        prob = float(np.sum(np.abs(amp) ** 2))
        u = np.real(amp)
        nrm = float(np.linalg.norm(u))
        return u / nrm, prob, nrm


# ==========================================================================
if __name__ == "__main__":
    print("----- output --------------------------------------------------")
    print("ChebIter degree and its equioscillating error")
    print(f"{'kappa':>9} {'eps':>9} {'degree':>8} {'achieved':>11}")
    for kap in (10.0, 100.0, 1e4):
        for eps in (1e-3, 1e-6):
            d = ChebIterPolynomial.mindegree(eps, 1.0 / kap)
            print(f"{kap:>9.0f} {eps:>9.0e} {d:>8} "
                  f"{ChebIterPolynomial.error_for_degree(d, 1.0/kap):>11.2e}")

    print("\nA 4x4 solve, against the classical value")
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    A = Q @ np.diag([0.99, 0.6, 0.3, 0.12]) @ Q.T
    b = rng.standard_normal(4)
    b /= np.linalg.norm(b)
    s = QSVT(A, b, kappa=0.99 / 0.12, target_error=1e-4)
    u, prob, nrm = s.solve()
    q = abs(float(b @ u)) * s.tau * nrm
    c = float(b @ np.linalg.solve(A, b))
    print(f"  degree {s.degree}, quantum {q:.6f}, classical {c:.6f}, "
          f"rel err {abs(q - c) / abs(c):.2e}")
    print("---------------------------------------------------------------")

# QuantumHomogenize

Quantum homogenization via fast inversion, for 2D periodic two-phase elasticity.

Computational homogenization of a periodic cell on a quantum computer. The
operator is block-encoded by the companion package
[`pyblockencode`](https://github.com/UW-ERSL/PyBlockEncode); this package
preconditions it by fast inversion of the mean part, establishes the
conditioning results that follow, and reduces the homogenized moduli to scalar
quadratic forms in the resolvent.

## Install

    pip install -r requirements.txt

and install the companion package:

    pip install git+https://github.com/UW-ERSL/PyBlockEncode

`pyqsp` supplies the QSP phase angles and does not build against modern
setuptools. If it fails:

    pip install "setuptools<60"
    pip install pyqsp --no-build-isolation

Everything except the phase angles works without it.

## Quick start

```python
import quantumhomogenize as qh

chi = qh.square_t(m=3, t=2)                    # centred square, v_f = 1/4
KH, GH, ratio = qh.bulk_shear(3, chi, r=10)    # two scalars, one solve each

circuit, info = qh.encode(qh.PRECONDITIONER(nu=0.3), m=3, materialize=True)
print(info)                                    # alpha, CX, depth, verification
```

    jupyter lab QuantumHomogenize_Tutorial.ipynb
    python verify.py                            # every proposition, asserted
    python -m quantumhomogenize.fastinvert      # the preconditioner circuit
    python -m quantumhomogenize.scope           # where the constant holds

## Layout

    QuantumHomogenize/
      QuantumHomogenize_Tutorial.ipynb   run-through, all outputs executed
      verify.py                          every proposition, with assertions
      requirements.txt
      quantumhomogenize/                 the package

| Module | Contents |
|---|---|
| `fourier_symbol` | Closed-form 2x2 symbol of the mean part, inverse, inverse square root |
| `fastinvert` | Block encoding of `A^{-1/2}`: QFT pair, multiplexed rotation, one ancilla |
| `material` | Plane-strain `D`; its plane-stress equivalent; `alpha` under plane strain |
| `macroload` | Macroscopic-strain load; the phase-interior node set |
| `microstructure` | Centred squares at any volume fraction, with measured oracle costs |
| `homogenized` | `C^H`, the Voigt term, bulk and shear moduli |
| `precondition` | `M = A^{-1/2} K^chi A^{-1/2}`; spectrum, kappa, kappa_eff, matrix-free apply |
| `conditioning` | Effective against worst case; truncation bias; budget insensitivity |
| `qspdegree` | Polynomial degree on the effective interval |
| `qsvt` | ChebIter polynomial, QSP phases, solver (vendored) |
| `fastsolve` | End to end: precondition, QSVT, modulus |
| `scope` | Where the effective constant holds and where it degrades |

`qsvt` is extracted from
[`SpectralCorrectionQSVT`](https://github.com/UW-ERSL/SpectralCorrectionQSVT):
`ChebIterPolynomial` is the closed-form optimum for the relative residual
(Gribling, Kerenidis and Szilagyi, arXiv:2109.04248, Cor. 8), and the solver
follows the real-part extraction of Martyn et al., PRX Quantum 2, 040203 (2021).
Only what this package uses is vendored, so there is no dependency on that
repository.

## What is verified

- Symbol against element-by-element assembly, 1e-16, every nu and m.
- `V^dag V = E_0^{-1} P` and `||V|| = E_0^{-1/2}`, machine precision.
- `Vhat` unit singular values at every mode, 5e-16.
- `spec(M) = [1/rho, 1]` exactly, five figures, every nu and m.
- Load vanishes at every phase-interior node, machine zero, every contrast.
- Bulk and shear from one quadratic form each, against the full tensor.
- Discarded spectral weight at the double-precision floor, flat from
  rho = 1e2 to 1e8; truncation bias 1e-30 to 1e-23.
- Preconditioner circuit block against the symbol, 2e-15.
- End-to-end solve within the 1e-3 target at m <= 4.

## What is NOT established

1. **Polylog gate count.** The multiplexers in `fastinvert` are exact but
   dense: one angle per mode, so CX grows as O(N^2) (76, 280, 1072, 4176 at
   m = 2..5). The polylog claim needs a register-controlled symbol preparation,
   an open assumption in the paper, not implemented here.

2. **Finite-contrast orthogonality.** The truncation argument behind the degree
   claim is proved only in the true-void limit, where disjoint supports give
   exact orthogonality. At finite contrast the overlap is measured at the
   double-precision floor across six decades, but the mechanism is
   unidentified. Spectral correction was tested as a bypass and fails: the
   corrected eigenvalue is four decades below the approximation interval.

3. **kappa_eff grows like log N.** About +0.38 per mesh doubling (2.88, 3.24,
   3.62, 4.00 at m = 3..6, hydrostatic, rho = 1e4, square v_f = 1/4, plane
   strain). Contrast independence holds cleanly; mesh
   independence does not. The degree claim is O(log N), not O(1).

4. **`fastsolve` uses a dense block encoding**, reintroducing the cost this
   construction removes, and caps at m = 4. Composing `fastinvert` with
   `pyblockencode` directly is the naive route the paper argues against; the
   composition must encode V as one object.

5. **Not implemented.** The geometric-mean form.

## Conventions

Paper 1 throughout: `N = 2^m`, dof index `d*N^2 + y*N + x` with x low and y high
and the displacement qubit most significant, `chi[ex, ey]` the inclusion
indicator. The material law is plane strain, since the cell is a transverse
section of a continuous-fiber composite; `quantumhomogenize/material.py` is the
one place `D` is built. Plane strain at `(E, nu)` is plane stress at
`E* = E/(1-nu^2)`, `nu* = nu/(1-nu)`, so paper 1's element and subnormalization
carry over by substitution: `alpha = E(33-32nu)/(6(1+nu)(1-2nu))` single phase.
Unit cell with `h = 1/N`: the Q4
stiffness is scale free in 2D so the operator is unchanged, the load carries one
factor of h, and `|Omega| = 1`.

Two choices the draft has not fixed. `A` is the stiff-phase operator
`max(E1,E2) * K`, giving `alpha_M <= 1` and spectrum `[1/rho, 1]`; the mean part
`(E1+E2)/2 * K` would give `alpha_M <= 2`. And `rho >= 1` is the contrast, which
is `max(r, 1/r)` in paper 1's signed `r = E1/E2`.

## Gate-level circuit for the isometry V (`isometry_circuit.py`, `revcirc.py`)

`fastinvert` builds the wavenumber-controlled stage of U_V from tabulated
multiplexers, which costs O(N^2) gates. `isometry_circuit` replaces the table
with reversible arithmetic, so the stage costs a number of gates independent
of N apart from O(log N) for the wavenumber input.

**Factorization.** With phi_j = pi k_j / N, a = sin phi1 cos phi2,
b = cos phi1 sin phi2, c = sin phi1 sin phi2 and r = (a^2 + b^2)^{1/2},

    Vhat(k) = e^{i(phi1 + phi2)} T . Rpair(2 beta, beta) . F0 . Rpsi(psi1, psi2) . Rq0(-beta),

    beta = atan2(b, a),   psi_i = atan2(G_i c, r),

where T and F0 are fixed 16 x 16 unitaries given in closed form and
G_1, G_2 depend on nu only. Three angles per wavenumber, no square root, no
division, no determinant.

**Arithmetic.** a, b, c come from two CORDIC rotations of pi (k1 +- k2)/N;
beta, psi1, psi2 from three CORDIC vectorings, whose decision bits control
the rotations directly. Gates are X, CX, CCX (Cuccaro adders); the arithmetic
is then reversed, so every work wire returns to zero.

**Verified** (`python verify_isometry.py`, about 1 minute):

- adders exhaustively; CORDIC to its truncation error;
- closed-form factorization to 3e-15 for every k, nu in {-0.3, 0, 0.35, 0.45};
- gate-level simulation of the whole circuit for every k, N = 4 ... 256,
  F = 12 ... 32 (39 runs): work register clean in every run, block error
  delta <= 7 N 2^-F;
- dense U_V, with the QFT built from gates, against V assembled element by
  element (N = 8, 16);
- independently, the exported 727-qubit circuit in Qiskit Aer (MPS) for every
  k at N = 4, agreeing to 9e-10 (`python verify_isometry.py --aer`).

**Cost per application of U_V:** Toffoli = 51 F^2 + 196 F + 66 + 4 log2 N
(measured), with F = ceil(log2(7 N / delta)) bits for block error delta;
about 6F controlled single-qubit rotations; about 4 F^2 qubits (Bennett
garbage, not optimized). For N = 1024 and delta = 1e-6: F = 33,
62,270 Toffolis, 4,170 qubits.

Not included: Clifford+T synthesis of the rotations (simulated exactly), and
qubit-count reduction by pebbling or measurement-based uncomputation.

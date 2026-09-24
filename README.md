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

3. **kappa_eff grows like log N.** About +0.39 per mesh doubling (2.99, 3.39,
   3.77, 4.16 at m = 3..6). Contrast independence holds cleanly; mesh
   independence does not. The degree claim is O(log N), not O(1).

4. **`fastsolve` uses a dense block encoding**, reintroducing the cost this
   construction removes, and caps at m = 4. Composing `fastinvert` with
   `pyblockencode` directly is the naive route the paper argues against; the
   composition must encode V as one object.

5. **Not implemented.** The geometric-mean form.

## Conventions

Paper 1 throughout: `N = 2^m`, dof index `d*N^2 + y*N + x` with x low and y high
and the displacement qubit most significant, `chi[ex, ey]` the inclusion
indicator, plane stress with `C = E/(1-nu^2)`. Unit cell with `h = 1/N`: the Q4
stiffness is scale free in 2D so the operator is unchanged, the load carries one
factor of h, and `|Omega| = 1`.

Two choices the draft has not fixed. `A` is the stiff-phase operator
`max(E1,E2) * K`, giving `alpha_M <= 1` and spectrum `[1/rho, 1]`; the mean part
`(E1+E2)/2 * K` would give `alpha_M <= 2`. And `rho >= 1` is the contrast, which
is `max(r, 1/r)` in paper 1's signed `r = E1/E2`.

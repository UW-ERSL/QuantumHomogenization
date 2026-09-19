# QuantumHomogenization

Quantum block-encoding for computational homogenization.  Two POCs:

- **Aim 1**: Structured operator encoding per Sünderhauf, Campbell & Camps
  (Quantum 8, 1226 (2024)) for the elasticity stiffness operator
  $K = A^T D_K A$ on periodic 2D meshes.
- **Aim 2**: Closed-form preconditioner $K_0^{-1}$ via
  $U_F^\dagger \, U_{\widehat{K}_0^{-1}} \, U_F$, built on the Aim 1 framework.

## Install

```bash
conda activate quantum
pip install -e .
```

## Run tests

```bash
pytest                  # full suite (~30 sec at L=4)
pytest -v               # verbose output
pytest -k smoke         # just the import smoke test
```

Or use VS Code's Testing panel (the flask icon in the left sidebar).

## Run individual modules

Each module has a `__main__` block with self-tests for development.  Open the
file in VS Code and press F5, or:

```bash
python -m qhomogenize.bencode.builder      # 4 reference matrices
python -m qhomogenize.task1.build_UA       # U_A vs numpy at L=2 and L=4
python -m qhomogenize.task2.test_K_uniform # full K = A^T D_K A end-to-end
```

## Generate figures

```bash
python generate_figures.py
```

Produces all figures into `./figures/`.  Two figures currently:
- `headline_cx_scaling.png` — Aim 1 structured-vs-baseline CX scaling
- `aim2_preconditioner_scaling.png` — Aim 2 preconditioner CX scaling +
  void-fraction-bounded condition number

## Aim 2 headline result

The preconditioned operator $P \cdot K \cdot K_0^{-1} \cdot P$ has a
**bounded, mesh-quasi-independent condition number** at fixed void fraction:

| Void fraction (achieved) | $\kappa$ at $M=4$ | $\kappa$ at $M=8$ |
|---|---|---|
| 25% | 2.90 | 3.78 |

By contrast, $\kappa(K)$ unpreconditioned scales as $M^2$. This validates
the central scientific claim of the Aim 2 proposal text: the preconditioner
delivers mesh-independent iteration counts for downstream QSVT-based
solvers.

The Aim 2 POC validates the *closed-form preconditioner construction*
(Fourier symbol, spatial QFT, zero-mode handling, integration with the
Aim 1 $K$).  The CX cost of $U_{K_0^{-1}}$ in isolation is the
Pauli-LCU baseline cost; a polylog-CX construction via reversible
trig + arithmetic + reciprocal is documented as future work in
`qhomogenize/task4/README.md` and was deliberately scoped out of the POC.

See `qhomogenize/task4/README.md` for the full Aim 2 module structure
and validation table.

## Documentation

See `qhomogenize/README.md` for the package-level documentation, build
sequence, and headline-result table comparing the structured composition
to the Pauli-LCU baseline.

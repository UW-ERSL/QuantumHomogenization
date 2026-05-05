# QuantumHomogenization

Quantum block-encoding for computational homogenization. POC implementing
structured operator encoding per Sünderhauf, Campbell & Camps (Quantum 8,
1226 (2024)) for the elasticity stiffness operator K = A^T D_K A on
periodic 2D meshes.

## Install

```bash
conda activate quantum   # or your env of choice
pip install -e .
```

## Run a test

```bash
python -m qhomogenize.tests.test_smoke
```

Or open any module file in VS Code and press F5 / the green Run button.

## Headline result

See `qhomogenize/README.md` for the package-level documentation, the build
sequence, and the structured-vs-baseline CX scaling table.

# QuantumHomogenization

Quantum block-encoding for computational homogenization.  POC implementing
structured operator encoding per Sünderhauf, Campbell & Camps (Quantum 8,
1226 (2024)) for the elasticity stiffness operator K = A^T D_K A on
periodic 2D meshes.

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

Produces all figures into `./figures/`.  The headline figure is the
structured-vs-baseline CX-scaling plot.  Open `generate_figures.py` in VS Code
and press F5 to run, or call individual figure functions interactively.

## Documentation

See `qhomogenize/README.md` for the package-level documentation, build
sequence, and headline-result table comparing the structured composition
to the Pauli-LCU baseline.

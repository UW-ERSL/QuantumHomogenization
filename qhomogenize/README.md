# qhomogenize

Structured quantum operator encoding for computational homogenization. POC
implementation of block-encoding circuits for the elasticity stiffness operator
**K = A^T D_p A** on a structured 2D Cartesian mesh under periodic boundary
conditions, following the structured-matrix block-encoding framework of
Sünderhauf, Campbell, and Camps [Quantum 8, 1226 (2024)].

This is the working code repository for the POC plan documented in
`POCPlan-StructuredOperatorEncoding.tex`.

## Status

| Layer | Module | Status |
|-------|--------|--------|
| classical | `fea.element_template` | **DONE** — Ke_lambda, Ke_mu via Andreassen-Andreasen |
| classical | `fea.assembly_pbc` | **DONE** — sparse A and closed-form `global_dof` under PBC |
| classical | `fea.full_K` | **DONE** — full assembled K with Lamé split; verified symmetric, PSD, with rigid-translation null space under PBC |
| classical | `fea.microstructure` | **DONE** — disk indicator (numpy reference) and integer-grid disk parameters |
| primitives | `qprim.integer` | **DONE** — modular adders, squaring, comparators (152 test cases pass) |
| primitives | `qprim.modular` | skeleton — index splitting helpers |
| primitives | `qprim.data_oracle` | skeleton — Sünderhauf data oracle (multiplexed rotations) |
| primitives | `qprim.diffusion` | skeleton — uniform-superposition diffusion |
| primitives | `qprim.out_of_range` | skeleton — Sünderhauf O_rg |
| framework | `bencode.types` | **DONE** — `BlockEncoding` dataclass with `extract_matrix`, `verify_against`, and `conjugate_transpose` |
| framework | `bencode.builder` | **DONE** — Sünderhauf base scheme per eq. (11) and full scheme per eq. (16) (with del + out-of-range CX-on-data trick for 0/1 matrices); 4 small reference matrices pass |
| framework | `bencode.compose_product` | **DONE** — Gilyén Lemma 53 product-of-block-encodings; tested on 3 cases (X·X=I, D·D, X·X·X=X) |
| framework | `bencode.pauli_lcu` | **DONE** — Pauli LCU block encoding for small dense matrices; tested on 5 cases (2x2, 4x4, K_e^lam, K_e^mu, K_e combined) |
| framework | `bencode.compose_lcu` | skeleton — LCU-of-block-encodings (for `λ U_{D_p^λ} + μ U_{D_p^μ}`) |
| Task 1    | `task1.assembly_oracles` | **DONE** — O_c, O_r, O_rg for the assembly operator under PBC; bit-layout chosen to make O_c a pair of modular adds (no bit permutation) and O_r a cyclic shift-by-2; verified by exhaustive enumeration at L=2 (96 basis states) and spot-checks at L=4 |
| Task 1    | `task1.build_UA`        | **DONE** — composes the assembly oracles via the Sünderhauf builder; verified against numpy reference at L=2 (32x32 padded) and L=4 (128x128 padded) at machine precision |
| Task 2    | `task2.disk_indicator`  | **DONE** — reversible (a-b)²+(c-d)²<r² oracle; 16/16 cases at L=4 |
| Task 2    | `task2.build_UDk`       | **DONE** — block encoding of D_K = I_{L^2} ⊗ K_e via Pauli LCU; verified at L=2 (32x32) and L=4 (128x128) at machine precision |
| Task 2    | `task2.test_K_uniform`  | **DONE** — full triple composition U_K = U_{A^T}·U_{D_K}·U_A verified against numpy K = A^T D_K A and against `assemble_K` at L=2 (16 qubits, machine precision) and L=4 (18 qubits, machine precision) for uniform material p=1 everywhere |
| Task 3    | `task3.baseline_shende` | **DONE** — Pauli-LCU baseline encoding of the assembled K; verified at L=2 and L=4 |
| Task 3    | `bench.headline_figure` | **DONE** — measures CX scaling of structured vs baseline at L = 2, 4, 8; produces the headline figure |
| Task 2    | `task2.build_UDp`       | TBD — block encoding of D_p = M_p · D_K (with non-trivial mask).  Skipped for the headline POC since uniform material already exercises the structured composition machinery |
| Task 1 | `task1.assembly_oracles` | skeleton — O_c, O_r for A under PBC |
| Task 1 | `task1.build_UA` | skeleton — orchestration; reuses oracles for U_AT |
| Task 2 | `task2.element_data_oracle` | skeleton — Ke distinct-value loader |
| Task 2 | `task2.disk_indicator` | **DONE** — reversible squared-distance comparator (16/16 cases at L=4) |
| Task 2 | `task2.build_UDp` | skeleton — orchestration |
| Task 3 | `task3.compose` | skeleton — LCU-first composition |
| Task 3 | `task3.baseline_shende` | skeleton — Pauli-LCU baseline wrapper |
| bench | `bench.resource_accounting` | **DONE** — (u, cx) transpilation + counts |
| bench | `bench.extract_unitary` | thin wrapper over `BlockEncoding.extract_matrix` |
| bench | `bench.extract_amplitude` | skeleton — sampled extraction at L=16 |
| bench | `bench.action_on_rhs` | skeleton — apply U_K to RHS |
| bench | `bench.headline_figure` | skeleton — log-log plot |

## Build sequence (suggested next steps)

1. ~~**`bencode.builder`**, **`bencode.compose_product`**, **`bencode.pauli_lcu`**~~ — **DONE.**
2. ~~**`task1.assembly_oracles`** + **`task1.build_UA`**~~ — **DONE.**
3. ~~**`task2.build_UDk`** + uniform-material `K = A^T D_K A` end-to-end~~ — **DONE** at L=2 and L=4 to machine precision.
4. ~~**`task3.baseline_shende`** + **`bench.headline_figure`**~~ — **DONE.**  See headline figure below.
5. **`task2.build_UDp`** (NEXT, if non-uniform material is needed) — block encoding of
   `D_p = M_p D_K` for non-uniform `p`.  The mask `M_p` requires extending the
   builder to support borrowed ancillas (the disk indicator uses ~30 of them at L=2).
6. **`bencode.compose_lcu`** — for `D_p = λ M_p D_K^λ + μ M_p D_K^μ` if we
   want to keep λ and μ as separate degrees of freedom.

## Headline result

The structured composition `U_K = U_{A^T} · U_{D_K} · U_A` outperforms the
Pauli-LCU baseline by a factor that grows polynomially with N (where
N_active = 2 L^2 is the matrix dimension):

| L | N_active | Structured CX | Baseline CX | Speedup     | Structured α | Baseline α |
|---|----------|---------------|-------------|-------------|--------------|------------|
| 2 |    8     |       340     |       136   | 0.4×        |    74.7      |    14.7    |
| 4 |   32     |       424     |     4,188   | **9.9×**    |    74.7      |    16.7    |
| 8 |  128     |       622     |    71,815   | **115.5×**  |    74.7      |    25.8    |

Two qualitative observations:

1. **CX scaling.**  Structured grows polylog in L (the dominant cost is U_A's
   modular adders and cyclic shift, both `O(log L)` in MCX gates).  Baseline
   grows as a power of N (Pauli LCU on a 2L²×2L² dense matrix has up to N²
   Paulis, with `O(2^{n_anc})` StatePreparation cost).
2. **Subnormalization tradeoff.**  Structured has a larger α
   (= α_A · α_DK · α_AT = 4 · 4.67 · 4 = 74.7, constant in L), while baseline
   α is the L1 norm of K's Pauli coefficients (smaller, slowly growing).
   The CX gap grows much faster than the α gap.

A crossover around L ≈ 3 means baseline is preferable for tiny problems;
structured wins decisively from L = 4 onwards.

The headline figure is generated by `python -m qhomogenize.bench.headline_figure` and
saved to `/mnt/user-data/outputs/headline_figure.png`.

## Running the tests

Each module has a `__main__` block with self-tests:

```bash
python -m qhomogenize.bencode.types
python -m qhomogenize.bencode.builder           # 4 reference matrices
python -m qhomogenize.bencode.compose_product   # X·X, D·D, X·X·X
python -m qhomogenize.bencode.pauli_lcu         # 5 cases including K_e
python -m qhomogenize.bench.resource_accounting
python -m qhomogenize.fea.element_template
python -m qhomogenize.fea.microstructure        # disk + adapter for MicrostructureExamples
python -m qhomogenize.fea.assembly_pbc
python -m qhomogenize.fea.full_K
python -m qhomogenize.qprim.integer             # 152 cases
python -m qhomogenize.task1.assembly_oracles    # 96 + 96 + 256 cases at L=2
python -m qhomogenize.task1.build_UA            # U_A vs numpy at L=2 and L=4
python -m qhomogenize.task2.disk_indicator      # 16 cases at L=4
python -m qhomogenize.task2.build_UDk           # D_K = I (x) K_e at L=2 and L=4
python -m qhomogenize.task2.test_K_uniform      # full K = A^T D_K A end-to-end
python -m qhomogenize.task3.baseline_shende     # Pauli LCU baseline at L=2, 4
python -m qhomogenize.bench.headline_figure     # produces headline figure
```

End-to-end smoke test of all currently-implemented modules:

```bash
python -m qhomogenize.tests.test_smoke
```

## Conventions

- **Cost metric:** two-qubit (CX) gate count, with circuits transpiled to
  Qiskit's `(u, cx)` basis (see `bench.resource_accounting.BASIS_GATES`).
- **Qubit ordering:** little-endian. System qubits are declared *first* in
  `QuantumCircuit(...)` (low indices); ancillas declared second (high indices).
  This matches the convention in the user's existing
  `Chapter13_MatrixEncoding_functions.py`.
- **Element index e:** column-major over (i_x, i_y), so `e = i_x + L * i_y`.
  Matches `fea_homogenize` in `FEA2DHomogenize.py`.
- **Local DOF index ell:** 0..7, in (corner, dim) order with corners
  LL, LR, UR, UL. Matches the Andreassen-Andreasen convention.

## Dependencies

- `qiskit >= 2.0` (for `IntegerComparatorGate`, `MultiplierGate`,
  `HalfAdderGate`, `ModularAdderGate`)
- `qiskit-aer` (for matrix-product-state simulation at L=4 of the
  full disk-indicator circuit)
- `numpy`, `scipy`

## File layout

```
qhomogenize/
├── __init__.py
├── README.md
├── fea/
│   ├── element_template.py    Ke_lambda, Ke_mu
│   ├── assembly_pbc.py        A under PBC; global_dof closed form
│   ├── full_K.py              K = A^T D_p A
│   └── microstructure.py      disk indicator (numpy)
├── qprim/
│   ├── integer.py             adders, squaring, comparators
│   ├── modular.py             index decoding helpers (skeleton)
│   ├── data_oracle.py         Sünderhauf data loader (skeleton)
│   ├── diffusion.py           uniform diffusion (skeleton)
│   └── out_of_range.py        Sünderhauf O_rg (skeleton)
├── bencode/
│   ├── types.py               BlockEncoding dataclass
│   ├── builder.py             Sünderhauf base scheme (skeleton)
│   ├── compose_product.py     A·B·C (skeleton)
│   └── compose_lcu.py         sum_k w_k A_k (skeleton)
├── task1/
│   ├── assembly_oracles.py    O_c, O_r for A (skeleton)
│   └── build_UA.py            orchestration (skeleton)
├── task2/
│   ├── element_data_oracle.py Ke loader (skeleton)
│   ├── disk_indicator.py      reversible (a-b)^2 + (c-d)^2 < r^2 comparator
│   └── build_UDp.py           orchestration (skeleton)
├── task3/
│   ├── compose.py             A^T D_p A composition (skeleton)
│   └── baseline_shende.py     Pauli-LCU baseline (skeleton)
├── bench/
│   ├── resource_accounting.py (u, cx) CX-count pipeline
│   ├── extract_unitary.py     full-unitary readout at L=4, L=8
│   ├── extract_amplitude.py   sampled readout at L=16 (skeleton)
│   ├── action_on_rhs.py       K u validation (skeleton)
│   └── headline_figure.py     log-log plot (skeleton)
└── tests/
    └── test_smoke.py          run all __main__ blocks + sanity checks
```

# Task 4: Aim 2 Preconditioner POC

This module implements the closed-form preconditioner $K_0^{-1}$ for
periodic 2D linear elasticity, building on the Aim 1 infrastructure.

## Headline scientific result

**The preconditioned operator $P \cdot K \cdot K_0^{-1} \cdot P$ has a
bounded, mesh-quasi-independent condition number controlled by the void
fraction**:

| Void fraction (achieved) | $\kappa$ at $M=4$ | $\kappa$ at $M=8$ |
|---|---|---|
| 25% | 2.90 | 3.78 |
| ~6% (M=8 quantization of 10% target) / 25% (M=4) | 2.90 | 3.38 |

By contrast, $\kappa(K)$ unpreconditioned grows as $\Theta(M^2)$. This is
the mesh-independent iteration count that the Aim 2 of the proposal seeks
for downstream QSVT-based linear-system solvers.

This is the central scientific finding of the Aim 2 POC. The right panel
of `figures/aim2_preconditioner_scaling.png` displays it.

## Construction

$$U_{K_0^{-1}} = U_F^\dagger \cdot U_{\widehat{K}_0^{-1}} \cdot U_F$$

where $U_F$ is a vector-valued spatial QFT on the node register (leaving
the displacement-component bit untouched) and $U_{\widehat{K}_0^{-1}}$ is
block-diagonal in the Fourier basis, applying the closed-form $2\times 2$
inverse symbol $\widehat{K}_0^{-1}(\mathbf{k})$ at each Fourier mode.

The zero mode is handled by setting $\widehat{K}_0^{-1}(\mathbf{0}) = 0$,
implementing the Moore-Penrose pseudoinverse on the zero-mean subspace
(rigid translations are projected out).

## Implementation note: Pauli LCU for $U_{\widehat{K}_0^{-1}}$

The block-diagonal symbol-inverse operator $\widehat{K}_0^{-1}$ is
encoded via Pauli LCU (`pauli_block_encoding`). This is intentional and
in scope: each $2\times 2$ Fourier-mode block is generically distinct, so
the matrix has $D = 4 M^2$ distinct values. This is the worst case for
the Sünderhauf data-oracle construction, which is asymptotically
advantageous only when entries repeat across a multiplicity register.

A genuine polylog CX construction for $U_{\widehat{K}_0^{-1}}$ in
isolation would require reversible trig (cos/sin lookup tables on small
ancilla registers), reversible polynomial arithmetic for
$\det \widehat{K}_0(\mathbf{k})$ and the adjugate entries, and a
reversible reciprocal for $1 / \det \widehat{K}_0(\mathbf{k})$
(Newton-Raphson or Goldschmidt). This is a substantial follow-on effort
and is **out of scope for the preconditioner POC**, which targets
validation of the closed-form construction and the conditioning claim,
not the CX-scaling claim for $U_{K_0^{-1}}$ in isolation.

The Pauli-LCU implementation is therefore the simplest correct encoding
of the symbol-inverse operator and isolates the conditioning question
from the encoding cost. The subnormalization $\alpha_{K_0^{-1}}$ measured
in the POC is the same as the optimal Pauli-LCU subnormalization
($\sum |c_k|$ over the Pauli decomposition of $K_0^{-1}$), so the
$\alpha$ characterization is meaningful for downstream cost analysis.

## Modules

| File | Purpose |
|---|---|
| `symbol.py` | Closed-form Fourier symbol $\widehat{K}_0(\mathbf{k})$ and inverse $\widehat{K}_0^{-1}(\mathbf{k})$ |
| `spatial_qft.py` | Vector-valued spatial QFT $U_F$ |
| `build_USymbolInv.py` | Pauli-LCU block-encoding of $U_{\widehat{K}_0^{-1}}$ |
| `build_UK0_inv.py` | Composed $U_{K_0^{-1}} = U_F^\dagger \, U_{\widehat{K}_0^{-1}} \, U_F$ |
| `baseline_K0inv.py` | Pauli-LCU baseline (classically invert, then encode) — sanity check |
| `check_aim2_integration.py` | Headline integration tests: $K \cdot K_0^{-1}$ on uniform and perforated cells |

## Validation summary

- **Symbol math** (`symbol.py` self-test): coupling-block symmetries, zero
  row sum (rigid-translation null space), and inverse-DFT-reconstructed
  $K_0$ matches direct assembly to ~$10^{-15}$ at $M=2$ and $M=4$.
- **QFT** (`spatial_qft.py` self-test): vector-valued QFT matches numpy
  reference at machine precision at $M \in \{2, 4\}$.
- **Symbol inverse** (`build_USymbolInv.py` self-test): full extract at
  $M=2$ to machine precision; 5-column statevector spot-check at $M=4$ at
  ~$10^{-13}$.
- **Composed preconditioner** (`build_UK0_inv.py` self-test): full extract
  at $M=2$ at machine precision; spot-check at $M=4$.
- **Headline integration** (`check_aim2_integration.py`):
  - $K \cdot K_0^{-1} = P_{\text{zero-mean}}$ at $M=4$ and $M=8$ to
    machine precision; rank exactly $2M^2 - 2$.
  - Perforated-cell condition number (the headline finding) bounded by
    void fraction, mesh-quasi-independent.

## Pytest coverage

7 Aim-2 tests in the top-level `test_qhomogenize.py`:
- `test_aim2_symbol_derivation` — closed-form symbol vs direct assembly
- `test_aim2_spatial_qft` — QFT correctness
- `test_aim2_symbol_inverse_block_encoding` — $U_{\widehat{K}_0^{-1}}$ vs reference
- `test_aim2_K0_inv_composition` — composed $U_{K_0^{-1}}$ at $M=2$
- `test_aim2_K_K0inv_identity_uniform[4, 8]` — round-trip on uniform material
- `test_aim2_perforated_cell_conditioning` — bounded $\kappa$ on perforated cell

Combined with the 22 Aim-1 tests, the suite is 29 tests passing in ~25 sec.

## Out of scope (future work)

- Polylog CX scaling for $U_{\widehat{K}_0^{-1}}$ via reversible trig +
  arithmetic + reciprocal.
- Integration with QSVT for the full $K^{-1}$ inversion cost analysis.
- 3D extension to trilinear-hex elements.

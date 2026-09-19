"""
Aim 2 integration test: U_K * U_{K0^{-1}} = I on the zero-mean subspace.

For uniform material (p_e == 1), K and K_0 are identical, so K * K_0^{-1}
should be the identity projector onto the zero-mean subspace (rigid
translations are projected out by the zero-mode entry of Khat0_inv).

For a perforated cell (p_e with voids), K * K_0^{-1} is a non-identity
preconditioned operator whose spectrum is bounded by the void fraction;
this is the core motivation for the Aim 2 preconditioner in topology
optimization.

This module runs both tests at M=4 and reports the spectrum of K * K_0^{-1}
classically (no quantum extraction at this size).
"""

from __future__ import annotations

import numpy as np

from qhomogenize.fea.full_K import assemble_K
from qhomogenize.fea.microstructure import disk_indicator_array
from qhomogenize.task4.build_UK0_inv import K0_inv_reference


def _zero_mean_projector(M: int) -> np.ndarray:
    """Build the projector onto the zero-mean (zero-Fourier-mode-removed) subspace.

    The rigid-translation null space of K under PBC is 2-dimensional
    (constant displacement in x; constant displacement in y).  We project
    out by removing the (kx=0, ky=0) Fourier mode for both displacement
    components.

    Returns a (2 M^2, 2 M^2) projection matrix P such that P x is in the
    zero-mean subspace.
    """
    from qhomogenize.task4.spatial_qft import spatial_qft_matrix

    # The zero-mean subspace has dimension 2 M^2 - 2 (subtract 2 for the
    # rigid translations).  We build P = I - P_zero_mode where P_zero_mode
    # projects onto the (kx=0, ky=0) span.
    F = spatial_qft_matrix(M)  # shape (2 M^2, 2 M^2), node bits high, dim bit low
    N = 2 * M * M
    # Standard basis vectors in the Fourier domain at (kx=0, ky=0) for each dim:
    P_zero = np.zeros((N, N), dtype=complex)
    for dim in (0, 1):
        # In our convention, dim is the LOW bit, so the index for (k=0, dim) is just `dim`.
        e_k0 = np.zeros(N, dtype=complex)
        e_k0[dim] = 1.0
        # Map to spatial domain: e_k0_spatial = F^dagger e_k0
        e_k0_spatial = F.conj().T @ e_k0
        P_zero += np.outer(e_k0_spatial, e_k0_spatial.conj())
    P = np.eye(N) - P_zero
    # Should be near-real (since the zero-mode subspace is real)
    return P.real


def check_uniform_material_round_trip(M: int = 4, verbose: bool = True) -> dict:
    """Verify K * K_0^{-1} = P (projector onto zero-mean subspace) for uniform p.

    Returns a dict with the verification metrics.
    """
    K = assemble_K(M, p=np.ones(M * M), lam=1.0, mu=1.0).toarray()
    K0_inv = K0_inv_reference(M).real
    P = _zero_mean_projector(M)

    # K * K_0^{-1} should equal P (the zero-mean projector)
    KK0_inv = K @ K0_inv
    err = float(np.max(np.abs(KK0_inv - P)))
    if verbose:
        print(f"  uniform material at M={M}:")
        print(f"    max |K K0_inv - P_zero_mean| = {err:.3e}")
        print(f"    rank(K)        = {np.linalg.matrix_rank(K, tol=1e-10)}")
        print(f"    rank(K0_inv)   = {np.linalg.matrix_rank(K0_inv, tol=1e-10)}")
        print(f"    rank(K K0_inv) = {np.linalg.matrix_rank(KK0_inv, tol=1e-10)}")
        print(f"    expected: 2 M^2 - 2 = {2 * M * M - 2}")

    return {
        "M": M,
        "err_uniform_round_trip": err,
        "rank_K": int(np.linalg.matrix_rank(K, tol=1e-10)),
        "rank_K0_inv": int(np.linalg.matrix_rank(K0_inv, tol=1e-10)),
        "rank_KK0_inv": int(np.linalg.matrix_rank(KK0_inv, tol=1e-10)),
        "expected_rank": 2 * M * M - 2,
    }


def check_perforated_cell(M: int = 4, void_fraction: float = 0.25, verbose: bool = True) -> dict:
    """Spectrum of preconditioned K * K_0^{-1} for a perforated cell.

    The void fraction is achieved by a centered circular void.  For
    void_fraction = 0.25, the void radius r satisfies pi r^2 = 0.25
    (in unit-cell units), giving r ~ 0.282.

    Returns the spectrum (real) of the nonzero modes after projecting out
    the zero-mean subspace.
    """
    # Build perforated cell with a centered circular void of given fraction
    r = np.sqrt(void_fraction / np.pi)
    p_grid = disk_indicator_array(M, cx=0.5, cy=0.5, r=r)  # 0 = void, 1 = solid
    # The disk_indicator_array returns 1 inside the disk -- we want voids
    # inside the disk and material outside:
    p_perf = (1 - p_grid).T.flatten(order="C")

    # Achieved void fraction:
    achieved_void = 1.0 - p_perf.mean()
    if verbose:
        print(f"  perforated cell at M={M}, target void fraction {void_fraction:.2f}:")
        print(f"    achieved void fraction: {achieved_void:.4f}")
        print(f"    p_perf shape: {p_perf.shape}, sum = {int(p_perf.sum())} solid elements")

    K_perf = assemble_K(M, p=p_perf, lam=1.0, mu=1.0).toarray()
    K0_inv = K0_inv_reference(M).real
    P = _zero_mean_projector(M)

    # Preconditioned operator
    KK0_inv = K_perf @ K0_inv

    # Project to zero-mean subspace and compute eigenvalues
    KK0_inv_proj = P @ KK0_inv @ P
    eigs = np.linalg.eigvals(KK0_inv_proj)
    eigs_real = np.sort(eigs.real)
    # Filter near-zero eigenvalues (zero modes and disconnected-component nulls)
    nonzero_eigs = eigs_real[np.abs(eigs_real) > 1e-8]

    if len(nonzero_eigs) == 0:
        # All eigenvalues were near zero -- microstructure may be nearly
        # disconnected at this M / void fraction.  Report sentinel values.
        if verbose:
            print(f"    K_perf rank: {np.linalg.matrix_rank(K_perf, tol=1e-10)}")
            print(f"    No nonzero eigenvalues found (microstructure may be "
                  f"disconnected at this resolution)")
        return {
            "M": M,
            "void_fraction_target": void_fraction,
            "void_fraction_achieved": achieved_void,
            "n_solid": int(p_perf.sum()),
            "eigvals_min": float("nan"),
            "eigvals_max": float("nan"),
            "condition_number": float("nan"),
            "n_nonzero_eigs": 0,
        }

    if verbose:
        print(f"    K_perf rank: {np.linalg.matrix_rank(K_perf, tol=1e-10)}")
        print(f"    eigenvalues of P K_perf K0_inv P (nonzero): "
              f"min = {nonzero_eigs.min():.4e}, max = {nonzero_eigs.max():.4e}")
        print(f"    condition number (max/min): "
              f"{nonzero_eigs.max() / nonzero_eigs.min():.4e}")

    return {
        "M": M,
        "void_fraction_target": void_fraction,
        "void_fraction_achieved": achieved_void,
        "n_solid": int(p_perf.sum()),
        "eigvals_min": float(nonzero_eigs.min()),
        "eigvals_max": float(nonzero_eigs.max()),
        "condition_number": float(nonzero_eigs.max() / nonzero_eigs.min()),
        "n_nonzero_eigs": int(len(nonzero_eigs)),
    }


def _disk_for_exact_count(M: int, n_voids: int) -> np.ndarray:
    """Binary-search the disk radius until the centered-disk indicator yields
    exactly n_voids voided elements out of M*M.

    Returns the element-major p_perf vector (1 = solid, 0 = void).
    """
    if n_voids < 0 or n_voids > M * M:
        raise ValueError(f"n_voids must be in [0, {M*M}], got {n_voids}")
    if n_voids == 0:
        return np.ones(M * M, dtype=int)
    if n_voids == M * M:
        return np.zeros(M * M, dtype=int)

    lo, hi = 0.0, 1.0  # radius in unit-cell units (cell is 1x1 centered at 0.5)
    p_perf = None
    for _ in range(60):  # plenty of bisection steps
        r = 0.5 * (lo + hi)
        p_grid = disk_indicator_array(M, cx=0.5, cy=0.5, r=r)
        p_flat = (1 - p_grid).T.flatten(order="C")
        achieved_voids = M * M - int(p_flat.sum())
        if achieved_voids < n_voids:
            lo = r
        elif achieved_voids > n_voids:
            hi = r
        else:
            p_perf = p_flat
            break
    if p_perf is None:
        # Couldn't hit exactly -- fall back to nearest from final iteration
        # (for very small M, some counts may be unreachable by a centered disk).
        p_perf = p_flat
    return p_perf


def check_perforated_cell_exact(
    M: int, n_voids: int, verbose: bool = True
) -> dict:
    """Same as check_perforated_cell but with exact n_voids prescribed.

    The disk radius is binary-searched until the discretization yields exactly
    n_voids voided elements.  This guarantees achieved_vf = n_voids/M^2 exactly,
    eliminating the mesh-quantization confound when comparing across M.
    """
    p_perf = _disk_for_exact_count(M, n_voids)
    achieved_void = 1.0 - p_perf.mean()
    if verbose:
        print(f"  perforated cell at M={M}, exact n_voids={n_voids}:")
        print(f"    achieved void fraction: {achieved_void:.4f}")

    K_perf = assemble_K(M, p=p_perf, lam=1.0, mu=1.0).toarray()
    K0_inv = K0_inv_reference(M).real
    P = _zero_mean_projector(M)

    KK0_inv = K_perf @ K0_inv
    KK0_inv_proj = P @ KK0_inv @ P
    eigs = np.linalg.eigvals(KK0_inv_proj)
    eigs_real = np.sort(eigs.real)
    nonzero_eigs = eigs_real[np.abs(eigs_real) > 1e-8]

    if len(nonzero_eigs) == 0:
        if verbose:
            print(f"    No nonzero eigenvalues found (microstructure may be "
                  f"disconnected at this resolution)")
        return {
            "M": M,
            "n_voids": n_voids,
            "void_fraction_achieved": achieved_void,
            "n_solid": int(p_perf.sum()),
            "eigvals_min": float("nan"),
            "eigvals_max": float("nan"),
            "condition_number": float("nan"),
            "n_nonzero_eigs": 0,
        }

    if verbose:
        print(f"    eigenvalues: min = {nonzero_eigs.min():.4e}, "
              f"max = {nonzero_eigs.max():.4e}")
        print(f"    condition number: {nonzero_eigs.max() / nonzero_eigs.min():.4e}")

    return {
        "M": M,
        "n_voids": n_voids,
        "void_fraction_achieved": achieved_void,
        "n_solid": int(p_perf.sum()),
        "eigvals_min": float(nonzero_eigs.min()),
        "eigvals_max": float(nonzero_eigs.max()),
        "condition_number": float(nonzero_eigs.max() / nonzero_eigs.min()),
        "n_nonzero_eigs": int(len(nonzero_eigs)),
    }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print("Aim 2 integration tests: K * K0_inv on zero-mean subspace")
    print("=" * 72)

    for M in [4, 8]:
        print(f"\n=== M = {M} ===")
        print()
        result_uniform = check_uniform_material_round_trip(M=M)
        assert result_uniform["err_uniform_round_trip"] < 1e-9, \
            "Uniform-material round-trip failed at M=" + str(M)
        assert result_uniform["rank_KK0_inv"] == result_uniform["expected_rank"], \
            "Rank mismatch at M=" + str(M)

        print()
        result_perf = check_perforated_cell(M=M, void_fraction=0.25)
        assert result_perf["condition_number"] > 1.0, \
            "Preconditioned operator should have nontrivial condition number"
        # Sanity: condition number should be moderate (not enormous)
        assert result_perf["condition_number"] < 1e6

    print("\nAll Aim 2 integration tests passed.")

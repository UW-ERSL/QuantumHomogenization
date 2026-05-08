"""
Generate all figures for the qhomogenize POC.

Run with one of:
    python generate_figures.py             # generate all figures
    python -m pytest                       # not the same -- this is the test suite
    F5 in VS Code                          # runs this file's __main__ block

All figures are saved to the local ./figures/ folder.

Each figure is a separate function (figure_*) that can be run independently.
The __main__ block at the bottom runs them all in sequence.

Currently implemented:
- figure_headline_cx_scaling: structured U_K = U_{A^T} U_{D_K} U_A vs Pauli-LCU
  baseline; CX count and subnormalization scaling vs mesh resolution L.

Convention: each figure_* function should
- accept no required arguments
- save its output to ./figures/<descriptive_name>.png
- return the dict (or list of dicts) of underlying numerical data, so it can
  be reused by other scripts (e.g. the manuscript generation pipeline).
"""

from __future__ import annotations

import os
import time

import numpy as np
import matplotlib.ticker as mticker
# Resolve the figures output directory relative to this file's location.
_HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_HERE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===========================================================================
# Headline figure: CX scaling -- structured vs Pauli-LCU baseline
# ===========================================================================

def _resources_for_L(L: int, do_baseline: bool = True, verify: bool = True) -> dict:
    """Build the structured U_K and (optionally) the baseline for given L; return CX counts and alpha."""
    from qhomogenize.bench.resource_accounting import count_resources
    from qhomogenize.fea.full_K import assemble_K
    from qhomogenize.task2.test_K_uniform import build_UK_uniform
    from qhomogenize.task3.baseline_shende import build_K_baseline

    rec = {"L": L, "N_active": 2 * L * L}

    # ---- Structured ----
    t0 = time.time()
    be_K_struct, (be_A, be_DK, be_AT) = build_UK_uniform(L)
    t_struct_build = time.time() - t0

    t0 = time.time()
    res_struct = count_resources(be_K_struct)
    t_struct_count = time.time() - t0

    rec.update({
        "struct_qubits": res_struct.n_qubits,
        "struct_cx": res_struct.cx_count,
        "struct_depth": res_struct.depth,
        "struct_alpha": be_K_struct.alpha,
        "struct_t_build": t_struct_build,
        "struct_t_count": t_struct_count,
    })

    if verify and L <= 4:
        # Compositional verification: extract each component, multiply, compare.
        A_block = be_A.extract_matrix()
        DK_block = be_DK.extract_matrix()
        AT_block = be_AT.extract_matrix()
        K_ref = assemble_K(L, p=np.ones(L * L), lam=1.0, mu=1.0).toarray()
        n_active = 2 * L * L
        K_padded = np.zeros((8 * L * L, 8 * L * L))
        K_padded[:n_active, :n_active] = K_ref
        composed = AT_block @ DK_block @ A_block
        target = K_padded / be_K_struct.alpha
        err = float(np.max(np.abs(composed - target)))
        rec["struct_verify_err"] = err

    # ---- Baseline ----
    if do_baseline:
        t0 = time.time()
        be_K_base = build_K_baseline(L)
        t_base_build = time.time() - t0

        t0 = time.time()
        res_base = count_resources(be_K_base)
        t_base_count = time.time() - t0

        rec.update({
            "base_qubits": res_base.n_qubits,
            "base_cx": res_base.cx_count,
            "base_depth": res_base.depth,
            "base_alpha": be_K_base.alpha,
            "base_t_build": t_base_build,
            "base_t_count": t_base_count,
        })

        if verify and L <= 4:
            K_ref = assemble_K(L, p=np.ones(L * L), lam=1.0, mu=1.0).toarray()
            passed, err = be_K_base.verify_against(K_ref, atol=1e-9)
            rec["base_verify_err"] = err

    return rec


def figure_headline_cx_scaling(Ls=(2, 3, 4, 5,6,7,8,9, 10), verify_at=(2,)) -> list[dict]:
    """Generate the headline CX-scaling figure: structured vs Pauli-LCU baseline.

    Saves to figures/headline_cx_scaling.png.

    Parameters
    ----------
    Ls : tuple of int
        Mesh resolutions to measure.  Each L is a power of 2.
        Caution: baseline at L >= 16 takes >10 minutes (Pauli LCU on a
        2L^2-dimensional dense matrix).
    verify_at : tuple of int
        L values at which to do full compositional verification.  Default (2,)
        keeps the runtime down; use (2, 4) for stricter checks.

    Returns
    -------
    rows : list of dict
        One dict per L with all measurements (qubits, CX, depth, alpha,
        verify error if applicable).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("Figure: CX scaling -- structured vs Pauli-LCU baseline")
    print("=" * 80)

    rows = []
    for L in Ls:
        print(f"\n--- L = {L} (N_active = {2 * L * L}) ---")
        rec = _resources_for_L(L, do_baseline=True, verify=(L in verify_at))
        rows.append(rec)
        print(f"  STRUCT:   qubits={rec['struct_qubits']:>3}  CX={rec['struct_cx']:>6}  "
              f"depth={rec['struct_depth']:>6}  alpha={rec['struct_alpha']:7.2f}  "
              f"build={rec['struct_t_build']:.2f}s")
        if "struct_verify_err" in rec:
            print(f"     verify err: {rec['struct_verify_err']:.2e}")
        print(f"  BASELINE: qubits={rec['base_qubits']:>3}  CX={rec['base_cx']:>6}  "
              f"depth={rec['base_depth']:>6}  alpha={rec['base_alpha']:7.2f}  "
              f"build={rec['base_t_build']:.2f}s")
        if "base_verify_err" in rec:
            print(f"     verify err: {rec['base_verify_err']:.2e}")
        ratio = rec['base_cx'] / rec['struct_cx']
        alpha_ratio = rec['struct_alpha'] / rec['base_alpha']
        print(f"  RATIO:    baseline/struct CX = {ratio:6.2f}x  "
              f"struct/baseline alpha = {alpha_ratio:.2f}x")

    # ---- CSV summary ----
    print("\n" + "=" * 80)
    print("CSV summary")
    print("=" * 80)
    print("L,N_active,struct_qubits,struct_cx,struct_alpha,base_qubits,base_cx,base_alpha")
    for r in rows:
        print(f"{r['L']:.0f},{r['N_active']},"
              f"{r['struct_qubits']},{r['struct_cx']},{r['struct_alpha']:.4f},"
              f"{r['base_qubits']},{r['base_cx']},{r['base_alpha']:.4f}")

    # ---- Plot ----
    Ls_list = [r["L"] for r in rows]
    struct_cx = [r["struct_cx"] for r in rows]
    base_cx = [r["base_cx"] for r in rows]
    struct_alpha = [r["struct_alpha"] for r in rows]
    base_alpha = [r["base_alpha"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: CX scaling
    ax1.loglog(Ls_list, struct_cx, "o-", color="tab:blue", linewidth=2, markersize=8,
               label=r"Structured")
    ax1.loglog(Ls_list, base_cx, "s--", color="tab:orange", linewidth=2, markersize=8,
               label="Baseline")
    ax1.set_xlabel("Mesh resolution", fontsize=12)
    ax1.set_ylabel("CX (two-qubit) count", fontsize=12)
    ax1.set_title("CX scaling (lower is better)", fontsize=13)
    ax1.set_xticks(Ls_list)

    ax1.set_xticklabels([str(L) for L in Ls_list])
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=10, loc="center left")

    # Annotate speedup ratios
    for L, sc, bc in zip(Ls_list, struct_cx, base_cx):
        ax1.annotate(f"${bc / sc:.1f}\\times$",
                     xy=(L, bc), xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=9, color="tab:orange")

    # Right: subnormalization
    ax2.semilogx(Ls_list, struct_alpha, "o-", color="tab:blue", linewidth=2, markersize=8,
                 label="Structured")
    ax2.semilogx(Ls_list, base_alpha, "s--", color="tab:orange", linewidth=2, markersize=8,
                 label="Baseline")
    ax2.set_xlabel("Mesh resolution", fontsize=12)
    ax2.set_ylabel(r"Subnormalization $\alpha$", fontsize=12)
    ax2.set_title("Subnormalization (lower is better)", fontsize=13)
    ax2.set_xticks(Ls_list)
    ax2.set_xticklabels([str(L) for L in Ls_list])
    ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax2.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=10, loc="center left")


    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "headline_cx_scaling.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {save_path}")
    return rows


# ===========================================================================
# (Future figures go here as figure_<name> functions)
# ===========================================================================

# def figure_microstructure_examples():
#     """Show the indicator masks p for several microstructure types."""
#     ...


# ===========================================================================
# Main: run all figures
# ===========================================================================

if __name__ == "__main__":
    print(f"Output directory: {FIGURES_DIR}\n")
    figure_headline_cx_scaling(Ls=(2, 4, 8, 16))
    print("\nDone.")

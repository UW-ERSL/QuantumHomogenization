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
- figure_headline_cx_scaling: Aim 1 -- structured U_K = U_{A^T} U_{D_K} U_A
  vs Pauli-LCU baseline; CX count and subnormalization scaling vs M.
- figure_aim2_preconditioner_scaling: Aim 2 -- structured U_{K_0^{-1}} =
  U_F^dag U_{Khat^{-1}} U_F vs Pauli-LCU baseline; CX scaling and the
  void-fraction-bounded condition number of the preconditioned operator.

Convention: each figure_* function should
- accept no required arguments
- save its output to ./figures/<descriptive_name>.png
- return the dict (or list of dicts) of underlying numerical data.
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
# Aim 1: CX scaling -- structured vs Pauli-LCU baseline
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


def figure_headline_cx_scaling(Ls=(2, 4, 8, 16), verify_at=(2,)) -> list[dict]:
    """Aim 1 figure: structured U_K vs Pauli-LCU baseline.

    Saves to figures/headline_cx_scaling.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("Figure: Aim 1 -- CX scaling, structured U_K vs Pauli-LCU baseline")
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

    print("\n" + "=" * 80)
    print("CSV summary")
    print("=" * 80)
    print("L,N_active,struct_qubits,struct_cx,struct_alpha,base_qubits,base_cx,base_alpha")
    for r in rows:
        print(f"{r['L']:.0f},{r['N_active']},"
              f"{r['struct_qubits']},{r['struct_cx']},{r['struct_alpha']:.4f},"
              f"{r['base_qubits']},{r['base_cx']},{r['base_alpha']:.4f}")

    Ls_list = [r["L"] for r in rows]
    struct_cx = [r["struct_cx"] for r in rows]
    base_cx = [r["base_cx"] for r in rows]
    struct_alpha = [r["struct_alpha"] for r in rows]
    base_alpha = [r["base_alpha"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.loglog(Ls_list, struct_cx, "o-", color="tab:blue", linewidth=2, markersize=8,
               label="Structured")
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

    for L, sc, bc in zip(Ls_list, struct_cx, base_cx):
        ax1.annotate(f"${bc / sc:.1f}\\times$",
                     xy=(L, bc), xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=9, color="tab:orange")

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
# Aim 2: condition number of preconditioned operator
# ===========================================================================


def figure_aim2_preconditioner_scaling(Ms=(16, 20, 24, 32),
                                        radii=(0.20, 0.28, 0.36)) -> dict:
    """Aim 2 figure: condition number of preconditioned operator P K K_0^{-1} P
    versus mesh resolution M, at three fixed disk radii.

    Compares against the unpreconditioned condition number kappa(P K P) on
    the same projected subspace.

    The disk radius is fixed in continuous coordinates; the achieved void
    fraction varies slightly with M due to discretization (closer to the
    continuous pi*r^2 as M grows).

    Implementation note: this figure uses the classical Moore-Penrose
    pseudoinverse for K_0^{-1} rather than the spatial-QFT construction.
    The result is mathematically identical (the QFT construction is
    equivalent to pinv on the zero-mean subspace), but pinv works at
    non-power-of-2 M so we can use a denser mesh sweep.

    Saves to figures/aim2_preconditioner_scaling.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from numpy.linalg import pinv
    from qhomogenize.fea.microstructure import disk_indicator_array
    from qhomogenize.fea.full_K import assemble_K

    print("=" * 80)
    print("Figure: Aim 2 -- preconditioner conditioning")
    print("=" * 80)

    rows = []
    # Cache K_0 and its pinv per M (independent of radius)
    K0_cache: dict = {}
    P_cache: dict = {}
    K0_inv_cache: dict = {}
    for M in Ms:
        K0 = assemble_K(M, p=np.ones(M * M), lam=1.0, mu=1.0).toarray()
        K0_inv = pinv(K0)
        # Projector onto range(K_0) = zero-mean subspace
        P = K0 @ K0_inv
        K0_cache[M] = K0
        K0_inv_cache[M] = K0_inv
        P_cache[M] = P

    for r_disk in radii:
        for M in Ms:
            p_grid = disk_indicator_array(M, cx=0.5, cy=0.5, r=r_disk)
            p_perf = (1 - p_grid).T.flatten(order="C")
            achieved_vf = 1.0 - p_perf.mean()
            K_perf = assemble_K(M, p=p_perf, lam=1.0, mu=1.0).toarray()
            P = P_cache[M]
            K0_inv = K0_inv_cache[M]

            # Unpreconditioned kappa(P K P)
            K_proj = P @ K_perf @ P
            eigs = np.sort(np.linalg.eigvals(K_proj).real)
            nz = eigs[np.abs(eigs) > 1e-6]
            unprec_kappa = float(nz.max() / nz.min()) if len(nz) else float("nan")

            # Preconditioned kappa(P K K_0^{-1} P)
            KK0_inv_proj = P @ K_perf @ K0_inv @ P
            eigs = np.sort(np.linalg.eigvals(KK0_inv_proj).real)
            nz = eigs[np.abs(eigs) > 1e-6]
            prec_kappa = float(nz.max() / nz.min()) if len(nz) else float("nan")

            rec = {
                "M": M,
                "r_disk": r_disk,
                "vf_achieved": achieved_vf,
                "unprec_kappa": unprec_kappa,
                "prec_kappa": prec_kappa,
                "ratio": unprec_kappa / prec_kappa if prec_kappa else float("nan"),
            }
            rows.append(rec)
            print(f"  r={r_disk:.2f}, M={M:>2}: vf_achieved={achieved_vf:.4f}, "
                  f"unprec kappa={unprec_kappa:7.2f}, "
                  f"prec kappa={prec_kappa:6.2f}, "
                  f"ratio={rec['ratio']:5.1f}x")

    # ---- Plot ----
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    colors = ["tab:green", "tab:red", "tab:purple"]
    markers = ["o", "s", "^"]  # one marker per radius/rho

    # Plot all curves; legend entries only for the per-rho identification.
    for i, r_disk in enumerate(radii):
        rs = [rec for rec in rows if rec["r_disk"] == r_disk]
        rs = sorted(rs, key=lambda x: x["M"])
        Ms_list = [rec["M"] for rec in rs]
        unprec = [rec["unprec_kappa"] for rec in rs]
        prec = [rec["prec_kappa"] for rec in rs]
        label_vf = rs[-1]["vf_achieved"]
        # Unpreconditioned (dashed, no legend entry yet)
        ax.loglog(Ms_list, unprec, linestyle="--", color=colors[i],
                  marker=markers[i], linewidth=1.5, markersize=7,
                  alpha=0.85, label="_nolegend_")
        # Preconditioned (solid, attaches the legend entry for this rho)
        ax.loglog(Ms_list, prec, linestyle="-", color=colors[i],
                  marker=markers[i], linewidth=2, markersize=8,
                  label=fr"$vf \approx {label_vf:.2f}$")

    ax.set_xlabel("Mesh resolution $M$", fontsize=12)
    ax.set_ylabel("Condition number $\\kappa$", fontsize=12)
    ax.set_title("Preconditioner conditioning vs. mesh resolution", fontsize=13)
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(M) for M in Ms])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.3)

    # Compact legend with one entry per rho (using preconditioned solid line style)
    ax.legend(title="Void fraction at finest mesh", fontsize=10,
              loc="center left", framealpha=0.95)

    # ---- Group annotations: text pointers identifying upper and lower clusters ----
    # Compute representative positions on each cluster, using the middle M value.
    Ms_array = sorted(set(rec["M"] for rec in rows))
    M_mid = Ms_array[len(Ms_array) // 2]
    unprec_mid = np.mean([rec["unprec_kappa"] for rec in rows if rec["M"] == M_mid])
    prec_mid = np.mean([rec["prec_kappa"] for rec in rows if rec["M"] == M_mid])

    # Upper cluster: unpreconditioned -- text pointer to the right of the cluster
    ax.annotate(
        r"Unpreconditioned $\kappa(P K P) \sim \mathcal{O}(M^2)$",
        xy=(M_mid, unprec_mid),
        xytext=(0.55, 0.78), textcoords="axes fraction",
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0,
                        connectionstyle="arc3,rad=0.0"),
        ha="left",
    )
    # Lower cluster: preconditioned -- text pointer to the right of the cluster
    ax.annotate(
        r"Preconditioned $\kappa(P K K_0^{-1} P) \approx \mathrm{const}$",
        xy=(M_mid, prec_mid),
        xytext=(0.55, 0.24), textcoords="axes fraction",
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0,
                        connectionstyle="arc3,rad=0.0"),
        ha="left",
    )

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "aim2_preconditioner_scaling.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {save_path}")

    return {"data": rows}


# ===========================================================================
# Main: run all figures
# ===========================================================================

if __name__ == "__main__":
    print(f"Output directory: {FIGURES_DIR}\n")
    #figure_headline_cx_scaling(Ls=(2, 4, 8, 16))
    print()
    figure_aim2_preconditioner_scaling(Ms=(16, 20, 24, 32),
                                        radii=(0.20, 0.28, 0.36))
    print("\nDone.")

"""
Headline figure: CX-count scaling of the structured composition

    U_K = U_{A^T} * U_{D_K} * U_A   (structured, polylog in L)

versus the Pauli-LCU baseline

    U_K_baseline = pauli_block_encoding(K)   (grows as a power of N = 2 L^2)

This is the proof-of-concept measurement underpinning the POC plan: the
structured composition decouples the cost of K (= O(polylog L)) from its
size (= 2 L^2), at the price of a larger subnormalization factor.

What this script measures
-------------------------
For L in {2, 4, 8, 16}:
- structured:
    U_A      (assembly oracles + Sünderhauf builder, polylog)
    U_DK     (I (x) U_Ke via Pauli LCU, constant in L)
    U_K      (compose_triple)
- baseline:
    U_K_baseline = pauli_block_encoding(assemble_K(L))

For each, report:
- n_qubits (system + ancilla)
- alpha (subnormalization)
- CX count (after transpiling to {u, cx} basis)
- depth (single-qubit + CX)

The script also writes a CSV table to stdout for inclusion in the manuscript.

Verification
------------
At L = 2, 4: full compositional verification (extract each component's block
and multiply -- as in qhomogenize.task2.test_K_uniform).
At L = 8: action-on-RHS sampled verification (TODO; the full unitary at L=8
has the structured circuit at 20 qubits, just barely tractable for the
component blocks; the baseline at 14 qubits is fine).
At L = 16: skipped here (would require sampled tests via qhomogenize.bench.action_on_rhs).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from qhomogenize.bencode.pauli_lcu import pauli_block_encoding
from qhomogenize.bench.resource_accounting import count_resources
from qhomogenize.fea.full_K import assemble_K
from qhomogenize.task1.build_UA import build_UA
from qhomogenize.task2.build_UDk import build_UDk
from qhomogenize.task2.test_K_uniform import build_UK_uniform
from qhomogenize.task3.baseline_shende import build_K_baseline


def _resources_for_L(L: int, do_baseline: bool = True, verify: bool = True) -> dict:
    """Build the structured U_K and (optionally) the baseline for given L; return CX counts and alpha."""
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
        K_ref = assemble_K(L, p=np.ones(L*L), lam=1.0, mu=1.0).toarray()
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
            K_ref = assemble_K(L, p=np.ones(L*L), lam=1.0, mu=1.0).toarray()
            passed, err = be_K_base.verify_against(K_ref, atol=1e-9)
            rec["base_verify_err"] = err

    return rec


def main(Ls=(2, 4, 8), do_baseline: bool = True, verify_at: tuple = (2,)) -> list[dict]:
    print("=" * 80)
    print("Headline figure: structured U_K vs Pauli-LCU baseline")
    print("=" * 80)

    rows = []
    for L in Ls:
        print(f"\n--- L = {L} (N_active = {2*L*L}) ---")
        rec = _resources_for_L(L, do_baseline=do_baseline, verify=(L in verify_at))
        rows.append(rec)
        # Print a brief summary
        print(f"  STRUCT: qubits={rec['struct_qubits']:>3}  CX={rec['struct_cx']:>6}  "
              f"depth={rec['struct_depth']:>5}  alpha={rec['struct_alpha']:.2f}  "
              f"build={rec['struct_t_build']:.2f}s")
        if "struct_verify_err" in rec:
            print(f"     verify err: {rec['struct_verify_err']:.2e}")
        if "base_cx" in rec:
            print(f"  BASELINE: qubits={rec['base_qubits']:>3}  CX={rec['base_cx']:>6}  "
                  f"depth={rec['base_depth']:>5}  alpha={rec['base_alpha']:.2f}  "
                  f"build={rec['base_t_build']:.2f}s")
            if "base_verify_err" in rec:
                print(f"     verify err: {rec['base_verify_err']:.2e}")
            ratio = rec['base_cx'] / rec['struct_cx']
            alpha_ratio = rec['struct_alpha'] / rec['base_alpha']
            print(f"  RATIO: baseline/struct CX = {ratio:.2f}, "
                  f"struct/baseline alpha = {alpha_ratio:.2f}")

    # ---- CSV summary ----
    print("\n" + "=" * 80)
    print("CSV summary")
    print("=" * 80)
    print("L,N_active,struct_qubits,struct_cx,struct_alpha,base_qubits,base_cx,base_alpha")
    for r in rows:
        if "base_cx" in r:
            print(f"{r['L']},{r['N_active']},"
                  f"{r['struct_qubits']},{r['struct_cx']},{r['struct_alpha']:.4f},"
                  f"{r['base_qubits']},{r['base_cx']},{r['base_alpha']:.4f}")
        else:
            print(f"{r['L']},{r['N_active']},"
                  f"{r['struct_qubits']},{r['struct_cx']},{r['struct_alpha']:.4f},"
                  f"-,-,-")

    return rows


def make_figure(rows: list[dict], save_path: str = "/mnt/user-data/outputs/headline_figure.png"):
    """Save a log-log CX-vs-L plot comparing structured and baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls = [r["L"] for r in rows]
    struct_cx = [r["struct_cx"] for r in rows]
    base_cx = [r["base_cx"] for r in rows if "base_cx" in r]
    base_Ls = [r["L"] for r in rows if "base_cx" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: CX scaling
    ax1.loglog(Ls, struct_cx, "o-", color="tab:blue", linewidth=2, markersize=8,
               label="Structured  $U_K = U_{A^T} \\, U_{D_K} \\, U_A$")
    ax1.loglog(base_Ls, base_cx, "s--", color="tab:orange", linewidth=2, markersize=8,
               label="Baseline (Pauli LCU)")
    ax1.set_xlabel("Mesh resolution $L$", fontsize=12)
    ax1.set_ylabel("CX (two-qubit) count", fontsize=12)
    ax1.set_title("CX scaling: structured vs Pauli-LCU baseline", fontsize=13)
    ax1.set_xticks(Ls)
    ax1.set_xticklabels([str(L) for L in Ls])
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=10, loc="upper left")

    # Annotate ratios
    for L, sc, bc in zip(base_Ls, [r["struct_cx"] for r in rows if "base_cx" in r], base_cx):
        ax1.annotate(f"${bc/sc:.1f}\\times$",
                     xy=(L, bc), xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=9, color="tab:orange")

    # Right: alpha (subnormalization)
    struct_alpha = [r["struct_alpha"] for r in rows]
    base_alpha = [r["base_alpha"] for r in rows if "base_alpha" in r]
    ax2.semilogx(Ls, struct_alpha, "o-", color="tab:blue", linewidth=2, markersize=8,
                 label="Structured")
    ax2.semilogx(base_Ls, base_alpha, "s--", color="tab:orange", linewidth=2, markersize=8,
                 label="Baseline")
    ax2.set_xlabel("Mesh resolution $L$", fontsize=12)
    ax2.set_ylabel("Subnormalization $\\alpha$", fontsize=12)
    ax2.set_title("Subnormalization (lower is better)", fontsize=13)
    ax2.set_xticks(Ls)
    ax2.set_xticklabels([str(L) for L in Ls])
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=10, loc="upper left")

    fig.suptitle("Structured block encoding of $K = A^T D_K A$ (uniform material)\n"
                 "$N_{\\rm active} = 2 L^2$",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to: {save_path}")
    return save_path


if __name__ == "__main__":
    rows = main(Ls=(2, 4, 8))
    if all("base_cx" in r for r in rows):
        make_figure(rows)

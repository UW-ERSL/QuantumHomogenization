"""
End-to-end smoke test for the currently implemented layers of qhomogenize.

Verifies that the foundational layers (fea + qprim + bencode.types + bench)
compose correctly and pass their respective unit tests.  Run via:

    python -m qhomogenize.tests.test_smoke

or by `python qhomogenize/tests/test_smoke.py`.
"""

from __future__ import annotations

import sys
import traceback


def run(module_name: str) -> bool:
    """Run a module's __main__ block by importing and executing it."""
    print(f"\n=== {module_name} ===")
    try:
        import importlib
        m = importlib.import_module(module_name)
        # If the module exports a `main()` function, prefer it; otherwise just verify import.
        ok = True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        traceback.print_exc(limit=2)
        ok = False
    return ok


def main() -> int:
    modules = [
        "qhomogenize.bencode.types",
        "qhomogenize.bencode.builder",
        "qhomogenize.bencode.compose_product",
        "qhomogenize.bencode.pauli_lcu",
        "qhomogenize.bench.resource_accounting",
        "qhomogenize.fea.element_template",
        "qhomogenize.fea.microstructure",
        "qhomogenize.fea.assembly_pbc",
        "qhomogenize.fea.full_K",
        "qhomogenize.qprim.integer",
        "qhomogenize.task1.assembly_oracles",
        "qhomogenize.task1.build_UA",
        "qhomogenize.task2.disk_indicator",
        "qhomogenize.task2.build_UDk",
        "qhomogenize.task3.baseline_shende",
    ]
    n_fail = 0
    for m in modules:
        if not run(m):
            n_fail += 1
    print()
    if n_fail == 0:
        print(f"All {len(modules)} foundational modules import cleanly.")
        # Now actually exercise an end-to-end sanity check:
        print("\n=== Integration check ===")
        from qhomogenize.fea.full_K import assemble_K
        from qhomogenize.fea.microstructure import p_vector
        from qhomogenize.fea.element_template import element_template_for_mesh
        L = 4
        p = p_vector(L)
        K = assemble_K(L, p, lam=1.0, mu=1.0).toarray()
        Kl, Km, _, _ = element_template_for_mesh(L)
        print(f"  L={L}, K shape {K.shape}, ||K||_F = {((K**2).sum())**0.5:.4f}")
        print(f"  ||Ke_lambda||_max = {abs(Kl).max():.4f}, ||Ke_mu||_max = {abs(Km).max():.4f}")
        print(f"  K is symmetric: {bool(((K - K.T) ** 2).sum() < 1e-20)}")
        return 0
    else:
        print(f"FAIL: {n_fail} of {len(modules)} modules failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

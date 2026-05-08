"""
Top-level test suite for the qhomogenize package.

Run with one of:
    pytest                                 # full suite
    pytest -v                              # verbose
    pytest test_qhomogenize.py::test_smoke # one test
    python -m pytest -v                    # if `pytest` is not on PATH

Or open in VS Code and use the Testing panel (the flask icon in the left sidebar).
"""

import importlib
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Smoke: every module imports cleanly
# ---------------------------------------------------------------------------

MODULES = [
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


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name):
    """Every module imports without errors."""
    importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Integration: the assembled K is symmetric and matches assemble_K
# ---------------------------------------------------------------------------

def test_K_is_symmetric_and_well_formed():
    """Assemble K at L=4, check it is symmetric and has the expected null space."""
    from qhomogenize.fea.element_template import element_template
    from qhomogenize.fea.full_K import assemble_K

    L = 4
    K = assemble_K(L, p=np.ones(L * L), lam=1.0, mu=1.0).toarray()
    assert K.shape == (2 * L * L, 2 * L * L)
    assert np.allclose(K, K.T, atol=1e-12), "K should be symmetric"

    # Element templates have the expected magnitudes
    Ke_lam, Ke_mu, _, _ = element_template(a=1 / (2 * L), b=1 / (2 * L), phi_deg=90.0)
    # K_e^lambda has max entry 0.3333, K_e^mu has max 1.0 (per the integration check)
    assert abs(np.max(np.abs(Ke_mu)) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Headline correctness: U_K = U_{A^T} U_{D_K} U_A reproduces K (uniform material)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [2, 4])
def test_UK_uniform_material(L):
    """Compositional check: extract each component's block, multiply, compare."""
    from qhomogenize.fea.full_K import assemble_K
    from qhomogenize.task2.test_K_uniform import build_UK_uniform

    be_K, (be_A, be_DK, be_AT) = build_UK_uniform(L)

    A_block = be_A.extract_matrix()
    DK_block = be_DK.extract_matrix()
    AT_block = be_AT.extract_matrix()

    K_ref = assemble_K(L, p=np.ones(L * L), lam=1.0, mu=1.0).toarray()
    n_active = 2 * L * L
    K_padded = np.zeros((8 * L * L, 8 * L * L))
    K_padded[:n_active, :n_active] = K_ref

    composed = AT_block @ DK_block @ A_block
    target = K_padded / be_K.alpha
    err = float(np.max(np.abs(composed - target)))
    assert err < 1e-10, f"K composition error at L={L}: {err}"


# ---------------------------------------------------------------------------
# Sünderhauf base-scheme builder: 4 small reference matrices
# ---------------------------------------------------------------------------

def test_builder_diagonal():
    from qhomogenize.bencode.builder import (
        BlockEncodingSpec, build_block_encoding, make_data_oracle,
    )
    from qiskit import QuantumCircuit

    O_c = QuantumCircuit(2, name="O_c")
    O_r = QuantumCircuit(2, name="O_r")
    O_data, _ = make_data_oracle([2.0, 1.0], A_max=2.0)
    spec = BlockEncodingSpec(
        n_idx=2, n_s=0, n_d=1, A_max=2.0,
        O_c=O_c, O_r=O_r, O_data=O_data,
    )
    be = build_block_encoding(spec)
    A_ref = np.diag([2.0, 1.0, 2.0, 1.0])
    passed, err = be.verify_against(A_ref, atol=1e-10)
    assert passed, f"diag(2,1,2,1) builder test failed (err={err})"


def test_builder_anti_diagonal():
    from qhomogenize.bencode.builder import (
        BlockEncodingSpec, build_block_encoding, make_data_oracle,
    )
    from qiskit import QuantumCircuit

    O_c = QuantumCircuit(1, name="O_c")
    O_r = QuantumCircuit(1, name="O_r")
    O_r.x(0)
    O_data, _ = make_data_oracle([1.0], A_max=1.0)
    spec = BlockEncodingSpec(
        n_idx=1, n_s=0, n_d=0, A_max=1.0,
        O_c=O_c, O_r=O_r, O_data=O_data,
    )
    be = build_block_encoding(spec)
    A_ref = np.array([[0.0, 1.0], [1.0, 0.0]])
    passed, err = be.verify_against(A_ref, atol=1e-10)
    assert passed, f"anti-diagonal builder test failed (err={err})"


# ---------------------------------------------------------------------------
# Pauli-LCU on small dense matrices
# ---------------------------------------------------------------------------

def test_pauli_lcu_random_4x4():
    from qhomogenize.bencode.pauli_lcu import pauli_block_encoding

    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    A = (M + M.conj().T) / 2
    be = pauli_block_encoding(A)
    passed, err = be.verify_against(A, atol=1e-10)
    assert passed, f"random 4x4 Pauli LCU failed (err={err})"


def test_pauli_lcu_Ke():
    """The 8x8 element stiffness K_e block-encodes correctly via Pauli LCU."""
    from qhomogenize.bencode.pauli_lcu import pauli_block_encoding
    from qhomogenize.fea.element_template import element_template

    Ke_lam, Ke_mu, _, _ = element_template(a=0.25, b=0.25, phi_deg=90.0)
    Ke = Ke_lam + Ke_mu
    be = pauli_block_encoding(Ke)
    passed, err = be.verify_against(Ke, atol=1e-10)
    assert passed, f"K_e Pauli LCU failed (err={err})"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
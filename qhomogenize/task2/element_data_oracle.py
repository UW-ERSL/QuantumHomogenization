"""
Sünderhauf data oracle for the element-stiffness templates K_e^lambda, K_e^mu.

K_e is an 8x8 matrix; the number of distinct nonzero values in either
K_e^lambda or K_e^mu is small (mesh-independent, on the order of a few).
This module enumerates the distinct values of each layer's element template
and dispatches to qprim.data_oracle.

For the within-element addressing (sparsity), K_e is dense, so S_c^block =
S_r^block = 8 within each block of the block-diagonal D_K.

TODO
----
1. Compute Ke_lambda, Ke_mu via fea.element_template.element_template().
2. Enumerate distinct values within each (rounded to numerical precision).
3. Dispatch to qprim.data_oracle.data_oracle().
"""

from __future__ import annotations

from qiskit import QuantumCircuit


def element_data_oracle(layer: str = "lambda", name: str = None) -> tuple[QuantumCircuit, dict]:
    """Build the data oracle for K_e^layer (layer in {'lambda', 'mu'}).

    TODO.
    """
    raise NotImplementedError("TODO")

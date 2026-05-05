"""
Sampled amplitude extraction at L = 16 (where full-unitary becomes memory-pinching).

Strategy: for a logarithmic sample of (i, j) pairs spanning the sparsity pattern,
extract A_{ij} / alpha by:
    1. Prepare |j> on the system register and |0> on ancilla.
    2. Apply U_K.
    3. Measure the system register; project onto |i> ancilla |0>.
    4. The post-selected amplitude is A_{ij} / alpha; the success probability
       gives |A_{ij} / alpha|^2.
       For sign / phase, use a controlled-Hadamard test if needed.

For the homogenization test: also do an action-on-RHS test (apply U_K to a
prepared right-hand-side vector |u> and check the result matches K u / alpha).

TODO
----
- Choose the sample of (i, j) pairs (e.g., diagonal entries, off-diagonals
  near the disk boundary).
- Implement the post-selected amplitude estimation via shots.
"""

from __future__ import annotations


def sample_amplitudes(*args, **kwargs):
    raise NotImplementedError("TODO")

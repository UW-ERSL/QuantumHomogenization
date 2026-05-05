"""
qhomogenize: Structured quantum operator encoding for computational homogenization.

POC implementation of block-encoding circuits for the elasticity stiffness
operator K = A^T D_p A on a structured 2D Cartesian mesh under periodic
boundary conditions, following the structured-matrix framework of
Sünderhauf, Campbell, and Camps [Quantum 8, 1226 (2024)].

Subpackages
-----------
fea       : Classical numpy/scipy reference (K_e, A, K, microstructures, RHS).
qprim     : Quantum primitive circuits (integer arithmetic, oracles, diffusion).
bencode   : Sünderhauf-style block-encoding builder and compositional calculus.
task1     : Block encoding of the assembly operator A.
task2     : Block encoding of the element-stiffness operator D_p.
task3     : Composition K = A^T D_p A and generic baseline.
bench     : Validation, resource accounting, and headline figures.

References
----------
- C. Sünderhauf, E. Campbell, J. Camps. "Block-encoding structured matrices
  for data input in quantum computing." Quantum 8, 1226 (2024).
- E. Andreassen, C. S. Andreasen. "How to determine composite material
  properties using numerical homogenization." CMS 83, 488-495 (2014).
"""

__version__ = "0.1.0"

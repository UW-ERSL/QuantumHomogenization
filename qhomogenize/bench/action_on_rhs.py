"""
Validate U_K by computing K u for one of the homogenization test-strain RHS vectors
and comparing against the classical result.

For each macroscopic test strain epsilon_0 in {(1,0,0), (0,1,0), (0,0,1)}, the
right-hand side vector u_0 = chi^(test-strain mode) is prepared on the system
register, and U_K is applied.  The post-selected output (project ancilla on |0>)
should match K u_0 / alpha_K (up to global phase / scaling).

This is the validation step recommended in Task 3 step 3 of the POC plan for
the L = 16 case where full-unitary extraction is too expensive.

TODO
----
- StatePreparation of u_0 on the system register.
- Apply U_K.
- Read out the system register via amplitude estimation or full statevector
  (whichever is cheaper at the chosen L).
- Compare to K u_0 / alpha_K from numpy.
"""

from __future__ import annotations


def apply_UK_to_rhs(*args, **kwargs):
    raise NotImplementedError("TODO")

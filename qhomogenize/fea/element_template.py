"""
Element template for plane-strain bilinear-quad finite elements.

Computes the 8x8 element stiffness matrices K_e^lambda and K_e^mu (the Lamé
split of the Andreassen-Andreasen homogenization formulation), and the 8x3
element load vectors fe_lambda, fe_mu corresponding to the three macroscopic
test strains used in homogenization.

This is a thin re-export of `FEA2DHomogenize.element_mat_vec` from the
existing classical pipeline, factored out so the quantum side can import
without dragging in the homogenization driver or scipy.

Reference
---------
E. Andreassen, C. S. Andreasen, "How to determine composite material
properties using numerical homogenization," CMS 83, 488-495 (2014).
"""

from __future__ import annotations

import numpy as np


def element_template(a: float = 0.5, b: float = 0.5, phi_deg: float = 90.0):
    """Compute K_e^lambda, K_e^mu, fe_lambda, fe_mu for a bilinear-quad element.

    Parameters
    ----------
    a, b : float, default 0.5
        Half element width and half element height.  For a unit cell of L x L
        elements over a unit square, take a = b = 1/(2*L).  The default
        a = b = 0.5 corresponds to a single unit-square element (L=1, lx=ly=1).
    phi_deg : float, default 90.0
        Angle between the two cell-wall directions, in degrees.  90 = orthogonal
        (rectangular cell).

    Returns
    -------
    Ke_lambda : ndarray, shape (8, 8)
        Lambda contribution to element stiffness.  Symmetric.
    Ke_mu : ndarray, shape (8, 8)
        Mu contribution to element stiffness.  Symmetric.
    fe_lambda : ndarray, shape (8, 3)
        Lambda contribution to element load (three macroscopic strain cases).
    fe_mu : ndarray, shape (8, 3)
        Mu contribution to element load.

    Notes
    -----
    The full element stiffness for material with Lamé parameters (lambda, mu) is
        K_e = lambda * Ke_lambda + mu * Ke_mu.
    The DOF ordering within the 8-vector is (u_x, u_y) at corners
    (lower-left, lower-right, upper-right, upper-left) -- this matches the
    Andreassen-Andreasen convention used in FEA2DHomogenize.py.
    """
    # Constitutive matrix contributions
    C_mu = np.diag([2.0, 2.0, 1.0])
    C_lambda = np.zeros((3, 3))
    C_lambda[:2, :2] = 1.0

    # Two Gauss points in both directions
    g = 1.0 / np.sqrt(3.0)
    xx = np.array([-g, g])
    yy = xx
    ww = np.array([1.0, 1.0])

    Ke_lambda = np.zeros((8, 8))
    Ke_mu = np.zeros((8, 8))
    fe_lambda = np.zeros((8, 3))
    fe_mu = np.zeros((8, 3))

    L = np.zeros((3, 4))
    L[0, 0] = 1.0
    L[1, 3] = 1.0
    L[2, 1:3] = 1.0

    phi_rad = np.deg2rad(phi_deg)

    for ii, x in enumerate(xx):
        for jj, y in enumerate(yy):
            # Differentiated shape functions
            dNx = 0.25 * np.array([-(1 - y), (1 - y), (1 + y), -(1 + y)])
            dNy = 0.25 * np.array([-(1 - x), -(1 + x), (1 + x), (1 - x)])

            # Jacobian
            node_coords = np.array([
                [-a, -b],
                [a, -b],
                [a + 2 * b / np.tan(phi_rad), b],
                [2 * b / np.tan(phi_rad) - a, b],
            ])
            J = np.array([dNx, dNy]) @ node_coords
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            invJ = (1.0 / detJ) * np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]])

            weight = ww[ii] * ww[jj] * detJ

            # Strain-displacement matrix
            G = np.block([[invJ, np.zeros((2, 2))], [np.zeros((2, 2)), invJ]])
            dN = np.zeros((4, 8))
            dN[0, 0::2] = dNx
            dN[1, 0::2] = dNy
            dN[2, 1::2] = dNx
            dN[3, 1::2] = dNy
            B = L @ G @ dN

            Ke_lambda += weight * (B.T @ C_lambda @ B)
            Ke_mu += weight * (B.T @ C_mu @ B)
            fe_lambda += weight * (B.T @ C_lambda @ np.eye(3))
            fe_mu += weight * (B.T @ C_mu @ np.eye(3))

    return Ke_lambda, Ke_mu, fe_lambda, fe_mu


def element_template_for_mesh(L_mesh: int, phi_deg: float = 90.0):
    """Convenience wrapper: half-widths sized for L_mesh x L_mesh elements on a unit cell.

    Parameters
    ----------
    L_mesh : int
        Number of elements per side (i.e. the L in the POC plan).
    phi_deg : float
        Cell-wall angle in degrees; 90 for rectangular.

    Returns
    -------
    Same as `element_template`.
    """
    a = 0.5 / L_mesh
    b = 0.5 / L_mesh
    return element_template(a=a, b=b, phi_deg=phi_deg)


# ---------------------------------------------------------------------------
# Quick test: symmetry, rank, and consistency with classical FEA driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    Kl, Km, fl, fm = element_template(a=0.5, b=0.5, phi_deg=90.0)

    # Symmetry
    assert np.allclose(Kl, Kl.T, atol=1e-12), "Ke_lambda not symmetric"
    assert np.allclose(Km, Km.T, atol=1e-12), "Ke_mu not symmetric"

    # Rank: a single-element stiffness has 3 rigid-body modes (2 translations + 1 rotation),
    # so for plane-strain the stiffness should be rank 8 - 3 = 5 for the sum K_e at lambda=mu=1.
    Ke = Kl + Km
    rank = np.linalg.matrix_rank(Ke, tol=1e-9)
    assert rank == 5, f"expected element-stiffness rank 5, got {rank}"

    # Eigenvalues non-negative
    eigs = np.sort(np.linalg.eigvalsh(Ke))
    assert eigs[0] > -1e-9, f"negative eigenvalue: {eigs[0]}"
    # rank 5 => eigs[0..2] near zero (3 rigid-body modes), eigs[3..7] strictly positive.
    assert abs(eigs[2]) < 1e-9, f"third smallest eig should be ~0: {eigs[2]}"
    assert eigs[3] > 1e-9, f"fourth smallest eig should be strictly positive: {eigs[3]}"

    print("Ke_lambda shape:", Kl.shape, " ||Ke_lambda||_max =", np.max(np.abs(Kl)))
    print("Ke_mu     shape:", Km.shape, " ||Ke_mu||_max     =", np.max(np.abs(Km)))
    print("fe_lambda shape:", fl.shape)
    print("fe_mu     shape:", fm.shape)
    print("rank(Ke_lambda + Ke_mu) =", rank, " (expected 5)")
    print("eigs(Ke) =", np.round(eigs, 6))
    print("OK")

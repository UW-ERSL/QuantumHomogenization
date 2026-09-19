import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
from scipy.sparse import diags, kron, eye
from scipy.sparse import lil_matrix

class TrussFEM:
    def __init__(self, nodes, elements,loads, fixed_dofs, E=200e9, A=0.0005):
        """
        Finite element model for 2D truss structures.
        
        Parameters:
        -----------
        nodes : ndarray, shape (n_nodes, 2)
            Node coordinates [x, y]
        elements : list of tuples (i, j)
            Element connectivity: each element connects nodes i and j
        E : float
            Young's modulus (Pa)
        A : float
            Cross-sectional area (m^2)
        """
        self.nodes = np.array(nodes)
        self.n_dofs = 2 * len(nodes)
        self.elements = elements
        self.n_elements = len(elements)
        self.E = E
        self.loads = loads
        self.fixed_dofs = fixed_dofs
        # Allow A to be a scalar or array; convert to array of length n_elements
        if np.isscalar(A):
            self.A = np.full(self.n_elements, A)
        else:
            self.A = np.asarray(A)
            if self.A.shape[0] != self.n_elements:
                raise ValueError(" A must be a scalar or match number of elements")
        self.n_nodes = len(nodes)
        self.n_dof = 2 * self.n_nodes
        self.L = self.compute_all_lengths()
        self.Te = self.compute_Te()
        self.initialArea = self.A.copy()
        self.initial_volume = np.sum(self.initialArea * self.L)
        self.displacements = None
    

    def compute_element_length(self, i, j): 
        """Compute length of element between nodes i and j."""
        dx = self.nodes[j, 0] - self.nodes[i, 0]
        dy = self.nodes[j, 1] - self.nodes[i, 1]
        return np.sqrt(dx**2 + dy**2)
    
    def compute_all_lengths(self):
        """Compute all lengths."""
        lengths = []
        for idx, (i, j) in enumerate(self.elements):
            lengths.append(self.compute_element_length(i, j))
        return np.array(lengths)
    
    
    def set_area(self, A):
        """Set cross-sectional areas."""
        if np.isscalar(A):
            self.A = np.full(self.n_elements, A)
        else:
            self.A = np.asarray(A)
            if self.A.shape[0] != self.n_elements:
                raise ValueError("A must be a scalar or match number of elements")
            

    def get_element_displacements(self, bar_idx):
        """
        Get element displacement vector for a given element.
        
        Parameters:
        -----------
        bar_idx : int
            Element index
        d : ndarray, shape (n_dof,)
            Global displacement vector
            
        Returns:
        --------
        d_elem : ndarray, shape (4,)
            Element displacement vector [u_i, v_i, u_j, v_j]
        """
        i, j = self.elements[bar_idx]
        d_elem = np.array([
            self.displacements[2*i],
            self.displacements[2*i + 1],
            self.displacements[2*j],
            self.displacements[2*j + 1]
        ])
        return d_elem
    
    def compute_Te(self):
        """
        Compute transformation matrices Te for all elements.
        Each Te is a 4x4 matrix for mapping local to global coordinates.

        Returns:
        --------
        Te_all : ndarray, shape (n_elements, 4, 4)
            Array of transformation matrices for each element.
        """
        Te_all = np.zeros((self.n_elements, 4, 4))
        for elem in range(self.n_elements):
            i, j = self.elements[elem]
            dx = self.nodes[j, 0] - self.nodes[i, 0]
            dy = self.nodes[j, 1] - self.nodes[i, 1]
            L = self.L[elem]
            c = dx / L
            s = dy / L
            Te = np.array([
                [ c*c,  c*s, -c*c, -c*s],
                [ c*s,  s*s, -c*s, -s*s],
                [-c*c, -c*s,  c*c,  c*s],
                [-c*s, -s*s,  c*s,  s*s]
            ])
            Te_all[elem] = Te
        return Te_all
    
    def get_element_stiffness_matrix(self, elem):
        """
        Compute element stiffness matrix in global coordinates.
        
        Parameters:
        -----------
        elem : int
            Element index
            
        Returns:
        --------
        K_elem : ndarray, shape (4, 4)
            Element stiffness matrix
        L : float
            Element length
        """
        # Use precomputed transformation matrix Te
        Te = self.Te[elem]
        k = self.E * self.A[elem] / self.L[elem]
        K_elem = k * Te
        return K_elem
    
    def assemble_stiffness(self):
        """
        Assemble global stiffness matrix.
        
        Parameters:
        -----------
        area : array-like, shape (n_elements,)
            Cross-sectional areas for each element
            
        Returns:
        --------
        K : ndarray, shape (n_dof, n_dof)
            Global stiffness matrix
        lengths : ndarray
            Element lengths (for computing volume)
        """
        K = np.zeros((self.n_dof, self.n_dof))
        for idx, (i, j) in enumerate(self.elements):
            K_elem = self.get_element_stiffness_matrix(idx)
     
            # Global DOF indices for this element
            dofs = [2*i, 2*i+1, 2*j, 2*j+1]
            
            # Vectorized element contribution to global matrix
            K[np.ix_(dofs, dofs)] += K_elem
        
        return K
    
    def solve(self):
        """
        Solve equilibrium equations for given design.
        Automatically excludes hanging nodes (nodes with no connected members).
        
        Validity Requirements:
        - At least one element must connect to each force application node
        - At least one element must connect to each fixed support node
        
        Parameters:
        -----------
        loads : ndarray, shape (n_dof,)
            Applied loads at each DOF
        fixed_dofs : list
            Indices of constrained DOFs
        area : array-like
            Cross-sectional areas for each element
            
        Returns:
        --------
        u : ndarray, shape (n_dof,)
            Displacement vector (hanging nodes have zero displacement)
        valid : bool
            True if solution is valid (no singularity, constraints satisfied)
        """
        
        area = self.A

        loads = self.loads
        fixed_dofs = self.fixed_dofs
        # Identify connected nodes (nodes with at least one active member)
        connected_nodes = set()
        for idx, (i, j) in enumerate(self.elements):
            if area[idx] > 0:
                connected_nodes.add(i)
                connected_nodes.add(j)
        
        # Condition 1: Check that force nodes have at least one connected element
        force_nodes = set()
        for dof_idx in range(len(loads)):
            if loads[dof_idx] != 0:
                node_id = dof_idx // 2
                force_nodes.add(node_id)
        
        for force_node in force_nodes:
            if force_node not in connected_nodes:
                return np.zeros(self.n_dof), False
        
        # Condition 2: Check that each fixed node has at least one connected element
        fixed_nodes = set()
        for dof_idx in fixed_dofs:
            node_id = dof_idx // 2
            fixed_nodes.add(node_id)
        
        for fixed_node in fixed_nodes:
            if fixed_node not in connected_nodes:
                return np.zeros(self.n_dof), False
        
        # DOFs for connected nodes only
        connected_dofs = set()
        for node in connected_nodes:
            connected_dofs.add(2 * node)
            connected_dofs.add(2 * node + 1)
        
        # Free DOFs = connected DOFs minus fixed DOFs
        free_dofs = sorted(list(connected_dofs - set(fixed_dofs)))
        
        if len(free_dofs) == 0:
            # No free DOFs means structure is either empty or fully constrained
            return np.zeros(self.n_dof), False
        
        K = self.assemble_stiffness()
        # Extract free-free submatrix
        K_free = K[np.ix_(free_dofs, free_dofs)]
        f_free = loads[free_dofs]
        
        # Check conditioning
        if np.linalg.cond(K_free) > 1e12:
            return np.zeros(self.n_dof), False
        
        # Check for mechanism (rank deficiency)
        rank = np.linalg.matrix_rank(K_free)
        if rank < len(free_dofs):
            # Structure has mechanisms (insufficient constraints)
            return np.zeros(self.n_dof), False
        
        # Solve reduced system
        try:
            u_free = np.linalg.solve(K_free, f_free)
        except np.linalg.LinAlgError:
            return np.zeros(self.n_dof), False
        
        # Reconstruct full displacement vector
        d = np.zeros(self.n_dof)
        d[free_dofs] = u_free
        # Hanging nodes and fixed nodes remain at zero displacement
        self.displacements = d
        return d, True
    
    def compute_compliance(self, displacements):
        """Compute compliance."""
        return displacements @ self.assemble_stiffness()[0] @ displacements
    
    def compute_stresses(self, displacements):
        """
        Compute member stresses from displacement solution.
        
        Parameters:
        -----------
        area : array-like
            Cross-sectional areas for each element
        d : ndarray
            Displacement vector
            
        Returns:
        --------
        stresses : ndarray
            Stress in each element (0 for inactive elements)
        """
        stresses = np.zeros(len(self.elements))
        d = displacements
        area = self.A
        for idx, (i, j) in enumerate(self.elements):
            if area[idx] == 0:
                continue
            
            # Element geometry
            dx = self.nodes[j, 0] - self.nodes[i, 0]
            dy = self.nodes[j, 1] - self.nodes[i, 1]
            L = np.sqrt(dx**2 + dy**2)
            c, s = dx/L, dy/L
            
            # Element displacements
            d_elem = np.array([d[2*i], d[2*i+1], d[2*j], d[2*j+1]])
            
            # Axial strain
            epsilon = (1/L) * np.array([-c, -s, c, s]) @ d_elem
            
            # Stress
            stresses[idx] = self.E * epsilon
        
        return stresses
    
    
    def compute_volume(self):
        """Compute total volume of design."""
        return np.sum(self.A * self.L)


    def optimize_areas(self, volume_fraction=0.5, xi_min=0.01, xi_max=2):
        """
        Optimizes element cross-sectional areas to minimize compliance.
        
        Parameters:
        -----------
        volume_fraction : float
            Allowed volume as a fraction of the initial volume (0 to 1).
        xi_min : float
            Lower bound for relative area to avoid numerical singularities.
        xi_max : float, optional
            Upper bound for relative area. Defaults to 1 (initial area).
        """
  
        
        # Target volume limit

        n_elems = self.n_elements
        A0 = self.initialArea
        print(f"Initial volume: {self.initial_volume:.4f} m^3")
       
        # 1. Define Objective Function (Compliance)
        def objective(xi):
            A = xi*A0
            self.set_area(A)
            u, valid = self.solve()
            if not valid:
                return 1e20  # Return large penalty if system is unstable
            
            # Compliance c = f^T * u
            compliance = self.loads @ u
            
            # Sensitivity: dc/dxi_e = -A0 * E * L_e * epsilon_e^2
            grad = np.zeros(n_elems)
            for elem in range(n_elems):
                i, j = self.elements[elem]
                
                # Element geometry
                dx = self.nodes[j, 0] - self.nodes[i, 0]
                dy = self.nodes[j, 1] - self.nodes[i, 1]
                L = self.L[elem]  # Already computed
                c, s = dx/L, dy/L
                
                # Element displacements
                u_elem = self.get_element_displacements(elem)
                
                # Axial strain
                epsilon = (1/L) * np.array([-c, -s, c, s]) @ u_elem
                
                # Gradient
                grad[elem] = -A0[elem] * self.E * L * epsilon**2
                    
            return compliance, grad

        # 3. Define Constraints (Volume)
        def volume_constraint(xi):
            # Must be >= 0 for SLSQP
            volume = np.sum(xi * A0 * self.L)
            return volume_fraction - volume/self.initial_volume

        def volume_gradient(xi):
            return -A0 * self.L / self.initial_volume

        # Constraint dictionary
        cons = {'type': 'ineq', 'fun': volume_constraint, 'jac': volume_gradient}
        
        # 4. Bounds for A (to prevent elements from disappearing entirely)
        bounds = [(xi_min, xi_max) for _ in range(n_elems)]
        
        # 5. Run Optimization
        print(f"Starting optimization (Target Volume: {volume_fraction:.4f} fraction)...")
        res = minimize(
            fun=objective,
            x0=np.ones(n_elems),
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9, 'disp': False, 'maxiter': 100}
        )
        
        if res.success:
            self.set_area(res.x * self.initialArea)
            print(" Optimization successful.")
            print(f"Final volume: {self.compute_volume():.3g} m^3")
        else:
            print(f"Optimization failed: {res.message}")
            
        return res

    def evaluate_design(self, desiredVolumeFraction = 1,
                   d_hat=np.inf, sigma_hat=np.inf):
        """
        Fully evaluate a design: solve FEM and check constraints.
        """
        d, valid = self.solve()
        
        if not valid:
            return {
                'volume': np.inf,
                'max_disp': np.inf,
                'max_stress': np.inf,
                'compliance': np.inf
            }
        
        stresses = self.compute_stresses(d)
        volume = self.compute_volume()
        max_disp = np.max(np.sqrt(d[0::2]**2 + d[1::2]**2))
        max_stress = np.max(np.abs(stresses))
        compliance = self.loads @ d
        
    
        metrics = {
            'volume': volume,
            'max_disp': max_disp,
            'max_stress': max_stress,
            'compliance': compliance
        }
         
        
        return metrics

    def print_metrics(self, metrics):
        """
        Print performance metrics in a formatted way.
        
        Parameters:
        -----------
        metrics : dict
            Performance metrics dictionary
        """
        print(f"  Volume: {metrics['volume']:.3g}")
        print(f"  Max displacement: {metrics['max_disp']:.4g}")
        print(f"  Max stress: {metrics['max_stress']:.3g}")
        print(f"  Compliance: {metrics['compliance']:.3g}")

    def plot_truss(self, design=None, 
               displacements=None,
               show_nodes=True, show_labels=False,
               title="Truss Structure", figsize=(12, 8),
               save_path=None):
        """
        Visualize truss structure with optional deformed shape.
        
        Parameters:
        -----------
        design : array-like, optional
            Binary array indicating active elements. If None, plot all elements.
        displacements : ndarray, optional
            Displacement vector (for plotting deformed shape)
        show_nodes : bool
            Whether to show node markers
        show_labels : bool
            Whether to show node number labels
        title : str
            Plot title
        figsize : tuple
            Figure size
        save_path : str, optional
            If provided, save figure to this path
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        scale_factor = 0.1/(abs(displacements).max()+1e-12) if displacements is not None else 1.0
        # Determine which elements to plot
        if design is None:
            active_elements = np.ones(len(self.elements), dtype=bool)
        else:
            active_elements = np.array(design, dtype=bool)
        
        A = self.A
        # Plot undeformed structure with thickness proportional to A
        ref_A = np.mean(A)
        for idx, (i, j) in enumerate(self.elements):
            lw = 0.1 + 5.0 * (A[idx] / ref_A) if active_elements[idx] else 0.5
            color = 'k' if active_elements[idx] else 'lightgray'
            alpha = 0.7 if active_elements[idx] else 0.3
            ax.plot([self.nodes[i, 0], self.nodes[j, 0]], 
                [self.nodes[i, 1], self.nodes[j, 1]], 
                color=color, linewidth=lw, alpha=alpha, zorder=1)
            # Plot element number at x% along the element, slightly above the member
            frac = 0.60
            x_pos = self.nodes[i, 0] + frac * (self.nodes[j, 0] - self.nodes[i, 0])
            y_pos = self.nodes[i, 1] + frac * (self.nodes[j, 1] - self.nodes[i, 1])
            # Offset perpendicular to the element
            dx = self.nodes[j, 0] - self.nodes[i, 0]
            dy = self.nodes[j, 1] - self.nodes[i, 1]
            length = np.hypot(dx, dy)
            if length > 0:
                # Perpendicular direction (normalized)
                perp_x = -dy / length
                perp_y = dx / length
                offset = 0.08  # Adjust as needed
                x_pos += offset * perp_x
                y_pos += offset * perp_y
            if (show_labels):
                ax.text(x_pos, y_pos, str(idx), color='red', fontsize=24, ha='center', va='center', alpha=0.8)
        
        # Plot deformed structure if displacements provided
        if displacements is not None:
            deformed_nodes = self.nodes.copy()
            for i in range(self.n_nodes):
                deformed_nodes[i, 0] += scale_factor * displacements[2*i]
                deformed_nodes[i, 1] += scale_factor * displacements[2*i+1]
            
            for idx, (i, j) in enumerate(self.elements):
                if active_elements[idx]:
                    ax.plot([deformed_nodes[i, 0], deformed_nodes[j, 0]], 
                        [deformed_nodes[i, 1], deformed_nodes[j, 1]], 
                        'b--', linewidth=1.5, alpha=0.7, zorder=2,
                        label='Deformed' if idx == 0 else '')
            
            # Plot deformed nodes
            if show_nodes:
                ax.plot(deformed_nodes[:, 0], deformed_nodes[:, 1], 
                    'bo', markersize=6, zorder=5, alpha=0.7)
        
        # Plot undeformed nodes
        if show_nodes:
            LScale = 1.0
            if show_labels:
                # Show node numbers next to nodes with shaded background
                for i, (x, y) in enumerate(self.nodes):
                    ax.text(
                        x + 0.03*LScale, y + 0.03*LScale, str(i),
                        color='black', fontsize=24, ha='center', va='bottom', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', edgecolor='none', alpha=0.7)
                    )
        # Mark fixed supports
        if self.fixed_dofs is not None:
            fixed_nodes = set()
            for dof in self.fixed_dofs:
                node_idx = dof // 2
                fixed_nodes.add(node_idx)
            
            fixed_nodes = list(fixed_nodes)
            if fixed_nodes:
                ax.plot(self.nodes[fixed_nodes, 0], 
                    self.nodes[fixed_nodes, 1], 
                    'ks', markersize=12, zorder=6, 
                    markerfacecolor='black')
        
        # Draw load arrows
        if self.loads is not None:
            arrow_scale = 0.2  # Arrow length relative to structure size
            max_load = np.max(np.abs(self.loads[self.loads != 0])) if np.any(self.loads != 0) else 1.0
            
           
            for i in range(self.n_nodes):
                fx = self.loads[2*i]
                fy = self.loads[2*i+1]
                
                if abs(fx) > 1e-6 or abs(fy) > 1e-6:
                    # Compute arrow length (proportional to load magnitude)
                    magnitude = np.sqrt(fx**2 + fy**2)
                    arrow_len = arrow_scale * (magnitude / max_load)
                    
                    # Normalized force direction
                    fx_norm = fx / magnitude
                    fy_norm = fy / magnitude
                    
                    # Arrow starts away from node (opposite to force direction)
                    # and points toward node (in force direction)
                    start_x = self.nodes[i, 0] 
                    start_y = self.nodes[i, 1] 
                    
                    # Arrow displacement (in direction of force)
                    dx = arrow_len * fx_norm
                    dy = arrow_len * fy_norm
                    
                    ax.arrow(start_x, start_y, dx, dy,
                            head_width=0.05, head_length=0.03, 
                            fc='red', ec='red', linewidth=2.5, zorder=7,
                            )
                    
                    # Add load magnitude label (at end of arrow)
                    load_kN = magnitude / 1000
                    end_x = start_x + dx
                    end_y = start_y + dy
                    ax.text(end_x, end_y, 
                        f'{load_kN:.1f} kN',
                        color='red', fontsize=16, fontweight='bold',
                        ha='center', va='bottom' if fy < 0 else 'top',
                        bbox=dict(boxstyle='round,pad=0.4',
                                    facecolor='white', edgecolor='red', alpha=0.9))
        
        ax.set_xlabel('x (m)', fontsize=20)
        ax.set_ylabel('y (m)', fontsize=20)
        ax.set_title(title, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.axis('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
      
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()



class Poisson1DFD:
    def __init__(self, n_elements, f= 1, length=1.0):
        """
        1D Poisson equation FEM model
        
        Parameters:
        -----------
        n_elements : int
            Number of finite elements
        f : float or array-like
            Source term (right-hand side)
        length : float
            Length of the domain
        """
        self.n_elements = n_elements
        self.length = length
        self.n_nodes = n_elements + 1
        self.node_coords = np.linspace(0, length, self.n_nodes)
        self.h = length / n_elements
        self.f = f* np.ones(self.n_nodes)
        self.fixed_dofs = [0, self.n_nodes - 1]  # Dirichlet BCs at both ends
    
    def assemble_stiffness(self):
        """Assemble global stiffness matrix."""
        # K = 2 * np.eye(self.n_nodes)
        # for i in range(self.n_nodes - 1):
        #     K[i, i+1] = -1
        #     K[i+1, i] = -1
        diagonals = [2 * np.ones(self.n_nodes), -1 * np.ones(self.n_nodes-1), -1 * np.ones(self.n_nodes-1)]
        K = diags(diagonals, [0, 1, -1], format='csr')
        return K

    def solve(self):
        """Solve the FEM system."""
        K = self.assemble_stiffness()
        rhs = (self.h **2) * self.f.copy()
        
        # Apply boundary conditions
        free_dofs = list(set(range(self.n_nodes)) - set(self.fixed_dofs))
        
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        f_f = rhs[free_dofs]
        
        u_f = spsolve(K_ff, f_f)

        u = np.zeros(self.n_nodes)
        u[free_dofs] = u_f
        return u
    def plot_solution(self, u):
        """Plot the solution."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.node_coords, u, marker='o')
        plt.title(f'1D Poisson FD Solution (max = {np.max(u):.4f})')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.grid(True)
        plt.show()

class Poisson2DFD:

    def __init__(self, nx, ny, f=1.0, Lx=1.0, Ly=1.0):
        """
        2D Poisson equation: -∇²u = f with homogeneous Dirichlet BC
        
        Parameters:
        -----------
        nx, ny : int
            Number of interior grid points in x and y
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.hx = Lx / (nx + 1)  # uniform spacing in x
        self.hy = Ly / (ny + 1)  # uniform spacing in y
        self.n_interior = nx * ny
        
        # Create a grid of interior coordinates
        x = np.linspace(self.hx, self.Lx - self.hx, nx)
        y = np.linspace(self.hy, self.Ly - self.hy, ny)

        # meshgrid creates the 2D layout; 'ij' indexing matches your (i, j) lexicographic order
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Flatten to match the vector u (nx * ny, 2)
        self.coords = np.column_stack((X.ravel(), Y.ravel()))
        
        # Source term
        if callable(f):
            self.f = np.array([f(x, y) for x, y in self.coords])
        else:
            self.f = f * np.ones(self.n_interior)
    
    def assemble_stiffness(self):
        """Assemble stiffness matrix (interior nodes only)."""
        Nx, Ny = self.nx, self.ny
        
        # Grid spacing ratio
        r = self.hx**2 / self.hy**2
        
        # 1D tridiagonal with anisotropic diagonal
        T = diags([2*(1+r)*np.ones(Nx), -np.ones(Nx-1), -np.ones(Nx-1)], 
                  [0, 1, -1], format='csr')
        I_x = eye(Nx, format='csr')
        I_y = eye(Ny, format='csr')
        
        # 2D Laplacian with anisotropic coupling
        K = kron(I_y, T) + kron(diags([-r*np.ones(Ny-1), -r*np.ones(Ny-1)], 
                                      [1, -1], format='csr'), I_x)
        return K
    
    def solve(self):
        """Solve -∇²u = f."""
        K = self.assemble_stiffness()
        rhs = (self.hx * self.hy) * self.f
        u = spsolve(K, rhs)
        return u 

    def plot_solution(self, u, title='2D Poisson Solution', save_path=None):
        """
        Plot the solution of 2D Poisson equation.
        
        Parameters:
        -----------
        u : array-like
            Solution vector (interior nodes only)
        title : str
            Title for the plots
        save_path : str, optional
            Path to save the figure
        """

        
        # Create full grid including boundaries (u=0)
        U_full = np.zeros((self.ny + 2, self.nx + 2))
        U_full[1:-1, 1:-1] = u.reshape(self.ny, self.nx)
        
        # Create coordinate meshes
        x = np.linspace(0, self.Lx, self.nx + 2)
        y = np.linspace(0, self.Ly, self.ny + 2)
        X, Y = np.meshgrid(x, y)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 5))
        levels = 20
        contour = plt.contourf(X, Y, U_full, levels=levels, cmap='Greys', vmax=0.75)
        plt.contour(X, Y, U_full, levels=levels, colors='black', 
            linewidths=1.25, alpha=0.9)
        plt.xlabel('x', fontsize=10)
        plt.ylabel('y', fontsize=10)
        plt.title(f'2D Poisson FD Solution (max = {np.max(u):.4f})')
        plt.gca().set_aspect('equal')
        fig.colorbar(contour, ax=plt.gca())
        plt.tight_layout()
        plt.show()

class Poisson2DFE:

    def __init__(self, nx, ny, f=1.0, Lx=1.0, Ly=1.0):
        """
        2D Poisson equation: -∇²u = f with homogeneous Dirichlet BC using FEM
        
        Parameters:
        -----------
        nx, ny : int
            Number of elements in x and y directions
        """
        self.nx = nx  # number of elements in x
        self.ny = ny  # number of elements in y
        self.Lx = Lx
        self.Ly = Ly
        self.hx = Lx / nx  # element size in x
        self.hy = Ly / ny  # element size in y
        
        # Total number of nodes (including boundary)
        self.n_nodes_x = nx + 1
        self.n_nodes_y = ny + 1
        self.n_nodes = self.n_nodes_x * self.n_nodes_y
        
        # Number of interior nodes
        self.n_interior_x = nx - 1
        self.n_interior_y = ny - 1
        self.n_interior = self.n_interior_x * self.n_interior_y
        
        # Create node coordinates
        x = np.linspace(0, self.Lx, self.n_nodes_x)
        y = np.linspace(0, self.Ly, self.n_nodes_y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        self.all_coords = np.column_stack((X.ravel(), Y.ravel()))
        
        # Interior node coordinates (excluding boundaries)
        x_int = np.linspace(self.hx, self.Lx - self.hx, self.n_interior_x)
        y_int = np.linspace(self.hy, self.Ly - self.hy, self.n_interior_y)
        X_int, Y_int = np.meshgrid(x_int, y_int, indexing='ij')
        self.coords = np.column_stack((X_int.ravel(), Y_int.ravel()))
        
        # Build interior node indices
        interior_nodes = []
        for i in range(1, self.nx):
            for j in range(1, self.ny):
                interior_nodes.append(self.node_index(i, j))
        self.interior_nodes = np.array(interior_nodes)
        # Source term (evaluated at all nodes for assembly)
        if callable(f):
            self.f = np.array([f(x, y) for x, y in self.coords])
        else:
            self.f = f * np.ones(self.n_interior)
    
    def node_index(self, i, j):
        """Convert 2D node indices to global node number."""
        return i * self.n_nodes_y + j
    
    def assemble_stiffness(self):
        """Assemble stiffness matrix (interior nodes only) using Kronecker product approach."""
        Nx, Ny = self.n_interior_x, self.n_interior_y
        
        hx = self.hx
        hy = self.hy
        
        # 1D stiffness matrix: K1D_x with entries k_ij = ∫ φ'_i φ'_j dx
        Kx = (1.0/hx) * diags([2*np.ones(Nx), -np.ones(Nx-1), -np.ones(Nx-1)], 
                            [0, 1, -1], format='csr')
        Ky = (1.0/hy) * diags([2*np.ones(Ny), -np.ones(Ny-1), -np.ones(Ny-1)], 
                            [0, 1, -1], format='csr')
        
        # 1D mass matrix: M1D with entries m_ij = ∫ φ_i φ_j dx
        Mx = (hx/6.0) * diags([4*np.ones(Nx), np.ones(Nx-1), np.ones(Nx-1)], 
                            [0, 1, -1], format='csr')
        My = (hy/6.0) * diags([4*np.ones(Ny), np.ones(Ny-1), np.ones(Ny-1)], 
                            [0, 1, -1], format='csr')
        
        # 2D stiffness: ∫∫ (∂u/∂x·∂v/∂x + ∂u/∂y·∂v/∂y) dxdy
        # K2D = My ⊗ Kx + Ky ⊗ Mx
        K = kron(My, Kx) + kron(Ky, Mx)
        
        return K
    
    def solve(self):
        """Solve -∇²u = f using FEM."""
        K = self.assemble_stiffness()
        rhs = (self.hx * self.hy) * self.f
        u_interior = spsolve(K, rhs)
        
        # Create full solution vector (including boundaries)
        u_full = np.zeros(self.n_nodes)
        u_full[self.interior_nodes] = u_interior
        
        return u_full
    
    def plot_solution(self, u, title='2D Poisson FEM Solution', save_path=None):
        """
        Plot the solution of 2D Poisson equation.
        
        Parameters:
        -----------
        u : array-like
            Solution vector (all nodes including boundaries)
        title : str
            Title for the plots
        save_path : str, optional
            Path to save the figure
        """
        # Reshape solution to 2D grid
        U_grid = u.reshape(self.n_nodes_x, self.n_nodes_y)
        
        # Create coordinate meshes
        x = np.linspace(0, self.Lx, self.n_nodes_x)
        y = np.linspace(0, self.Ly, self.n_nodes_y)
        X, Y = np.meshgrid(x, y)
        
        # Create figure
        fig = plt.figure(figsize=(16, 5))
        levels = 20
        gray_levels = np.linspace(1.0, 0.7, levels)
        contour = plt.contourf(X, Y, U_grid.T, levels=levels,
                       colors=[str(g) for g in gray_levels])
        plt.contour(X, Y, U_grid.T, levels=levels, colors='black', 
                linewidths=1.25, alpha=0.9)
        plt.xlabel('x', fontsize=10)
        plt.ylabel('y', fontsize=10)
        plt.title(f'{title} (max = {np.max(u):.4f})')
        plt.gca().set_aspect('equal')
        fig.colorbar(contour, ax=plt.gca())
        plt.tight_layout()
        plt.show()


class PlaneStressFEM:
    """
    2D Plane Stress FEM for rectangular grid mesh
    
    Coordinate system:
        y ^
          |
          +----> x
    
    Element numbering (4-node quad):
        3---2
        |   |
        0---1
    """
    
    def __init__(self, nx, ny, lx=1.0, ly=1.0, E=200e9, nu=0.3, t=0.1):
        """
        Parameters:
        -----------
        nx, ny : int
            Number of elements in x and y directions
        lx, ly : float
            Domain dimensions (length in x and y)
        E : float
            Young's modulus
        nu : float
            Poisson's ratio
        t : float
            Thickness (for plane stress)
        """
        self.nelx = nx
        self.nely = ny
        self.lx = lx
        self.ly = ly
        self.E = E
        self.nu = nu
        self.t = t
        
        # Derived quantities
        self.n_elements = nx * ny
        self.n_nodes_x = nx + 1
        self.n_nodes_y = ny + 1
        self.n_nodes = self.n_nodes_x * self.n_nodes_y
        self.n_dofs = 2 * self.n_nodes
        
        # Element dimensions (all elements identical)
        self.elem_width = lx / nx
        self.elem_height = ly / ny
        
        # Node coordinates
        self.node_coords = self._generate_node_coordinates()
        
        # Element connectivity (which nodes form each element)
        self.connectivity = self._generate_connectivity()
        
        # Element stiffness matrix (identical for all elements)
        self.Ke = self._compute_element_stiffness()
        
        # Design variables (element densities)
        self.xi = np.ones(self.n_elements)  # Start with full material
        
        # SIMP penalization
        self.penal = 3.0  # Penalty parameter for SIMP
        self.xi_min = 1e-9  # Minimum to avoid singularity
        
        # Boundary conditions
        self.fixed_dofs = []
        self.forces = np.zeros(self.n_dofs)
        
        # Solution
        self.displacements = None  # Global displacement vector
        self.compliance = None
        self.title = "Plane Stress FEM Model"
    
    def _generate_node_coordinates(self):
        """Generate coordinates for all nodes"""
        coords = np.zeros((self.n_nodes, 2))
        
        for i in range(self.n_nodes_y):
            for j in range(self.n_nodes_x):
                node_id = i * self.n_nodes_x + j
                coords[node_id, 0] = j * self.elem_width
                coords[node_id, 1] = i * self.elem_height
        
        return coords
    
    def _generate_connectivity(self):
        """
        Generate element connectivity matrix
        
        Returns:
        --------
        connectivity : array of shape (n_elements, 4)
            connectivity[e] = [n0, n1, n2, n3] are the 4 node IDs for element e
            
        Node ordering for each element:
            3---2
            |   |
            0---1
        """
        connectivity = np.zeros((self.n_elements, 4), dtype=int)
        
        for ely in range(self.nely):
            for elx in range(self.nelx):
                elem_id = ely * self.nelx + elx
                
                # Bottom-left node
                n0 = ely * self.n_nodes_x + elx
                n1 = n0 + 1
                n2 = n0 + self.n_nodes_x + 1
                n3 = n0 + self.n_nodes_x
                
                connectivity[elem_id] = [n0, n1, n2, n3]
        
        return connectivity
    
    def _compute_element_stiffness(self):
        """
        Compute element stiffness matrix for 4-node bilinear quad
        Uses 2×2 Gauss quadrature
        
        Returns:
        --------
        Ke : array of shape (8, 8)
            Element stiffness matrix in global coordinates
            DOF ordering: [u0, v0, u1, v1, u2, v2, u3, v3]
        """
        # Material matrix (plane stress)
        D = (self.E / (1 - self.nu**2)) * np.array([
            [1,      self.nu, 0              ],
            [self.nu, 1,      0              ],
            [0,      0,      (1-self.nu)/2   ]
        ])
        
        # Gauss points and weights (2×2 quadrature)
        gauss = 1.0 / np.sqrt(3.0)
        gp = np.array([[-gauss, -gauss],
                       [ gauss, -gauss],
                       [ gauss,  gauss],
                       [-gauss,  gauss]])
        gw = np.array([1.0, 1.0, 1.0, 1.0])
        
        # Element dimensions
        a = self.elem_width
        b = self.elem_height
        
        # Initialize element stiffness
        Ke = np.zeros((8, 8))
        
        # Loop over Gauss points
        for i in range(4):
            xi, eta = gp[i]
            w = gw[i]
            
            # Shape function derivatives in natural coordinates
            dN_xi = 0.25 * np.array([
                -(1-eta),  (1-eta),  (1+eta), -(1+eta)
            ])
            dN_eta = 0.25 * np.array([
                -(1-xi), -(1+xi),  (1+xi),  (1-xi)
            ])
            
            # Jacobian matrix
            J = np.array([
                [a/2, 0  ],
                [0,   b/2]
            ])
            
            detJ = a * b / 4.0
            J_inv = np.linalg.inv(J)
            
            # Shape function derivatives in physical coordinates
            dN = J_inv @ np.array([dN_xi, dN_eta])
            dN_dx = dN[0, :]
            dN_dy = dN[1, :]
            
            # Strain-displacement matrix B (3×8)
            B = np.zeros((3, 8))
            for node in range(4):
                B[0, 2*node]     = dN_dx[node]
                B[1, 2*node + 1] = dN_dy[node]
                B[2, 2*node]     = dN_dy[node]
                B[2, 2*node + 1] = dN_dx[node]
            
            # Add contribution to element stiffness
            Ke += w * self.t * detJ * (B.T @ D @ B)
        
        return Ke
    
    def set_xi(self, xi):
        """Set element densities (design variables)"""
        self.xi = np.clip(xi, self.xi_min, 1.0)
    
    def apply_boundary_condition(self, node_ids, dof_x=True, dof_y=True):
        """
        Fix nodes (set displacements to zero)
        
        Parameters:
        -----------
        node_ids : array-like
            Node IDs to fix
        dof_x : bool
            Fix x-direction displacement
        dof_y : bool
            Fix y-direction displacement
        """
        for node in node_ids:
            if dof_x:
                self.fixed_dofs.append(2*node)
            if dof_y:
                self.fixed_dofs.append(2*node + 1)
        
        self.fixed_dofs = list(set(self.fixed_dofs))  # Remove duplicates
    
    def apply_force(self, node_id, fx=0.0, fy=0.0):
        """Apply force at a node"""
        self.forces[2*node_id]     += fx
        self.forces[2*node_id + 1] += fy
    
    def assemble_global_stiffness(self):
        """
        Assemble global stiffness matrix using element densities
        
        K_global = Σ_e (ρ_e^p) * Ke
        
        Returns sparse CSR matrix
        """
        # Prepare sparse matrix data
        row_indices = []
        col_indices = []
        values = []
        
        for elem in range(self.n_elements):
            # Get element nodes
            nodes = self.connectivity[elem]
            
            # Element DOFs (8 DOFs: 2 per node)
            elem_dofs = np.zeros(8, dtype=int)
            for i, node in enumerate(nodes):
                elem_dofs[2*i]   = 2*node      # x-displacement
                elem_dofs[2*i+1] = 2*node + 1  # y-displacement
            
            # SIMP interpolation: E_e = E_min + (E_0 - E_min) * xi^p
            density_factor = self.xi[elem] ** self.penal
            
            # Scaled element stiffness
            Ke_scaled = density_factor * self.Ke
            
            # Add to global system
            for i in range(8):
                for j in range(8):
                    row_indices.append(elem_dofs[i])
                    col_indices.append(elem_dofs[j])
                    values.append(Ke_scaled[i, j])
        
        # Create sparse matrix
        K = coo_matrix((values, (row_indices, col_indices)), 
                       shape=(self.n_dofs, self.n_dofs))
        
        return K.tocsr()
    
    def solve(self):
        """
        Solve FEA system: K*U = F
        
        Returns:
        --------
        U : array
            Global displacement vector
        is_valid : bool
            Whether solution succeeded
        """
        # Assemble global stiffness
     
        K = self.assemble_global_stiffness()
        
        # Free DOFs
        free_dofs = np.setdiff1d(np.arange(self.n_dofs), self.fixed_dofs)
        
        # Extract free system
        K_free = K[free_dofs, :][:, free_dofs]
        F_free = self.forces[free_dofs]
        
        try:
            # Solve
            d_free = spsolve(K_free, F_free)
            
            # Assemble full displacement vector
            self.displacements = np.zeros(self.n_dofs)
            self.displacements[free_dofs] = d_free
            
            # Compute compliance: c = F^T * U = U^T * K * U
            self.compliance = (self.forces @ self.displacements)
            return self.displacements, True
        except:
            return None, False
    
    def get_element_displacements(self, elem_id):
        """Get displacement vector for element elem_id"""
        if self.displacements is None:
            raise RuntimeError("Must solve first!")
        
        nodes = self.connectivity[elem_id]
        U_elem = np.zeros(8)
        
        for i, node in enumerate(nodes):
            U_elem[2*i]   = self.displacements[2*node]
            U_elem[2*i+1] = self.displacements[2*node + 1]
        
        return U_elem
    
    def get_element_strain_energy(self, elem_id):
        """
        Compute strain energy for element elem_id
        
        Returns:
        --------
        strain_energy : float
            U_e^T * Ke * U_e
        """
        d_elem = self.get_element_displacements(elem_id)
        
        # Strain energy (without density scaling for QAOA Hamiltonian)
        strain_energy = d_elem @ self.Ke @ d_elem
        
        return strain_energy
    
    def compute_elem_strain_energies(self):
        se = np.zeros(self.n_elements)
        ke = self.Ke
        u = self.displacements
        elem_dofs = np.zeros(8, dtype=int)
        for e in range(self.n_elements):
            # Map element displacements using the edof matrix
            nodes = self.connectivity[e]
            
            # Element DOFs (8 DOFs: 2 per node)
            for i, node in enumerate(nodes):
                elem_dofs[2*i]   = 2*node      # x-displacement
                elem_dofs[2*i+1] = 2*node + 1  # y-displacement
            u_e = u[elem_dofs]
            se[e] = u_e.T @ ke @ u_e
            
        return se
    def get_compliance(self):
        """Get current compliance"""
        if self.compliance is None:
            raise RuntimeError("Must solve first!")
        
        return self.compliance
    
    def evaluate_design(self):
        """Evaluate current design"""
        volume = np.sum(self.xi) * self.elem_width * self.elem_height * self.t
        
        return {
            'compliance': self.compliance,
            'volume': volume,
            'volume_fraction': np.mean(self.xi),
        }
    
    def print_metrics(self, metrics):
        """
        Print performance metrics for the current design.
        """
        print(f"  Compliance: {metrics['compliance']:.4g}")
        print(f"  Volume: {metrics['volume']:.4g}")
        print(f"  Volume fraction: {metrics['volume_fraction']:.4f}")

    def plot_mesh(self, show_bc=True, show_loads=True, ax=None):
        """Plot the mesh, boundary conditions, and loads (undeformed)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        # Plot elements

        min_xi_less_than_one = np.any(np.min(self.xi) < 1)
        for elem in range(self.n_elements):
            nodes = self.connectivity[elem]
            x = self.node_coords[nodes, 0]
            y = self.node_coords[nodes, 1]
            # Close the loop
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax.plot(x, y, 'k-', linewidth=0.5)
            # Fill element with color based on density xi
            if (min_xi_less_than_one):
                color = plt.cm.gray_r(self.xi[elem])
                ax.fill(x, y, color=color, alpha=0.7, edgecolor=None)

        # Plot boundary conditions (fixed DOFs)
        if show_bc and self.fixed_dofs:
            fixed_nodes = set([dof // 2 for dof in self.fixed_dofs])
            ax.plot(self.node_coords[list(fixed_nodes), 0],
                    self.node_coords[list(fixed_nodes), 1],
                    'ks', markersize=10, markerfacecolor='black', label='Fixed')

        # Plot loads
        if show_loads and np.any(np.abs(self.forces) > 1e-12):
            arrow_scale = 0.1 * max(self.lx, self.ly)
            max_force = np.max(np.abs(self.forces)) if np.any(self.forces != 0) else 1.0
            for node in range(self.n_nodes):
                fx = self.forces[2*node]
                fy = self.forces[2*node+1]
                if abs(fx) > 1e-12 or abs(fy) > 1e-12:
                    mag = np.sqrt(fx**2 + fy**2)
                    if mag == 0:
                        continue
                    dx = arrow_scale * fx / max_force
                    dy = arrow_scale * fy / max_force
                    # The arrow starts at the node and points in the direction of the force
                    ax.arrow(self.node_coords[node, 0],
                        self.node_coords[node, 1],
                        dx, dy,
                        head_width=0.03*max(self.lx, self.ly),
                        head_length=0.05*max(self.lx, self.ly),
                        fc='red', ec='red', linewidth=2, zorder=5)
                    ax.text(self.node_coords[node, 0] + dx,
                        self.node_coords[node, 1] + dy,
                        f'{mag:.2g}', color='red', fontsize=9, ha='center', va='center')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(self.title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return 
    

    def plot_displacement(self,  ax=None):
        """Plot deformed mesh"""
        if self.displacements is None:
            raise RuntimeError("Must solve first!")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        scale = 0.1*max(self.lx, self.ly)/np.max(np.abs(self.displacements))
        # Plot deformed mesh (colored by density)
        for elem in range(self.n_elements):
            nodes = self.connectivity[elem]
            
            # Deformed coordinates
            x = self.node_coords[nodes, 0] + scale * self.displacements[2*nodes]
            y = self.node_coords[nodes, 1] + scale * self.displacements[2*nodes + 1]
            
            # Close the loop
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            
            color = plt.cm.gray_r(self.xi[elem])
            ax.plot(x, y, color=color, linewidth=0.5)
        
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Deformed Shape (scale={scale:0.2g})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return

class PlaneStressOC:
    """
    Optimality Criteria (OC) optimizer compatible with PlaneStressFEM.
    """
    
    def __init__(self, fea, volume_fraction, filter_radius=1.5, 
                 move_limit=0.2, max_iter=100, tol=0.01):
        self.fea = fea
        self.vf = volume_fraction
        self.rmin = filter_radius
        self.move = move_limit
        self.max_iter = max_iter
        self.tol = tol
        
        # Initialize with uniform density at target volume fraction
        self.fea.set_xi(np.ones(fea.n_elements) * volume_fraction)
        
        # Prepare filter weights based on unit element coordinates
        self._prepare_filter()
        
        self.history = {'iteration': [], 'compliance': [], 'volume_fraction': [], 'change': []}
    
    def _prepare_filter(self):
        """Prepare sensitivity filter weights assuming unit-sized elements."""
        nelx, nely = self.fea.nelx, self.fea.nely
        self.H = np.zeros((self.fea.n_elements, self.fea.n_elements))
        
        for e1 in range(self.fea.n_elements):
            ely1, elx1 = divmod(e1, nelx)
            cx1, cy1 = elx1 + 0.5, ely1 + 0.5
            
            # Find neighbors within radius rmin
            i_min, i_max = int(max(elx1 - self.rmin, 0)), int(min(elx1 + self.rmin + 1, nelx))
            j_min, j_max = int(max(ely1 - self.rmin, 0)), int(min(ely1 + self.rmin + 1, nely))
            
            for ely2 in range(j_min, j_max):
                for elx2 in range(i_min, i_max):
                    e2 = ely2 * nelx + elx2
                    dist = np.sqrt((elx1 - elx2)**2 + (ely1 - ely2)**2)
                    if dist < self.rmin:
                        self.H[e1, e2] = self.rmin - dist
        
        self.Hs = np.sum(self.H, axis=1)

    def _compute_sensitivities(self):
        """Compute compliance sensitivities dC/dxi."""
        dc = np.zeros(self.fea.n_elements)
        p = 3.0  # SIMP penalty
        ke = self.fea.Ke
        u = self.fea.displacements
        elem_dofs = np.zeros(8, dtype=int)
        for e in range(self.fea.n_elements):
            # Map element displacements using the edof matrix
            nodes = self.fea.connectivity[e]
            
            # Element DOFs (8 DOFs: 2 per node)
            for i, node in enumerate(nodes):
                elem_dofs[2*i]   = 2*node      # x-displacement
                elem_dofs[2*i+1] = 2*node + 1  # y-displacement
            u_e = u[elem_dofs]
            strain_energy = u_e.T @ ke @ u_e
            
            # Sensitivity formula: -p * xi^(p-1) * u^T * ke * u
            dc[e] = -p * (self.fea.xi[e]**(p - 1)) * strain_energy
            
        return dc

    def _filter_sensitivities(self, dc):
        """Apply density-based sensitivity filter."""
        # Standard sensitivity filter: (H * (xi * dc)) / (xi * Hs)
        dc_filtered = (self.H @ (self.fea.xi * dc)) / (self.fea.xi * self.Hs)
        return dc_filtered

    def _oc_update(self, dc):
        """Optimality Criteria update step with bisection for Lagrange multiplier."""
        l1, l2 = 0, 1e9
        xi_old = self.fea.xi.copy()
        
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            # OC update rule with move limits and physical bounds [0, 1]
            xi_new = np.maximum(0.001, # Avoid zero for numerical stability
                     np.maximum(xi_old - self.move,
                     np.minimum(1.0,
                     np.minimum(xi_old + self.move,
                                xi_old * np.sqrt(-dc / lmid)))))
            
            # Check volume constraint (average density)
            if np.mean(xi_new) > self.vf:
                l1 = lmid
            else:
                l2 = lmid
        return xi_new

    def optimize(self, verbose=True):
        if verbose:
            print(f"Starting OC Optimization: Target VF={self.vf}, Filter R={self.rmin}")

        for it in range(self.max_iter):
            # 1. FEA Solve
            self.fea.solve()
            
            # 2. Get Metrics and Sensitivities
            metrics = self.fea.evaluate_design()
            dc = self._compute_sensitivities()
            
            # 3. Filter and Update
            dc_filtered = self._filter_sensitivities(dc)
            xi_old = self.fea.xi.copy()
            xi_new = self._oc_update(dc_filtered)
            
            # 4. Update FEM state
            self.fea.set_xi(xi_new)
            
            # 5. Convergence check
            change = np.max(np.abs(xi_new - xi_old))
            self.history['iteration'].append(it)
            self.history['compliance'].append(metrics['compliance'])
            self.history['volume_fraction'].append(metrics['volume_fraction'])
            self.history['change'].append(change)

            if verbose and it % 5 == 0:
                print(f"Iter {it:3d}: Compliance={metrics['compliance']:.4g}, volume_fraction={metrics['volume_fraction']:.4f}, Change={change:.4f}")

            
        
        return self.history

"""
Examples of truss problems

"""

def truss2x2(E=200e9, A=0.0005):
    # Example usage with the 2x2 truss
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [1.0, 0.0],   # Node 1 (bottom right)  - LOADED
        [0.0, 1.0],   # Node 2 (top left) - FIXED
        [1.0, 1.0]    # Node 3 (top right)
    ])

    elements = [
        (0, 1), (2, 3),  # Horizontal
        (0, 2), (1, 3),  # Vertical
        (0, 3), (1, 2)   # Diagonals
    ]

    # Define problem
    fixed_dofs = [0, 1, 4, 5]  # Nodes 0 and 2 fixed
    loads = np.zeros(2 * len(nodes))
    loads[2*1 + 1] = -100000  # 100 kN downward at node 1

     # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)

    return fem_model

def truss2x3(E=200e9, A=0.0005):
    # Example usage with the 2x3 truss
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [1.0, 0.0],   # Node 1 (bottom right) - LOADED
        [0.0, 1.0],   # Node 2 (mid left)  - FIXED
        [1.0, 1.0],   # Node 3 (mid right) 
        [0.0, 2.0],   # Node 4 (top left)  - FIXED
        [1.0, 2.0]    # Node 5 (top right)
    ])

    elements = [
        (0, 1), (0, 2), (0, 3), 
        (1, 2), (1, 3),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
        (4, 5)
    ]

    # Define problem
    fixed_dofs = [0, 1, 4, 5, 8, 9]  # Nodes 0, 2, and 4 fixed
    loads = np.zeros(2 * len(nodes))
    loads[2*1 + 1] = -10000  # 10 kN downward at node 1

    # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)

    return fem_model


def truss3x2(E=200e9, A=0.0005):
    # Example usage with the 3x2 truss
    # Nodes: 6
    # Elements: 11
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [2.0, 0.0],   # Node 1 (bottom center)
        [4.0, 0.0],   # Node 2 (bottom right) - FIXED
        [0.0, 2.0],   # Node 3 (top left)
        [2.0, 2.0],   # Node 4 (top center) - LOADED
        [4.0, 2.0]    # Node 5 (top right)
    ])

    elements = [
        (0, 1), (1, 2), (3, 4), (4, 5),  # Horizontal
        (0, 3), (1, 4), (2, 5),          # Vertical
        (0, 4), (1, 3), (1, 5), (2, 4)   # Diagonals
    ]

    # Define problem
    fixed_dofs = [0, 1, 4, 5]  # Nodes 0 and 2 fixed
    loads = np.zeros(2 * len(nodes))
    loads[2*4 + 1] = -10000  # 10 kN downward at node 4

    # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)

    return fem_model

def truss3x3(E=200e9, A=0.0005):
    # Example usage with the 3x3 truss
    # Nodes: 9
    # Elements: 26
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [2.0, 0.0],   # Node 1 (bottom center)
        [4.0, 0.0],   # Node 2 (bottom right) - FIXED
        [0.0, 1.5],   # Node 3 (middle left)
        [2.0, 1.5],   # Node 4 (middle center)
        [4.0, 1.5],   # Node 5 (middle right)
        [0.0, 3.0],   # Node 6 (top left)
        [2.0, 3.0],   # Node 7 (top center) - LOADED
        [4.0, 3.0]    # Node 8 (top right)
    ])

    elements = [
        (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),  # Horizontal
        (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8),  # Vertical
        (0, 4), (1, 3), (1, 5), (2, 4), (3, 7), (4, 6), (4, 8), (5, 7),  # Diagonals
        (0, 8), (2, 6), (0, 7), (1, 6), (1, 8), (2, 7)   # Long diagonals
    ]

    

    # Define problem
    fixed_dofs = [0, 1, 4, 5]  # Nodes 0 and 2 fixed
    loads = np.zeros(2 * len(nodes))
    loads[2*7 + 1] = -10000  # 10 kN downward at node 7

    # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)

    return fem_model

def truss_grid(M = 8, N = 4, Lx=1.0, E=200e9, A=0.0005):
        """
        Create a 2D truss ground structure with an M x N grid of nodes.
        Each square cell is fully connected: horizontal, vertical, and both diagonals.

        Parameters:
        -----------
        M : int
            Number of nodes in x-direction (columns)
        N : int
            Number of nodes in y-direction (rows)
        Lx : float
            Total width of the grid
        Ly : float
            Total height of the grid
        E : float
            Young's modulus
        A : float
            Cross-sectional area

        Returns:
        --------
        TrussFEM instance
        """
        # Node coordinates
        Ly= Lx * (N - 1) / (M - 1)  # Maintain aspect ratio
        x_coords = np.linspace(0, Lx, M)
        y_coords = np.linspace(0, Ly, N)
        nodes = np.array([[x, y] for y in y_coords for x in x_coords])

        # Helper to get node index from (ix, iy)
        def node_id(ix, iy):
            return iy * M + ix

        elements = []
        for iy in range(N):
            for ix in range(M):
                n0 = node_id(ix, iy)
                # Horizontal (right neighbor)
                if ix < M - 1:
                    n1 = node_id(ix + 1, iy)
                    elements.append((n0, n1))
                # Vertical (top neighbor)
                if iy < N - 1:
                    n2 = node_id(ix, iy + 1)
                    elements.append((n0, n2))
                # Diagonal up-right
                if ix < M - 1 and iy < N - 1:
                    n3 = node_id(ix + 1, iy + 1)
                    elements.append((n0, n3))
                # Diagonal up-left
                if ix > 0 and iy < N - 1:
                    n4 = node_id(ix - 1, iy + 1)
                    elements.append((n0, n4))

        # Example boundary conditions: fix left wall,
        fixed_dofs = []
        for iy in range(N):
            n = node_id(0, iy)
            fixed_dofs.extend([2 * n, 2 * n + 1])
        print(f"Fixed DOFs at left wall nodes: {[node_id(0, iy) for iy in range(N)]}")
        loads = np.zeros(2 * M * N)
        bottom_right = node_id(M - 1, 0)
        print(f"Applying load at node {bottom_right} (bottom right corner)")
        loads[2 * bottom_right + 1] = -10000  # Downward force at bottom right node

        fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)
        return fem_model

def truss3x3Substructure(E=200e9, A=0.0005):
    
    fem_model = truss3x3(E=E, A=A)
    #  Plot with a specific design
    sub_structure = np.array([
        # Horizontal members (indices 0-5)
        1, 1,  # Bottom row: (0,1), (1,2) - CRITICAL for node 1 stability
        1, 1,  # Middle row: (3,4), (4,5) - lateral bracing
        0, 0,  # Top row: not needed (nodes 6,8 hanging)
        
        # Vertical members (indices 6-11)
        1, 0,  # Left column: (0,3) active
        1, 1,  # Center column: (1,4), (4,7) - load path
        1, 0,  # Right column: (2,5) active
        
        # In-square diagonals (indices 12-19)
        1, 0,  # (0,4) diagonal bracing
        0, 1,  # (2,4) diagonal bracing
        1, 0,  # (3,7) diagonal bracing
        0, 1,  # (5,7) diagonal bracing
        
        # Long diagonals (indices 20-25)
        0, 0, 0, 0, 0, 0
    ], dtype=int)
    area = fem_model.A * sub_structure  # Zero area for inactive members
    fem_model.set_area(area)
    print(f"Number of active members: {np.sum(sub_structure)}")  # 10 members
    return fem_model

def truss_10bar(E=200e9, A=0.005):
    """
    Classic 10-bar truss ground structure
    
    Reference: Rajeev, S., & Krishnamoorthy, C. S. (1992). 
    "Discrete optimization of structures using genetic algorithms."
    Journal of Structural Engineering, 118(5), 1233-1250.
    
    Node layout:
    - Bottom row (y=0): Nodes 0, 1, 2
    - Top row (y=H):   Nodes 3, 4, 5
    
    Boundary conditions:
    - Node 0: Fixed (bottom left)
    - Node 2: Fixed (bottom right)
    
    Loading:
    - Node 1: Vertical load (bottom center)
    - Node 3: Vertical load (top left)
    
    Ground structure: 10 potential members
    - 4 horizontals (top and bottom chords)
    - 2 verticals (left and right, NO center vertical)
    - 4 diagonals (cross-bracing)
    """
    
    # Standard dimensions from benchmark (360 inches = 9.14 m)
    L = 9.14  # Bay width (meters)
    H = 9.14  # Height (meters)
    
    # Node coordinates
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [L,   0.0],   # Node 1 (bottom center) - LOADED
        [2*L, 0.0],   # Node 2 (bottom right) - FIXED
        [0.0, H],     # Node 3 (top left) - LOADED
        [L,   H],     # Node 4 (top center)
        [2*L, H]      # Node 5 (top right)
    ])
    
    # Ground structure with exactly 10 potential bars
    elements = [
        (0, 1),  # Bar 0: Bottom chord left
        (1, 2),  # Bar 1: Bottom chord right
        (3, 4),  # Bar 2: Top chord left
        (4, 5),  # Bar 3: Top chord right
        (0, 3),  # Bar 4: Vertical left
        (2, 5),  # Bar 5: Vertical right
        (0, 4),  # Bar 6: Diagonal (0→4)
        (1, 3),  # Bar 7: Diagonal (1→3)
        (1, 5),  # Bar 8: Diagonal (1→5)
        (2, 4)   # Bar 9: Diagonal (2→4)
    ]
    
    # Fixed degrees of freedom
    fixed_dofs = [0, 1, 4, 5]  # Nodes 0 and 2 fully fixed (x and y)
    
    # Applied loads (standard benchmark loads)
    loads = np.zeros(2 * len(nodes))
    P = 444822  # 100 kips = 444,822 N
    loads[2*1 + 1] = -P  # Downward load at node 1 (bottom center)
    loads[2*3 + 1] = -P  # Downward load at node 3 (top left)
    
    # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)
    
    return fem_model

"""
Examples of plane stress problems

"""

def PlaneStressCantilever(nx=60, ny=20, E= 200e9, nu=0.3):
    """
    Classic cantilever beam topology optimization problem
    
    Setup:
        |============================
        |                           ↓ F
        |============================
        
    Fixed left edge, load at center of right edge
    """
   
    # Create mesh
    ly = 1.0/ny
    lx = nx * (ly / ny)

    fea2D = PlaneStressFEM(nx=nx, ny=ny, lx=lx, ly=ly, E=E, nu=nu)
    
    # Boundary conditions: Fix left edge
    left_nodes = np.arange(0, fea2D.n_nodes, fea2D.n_nodes_x)
    fea2D.apply_boundary_condition(left_nodes, dof_x=True, dof_y=True)
    
    # Load: Downward force at center of right edge
    center_right_node = fea2D.n_nodes_x - 1 + (ny // 2) * fea2D.n_nodes_x
    fea2D.apply_force(center_right_node, fx=0.0, fy=-10000)
    fea2D.title = "Plane Stress Cantilever Beam"
    return fea2D


class MicrostructureGenerator:
    def __init__(self, nx, ny, inclusion_fraction=0.3, micro_type='disk'):
        self.nx = nx
        self.ny = ny
        self.inclusion_fraction = inclusion_fraction
        self.data = self.generate(type=micro_type)
        

    def generate(self, type='disk'):
        nx, ny = self.nx, self.ny
     
        # Create microstructure
        if type == 'disk':
            x = np.linspace(0, nx, nx)
            y = np.linspace(0, ny, ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            center_x, center_y = nx/2, ny/2
            radius = ((self.inclusion_fraction*nx*ny)/np.pi)**0.5
            micro = (((X-center_x)**2 + (Y-center_y)**2) > radius**2).astype(float)  # disk inclusion
            function = lambda x, y: ((x - center_x)**2 + (y - center_y)**2) > radius**2
            # Compute and print actual volume fraction

        elif type == 'square':
            micro = np.ones((nx, ny))
            side = int((self.inclusion_fraction * nx * ny)**0.5)
            start_x = (nx - side) // 2
            start_y = (ny - side) // 2
            micro[start_x:start_x + side, start_y:start_y + side] = 0 # square inclusion
        elif type == '4disks':
            micro = np.ones((nx, ny))
            np.random.seed(0)
            n_inclusions = 4
            # Calculate radius for each disk to achieve desired volume fraction
            area_total = nx * ny
            area_inclusions = self.inclusion_fraction * area_total
            radius = np.sqrt(area_inclusions / (n_inclusions * np.pi))
            centers = [(nx//4, ny//4), (3*nx//4, ny//4), (nx//4, 3*ny//4), (3*nx//4, 3*ny//4)]
            for center in centers:
                x0, y0 = center
                y, x = np.ogrid[-x0:nx - x0, -y0:ny - y0]
                mask = x*x + y*y <= radius*radius
                micro[mask] = 0
        elif type == 'random_disks':
            micro = np.ones((nx, ny))
            n_inclusions = 10
            # Calculate radius for each disk to achieve desired volume fraction
            area_total = nx * ny
            area_inclusions = self.inclusion_fraction * area_total
            radius = np.sqrt(area_inclusions / (n_inclusions * np.pi))
            for _ in range(n_inclusions):
                x0 = np.random.randint(radius, nx - radius)
                y0 = np.random.randint(radius, ny - radius)
                y, x = np.ogrid[-x0:nx - x0, -y0:ny - y0]
                mask = x*x + y*y <= radius*radius
                micro[mask] = 0
        elif type == 'random_elements':
            micro = np.ones((nx, ny))
            n_elements = int(np.round(self.inclusion_fraction * nx * ny))
            n_elements = max(0, min(n_elements, nx * ny))
            if n_elements > 0:
                idx = np.random.choice(nx * ny, size=n_elements, replace=False)
                micro.flat[idx] = 0


        elif type == 'primitive_tpms':
            # Primitive/Egg-crate TPMS: cos(x) + cos(y) = t
            x = np.linspace(0, 2*np.pi, nx, endpoint=False)
            y = np.linspace(0, 2*np.pi, ny, endpoint=False)
            X, Y = np.meshgrid(x, y, indexing='ij')

            # Adjust t to achieve target volume fraction
            # For primitive: VF ≈ 0.5 - t/4 (approximate)
            t = 2 * (0.5 - self.inclusion_fraction)

            phi = np.cos(X) + np.cos(Y) - t
            micro = (phi > 0).astype(float)

            print(f"TPMS thickness parameter t: {t:.3f}")

        elif type == 'gyroid_tpms':
            # Gyroid-like 2D: sin(x)cos(y) + cos(x)sin(y) = t
            x = np.linspace(0, 2*np.pi, nx, endpoint=False)
            y = np.linspace(0, 2*np.pi, ny, endpoint=False)
            X, Y = np.meshgrid(x, y, indexing='ij')

            # For gyroid: adjust t iteratively to hit volume fraction
            t = 0.0  # Start at symmetric point
            phi = np.sin(X)*np.cos(Y) + np.cos(X)*np.sin(Y) - t
            micro = (phi > 0).astype(float)

            # Simple bisection to find correct t
            t_low, t_high = -1.41, 1.41
            for _ in range(20):
                vf = np.sum(micro) / (nx * ny)
                if abs(vf - self.inclusion_fraction) < 0.01:
                    break
                if vf > self.inclusion_fraction:
                    t_low = t
                else:
                    t_high = t
                t = (t_low + t_high) / 2
                phi = np.sin(X)*np.cos(Y) + np.cos(X)*np.sin(Y) - t
                micro = (phi > 0).astype(float)

            print(f"TPMS thickness parameter t: {t:.3f}")
        elif type == 'schwarz_tpms':
            # Schwarz-like 2D: cos(x)cos(y) = t
            x = np.linspace(0, 2*np.pi, nx, endpoint=False)
            y = np.linspace(0, 2*np.pi, ny, endpoint=False)
            X, Y = np.meshgrid(x, y, indexing='ij')

            # Bisection for volume fraction
            t_low, t_high = -1.0, 1.0
            t = 0.0
            for _ in range(20):
                phi = np.cos(X)*np.cos(Y) - t
                micro = (phi > 0).astype(float)
                vf = np.sum(micro) / (nx * ny)
                if abs(vf - self.inclusion_fraction) < 0.01:
                    break
                if vf > self.inclusion_fraction:
                    t_low = t
                else:
                    t_high = t
                t = (t_low + t_high) / 2

            print(f"TPMS thickness parameter t: {t:.3f}")
        else:
            raise ValueError(f"Unknown microstructure type: {type}")

        inclusion_fraction_actual = 1- np.sum(micro) / (nx * ny)
        print(f"Target inclusion fraction: {self.inclusion_fraction:.3f}")
        print(f"Actual inclusion fraction: {inclusion_fraction_actual:.3f}")
        return micro

    def plot(self):
        nx, ny = self.nx, self.ny
        _, ax = plt.subplots()
        ax.imshow(self.data, cmap='Greys', origin='lower', vmin=0, vmax=1.4)
        ax.set_title('')
        ax.set_xlim(-0.5, ny - 0.5)
        ax.set_ylim(-0.5, nx - 0.5)
        ax.set_xticks(np.arange(-0.5, ny, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nx, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.6)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.show()

class FEA2DHomogenize:
    @staticmethod
    def fea_homogenize(E_incl, nu_incl, E_matrix, nu_matrix, microData):
        """
        How to determine composite material properties using numerical homogenization
        Erik Andreassen, Casper Schousboe Andreasen
        Computational Materials Science
        Volume 83, 15 February 2014, Pages 488-495

        Returns
        -------
        CH : ndarray, shape (3, 3)
            Homogenized elasticity tensor.
        """
        print('--- Homogenization of 2D plane-strain periodic material ---')
        
        # INITIALIZE
        [nx, ny] = microData.shape
        lx = nx
        ly = ny
        
        # matrix = 0, inclusion = 1
        E0 = np.array([E_matrix, E_incl])  # Young's modulus of the two materials
        nu0 = np.array([nu_matrix, nu_incl])  # Poisson's ratio of the two materials
        # Convert to Lame parameters
        lambda_mat = E0 * nu0 / ((1 + nu0) * (1 - 2 * nu0))
        mu_mat = E0 / (2 * (1 + nu0))
        
        phi = 90.0  # Angle between horizontal and vertical cell wall
       
        nelx = nx
        nely = ny
      
        
        # Compute element size
        dx = lx / nelx
        dy = ly / nely
        nel = nelx * nely
        
        # Get element matrices and vectors
        ke_lambda, ke_mu, fe_lambda, fe_mu = FEA2DHomogenize.element_mat_vec(dx/2, dy/2, phi)
        # Node numbers and element degrees of freedom for full (not periodic) mesh
        nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape(1 + nely, 1 + nelx, order='F')
        edof_vec = (2 * nodenrs[:-1, :-1] + 1).flatten(order='F')
        edof_mat = np.tile(edof_vec.reshape(-1, 1), (1, 8)) + \
                   np.tile(np.array([0, 1, 2*nely + 2, 2*nely + 3, 2*nely, 2*nely + 1, -2, -1]), (nel, 1))
        
        # IMPOSE PERIODIC BOUNDARY CONDITIONS
        nn = (nelx + 1) * (nely + 1)  # Total number of nodes
        nnP = nelx * nely  # Total number of unique nodes
        nnP_array = np.arange(1, nnP + 1).reshape(nely, nelx, order='F')
        
        # Extend with a mirror of the top border
        nnP_array = np.vstack([nnP_array, nnP_array[0, :]])
        # Extend with a mirror of the left border
        nnP_array = np.column_stack([nnP_array, nnP_array[:, 0]])
        
        # Make a vector into which we can index using edofMat
        dof_vector = np.zeros(2 * nn, dtype=int)
        dof_vector[0::2] = 2 * nnP_array.flatten(order='F') - 1
        dof_vector[1::2] = 2 * nnP_array.flatten(order='F')
        edof_mat = dof_vector[edof_mat - 1]  # -1 for 0-based indexing
        
        ndof = 2 * nnP  # Number of dofs
        
        # ASSEMBLE STIFFNESS MATRIX
        # Indexing vectors
        iK = np.kron(edof_mat, np.ones((8, 1))).T
        jK = np.kron(edof_mat, np.ones((1, 8))).T
        
        # Material properties in the different elements
        lambda_elem = lambda_mat[0] * (microData == 0) + lambda_mat[1] * (microData == 1)
        mu_elem = mu_mat[0] * (microData == 0) + mu_mat[1] * (microData == 1)
        
        # The corresponding stiffness matrix entries
        sK = np.outer(ke_lambda.flatten(order='F'), lambda_elem.flatten(order='F')) + \
             np.outer(ke_mu.flatten(order='F'), mu_elem.flatten(order='F'))
        
        K = coo_matrix((sK.flatten(order='F'), (iK.flatten(order='F') - 1, jK.flatten(order='F') - 1)),
                       shape=(ndof, ndof)).tocsr()
        
        # LOAD VECTORS AND SOLUTION
        # Assemble three load cases corresponding to the three strain cases
        sF = np.outer(fe_lambda.flatten(order='F'), lambda_elem.flatten(order='F')) + \
             np.outer(fe_mu.flatten(order='F'), mu_elem.flatten(order='F'))
        
        iF = np.tile(edof_mat.T, (3, 1))
        jF = np.vstack([np.ones((8, nel)), 2 * np.ones((8, nel)), 3 * np.ones((8, nel))])
        
        F = coo_matrix((sF.flatten(order='F'), (iF.flatten(order='F') - 1, jF.flatten(order='F') - 1)),
                       shape=(ndof, 3)).toarray()
        
        # Solve (remember to constrain one node)
        chi = np.zeros((ndof, 3))
        chi[2:, :] = spsolve(K[2:, 2:].tocsr(), F[2:, :])
        
        # HOMOGENIZATION
        # The displacement vectors corresponding to the unit strain cases
        chi0 = np.zeros((nel, 8, 3))
        
        # The element displacements for the three unit strains
        chi0_e = np.zeros((8, 3))
        ke = ke_mu + ke_lambda  # Here the exact ratio does not matter
        fe = fe_mu + fe_lambda  # because it is reflected in the load vector
        
        # Solve for the element displacements
        free_dofs = np.array([2, 4, 5, 6, 7])
        chi0_e[free_dofs, :] = np.linalg.solve(ke[np.ix_(free_dofs, free_dofs)], fe[free_dofs, :])
        
        # epsilon0_11 = (1, 0, 0)
        chi0[:, :, 0] = np.tile(chi0_e[:, 0], (nel, 1))
        # epsilon0_22 = (0, 1, 0)
        chi0[:, :, 1] = np.tile(chi0_e[:, 1], (nel, 1))
        # epsilon0_12 = (0, 0, 1)
        chi0[:, :, 2] = np.tile(chi0_e[:, 2], (nel, 1))
        
        # Compute homogenized tensor
        CH = np.zeros((3, 3))
        cell_volume = lx * ly
        
        for i in range(3):
            for j in range(3):
                # Extract displacements for load case i and j
                chi_i = chi0[:, :, i] - chi[edof_mat - 1, i]
                chi_j = chi0[:, :, j] - chi[edof_mat - 1, j]
                
                sum_lambda = np.sum((chi_i @ ke_lambda) * chi_j, axis=1)
                sum_mu = np.sum((chi_i @ ke_mu) * chi_j, axis=1)
                
                sum_lambda = sum_lambda.reshape(nely, nelx, order='F')
                sum_mu = sum_mu.reshape(nely, nelx, order='F')
                
                # Homogenized elasticity tensor
                CH[i, j] = (1 / cell_volume) * np.sum(lambda_elem * sum_lambda + mu_elem * sum_mu)
        
        return CH

    @staticmethod
    def element_mat_vec(a, b, phi):
        """
        Compute element stiffness matrix and force vector numerically.

        Parameters
        ----------
        a : float
            Half element width.
        b : float
            Half element height.
        phi : float
            Angle between horizontal and vertical cell wall (degrees).

        Returns
        -------
        ke_lambda : ndarray, shape (8, 8)
            Element stiffness matrix for lambda contribution.
        ke_mu : ndarray, shape (8, 8)
            Element stiffness matrix for mu contribution.
        fe_lambda : ndarray, shape (8, 3)
            Element force vector for lambda contribution.
        fe_mu : ndarray, shape (8, 3)
            Element force vector for mu contribution.
        """
        # Constitutive matrix contributions
        C_mu = np.diag([2, 2, 1])
        C_lambda = np.zeros((3, 3))
        C_lambda[:2, :2] = 1
        
        # Two Gauss points in both directions
        xx = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        yy = xx
        ww = np.array([1, 1])
        
        # Initialize
        ke_lambda = np.zeros((8, 8))
        ke_mu = np.zeros((8, 8))
        fe_lambda = np.zeros((8, 3))
        fe_mu = np.zeros((8, 3))
        
        L = np.zeros((3, 4))
        L[0, 0] = 1
        L[1, 3] = 1
        L[2, 1:3] = 1
        
        phi_rad = phi * np.pi / 180
        
        for ii in range(len(xx)):
            for jj in range(len(yy)):
                # Integration point
                x = xx[ii]
                y = yy[jj]
                
                # Differentiated shape functions
                dNx = 0.25 * np.array([-(1 - y), (1 - y), (1 + y), -(1 + y)])
                dNy = 0.25 * np.array([-(1 - x), -(1 + x), (1 + x), (1 - x)])
                
                # Jacobian
                node_coords = np.array([
                    [-a, -b],
                    [a, -b],
                    [a + 2*b/np.tan(phi_rad), b],
                    [2*b/np.tan(phi_rad) - a, b]
                ])
                
                J = np.array([dNx, dNy]) @ node_coords
                detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                invJ = (1 / detJ) * np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]])
                
                # Weight factor at this point
                weight = ww[ii] * ww[jj] * detJ
                
                # Strain-displacement matrix
                G = np.block([[invJ, np.zeros((2, 2))],
                             [np.zeros((2, 2)), invJ]])
                
                dN = np.zeros((4, 8))
                dN[0, 0::2] = dNx
                dN[1, 0::2] = dNy
                dN[2, 1::2] = dNx
                dN[3, 1::2] = dNy
                
                B = L @ G @ dN
                
                # Element matrices
                ke_lambda += weight * (B.T @ C_lambda @ B)
                ke_mu += weight * (B.T @ C_mu @ B)
                
                # Element loads
                fe_lambda += weight * (B.T @ C_lambda @ np.eye(3))
                fe_mu += weight * (B.T @ C_mu @ np.eye(3))
        
        return ke_lambda, ke_mu, fe_lambda, fe_mu


    @staticmethod
    def get_E_nu_from_C(C_voigt):
        nu = C_voigt[0,1] / (C_voigt[0,0] + C_voigt[0,1])
        E = 2 * C_voigt[2,2] * (1 + nu)
        return E, nu
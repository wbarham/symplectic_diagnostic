import numba as nb
import numpy as np
import scipy.sparse as sp

@nb.njit
def p2g_B1(pos, vec, dx):
    """
    Computes grid indices and weights for charge deposition
    using linear B-splines.

    Args:
        pos: Particle positions in the domain (array of shape [N_particles])
        vec: Pre-allocated array for grid vector (shape [Nx])
        dx: Grid spacing
    """

    Nx = len(vec)
    vec[:] = 0.0

    # Loop over particles
    for i in range(len(pos)):
        # Obtain the nearest grid point 
        grid_idx = int(np.floor(pos[i] / dx))  
        # Obtain the relative position within the grid point
        grid_frac = (pos[i] - grid_idx * dx) / dx

        # Linear B-spline weights
        w0 = 1 - grid_frac
        w1 = grid_frac

        # Indices
        idx0 = grid_idx % Nx
        idx1 = (grid_idx + 1) % Nx

        # Compute contributions to the grid vector from ith particle
        vec[idx0] += w0
        vec[idx1] += w1

@nb.njit(parallel=True)
def p2g_B1_par(pos, vec, dx):
    """
    Computes grid indices and weights for charge deposition
    using linear B-splines using threading.

    Args:
        pos: Particle positions in the domain (array of shape [N_particles])
        vec: Pre-allocated array for grid vector (shape [Nx])
        dx: Grid spacing
    """

    Nx = len(vec)
    vec[:] = 0.0

    num_threads =  nb.get_num_threads()
    thread_bins = np.zeros((num_threads, Nx), dtype=np.float64)

    for i in nb.prange(len(pos)):
        thread_id = nb.get_thread_id()
        grid_idx = int(np.floor(pos[i] / dx))
        grid_frac = (pos[i] - grid_idx * dx) / dx

        # Linear B-spline weights
        w0 = 1 - grid_frac
        w1 = grid_frac

        # Indices
        idx0 = grid_idx % Nx
        idx1 = (grid_idx + 1) % Nx

        thread_bins[thread_id, idx0] += w0
        thread_bins[thread_id, idx1] += w1

    # Final reduction
    for t in range(num_threads):
        for j in range(Nx):
            vec[j] += thread_bins[t, j]

@nb.njit
def p2g_B2(pos, vec, dx):
    """
    Computes grid indices and weights for charge deposition
    using quadratic B-splines.

    Args:
        pos: Particle positions in the domain (array of shape [N_particles])
        vec: Pre-allocated array for grid vector (shape [Nx])
        dx: Grid spacing
    """

    Nx = len(vec)
    vec[:] = 0.0

    # Parallel loop over particles
    for i in range(len(pos)):  # nb.prange enables parallel for loop

        # Obtain the central grid point
        grid_idx = int(np.floor(pos[i] / dx))  
        # offset from center
        r = (pos[i] - grid_idx * dx) / dx

        # Quadratic interpolation
        w0 = 1/2 * (1 - r)**2
        w1 = 1/2 + r - r**2
        w2 = 1/2 * r**2

        # Indices
        idx0 = (grid_idx - 1) % Nx
        idx1 = (grid_idx + 0) % Nx
        idx2 = (grid_idx + 1) % Nx

        vec[idx0] += w0
        vec[idx1] += w1
        vec[idx2] += w2

@nb.njit(parallel=True)
def p2g_B2_par(pos, vec, dx):
    """
    Computes grid indices and weights for charge deposition
    using quadratic B-splines with threadwise parallelism.

    Args:
        pos: Particle positions in the domain (array of shape [N_particles])
        vec: Pre-allocated array for grid vector (shape [Nx])
        dx: Grid spacing
    """

    Nx = len(vec)
    vec[:] = 0.0

    num_threads = nb.get_num_threads()
    thread_vecs = np.zeros((num_threads, Nx), dtype=np.float64)

    for i in nb.prange(len(pos)):
        thread_id = nb.get_thread_id()

        # Obtain the central grid point
        grid_idx = int(np.floor(pos[i] / dx))  
        # offset from center
        r = (pos[i] - grid_idx * dx) / dx

        # Quadratic interpolation
        w0 = 1/2 * (1 - r)**2
        w1 = 1/2 + r - r**2
        w2 = 1/2 * r**2

        # Indices
        idx0 = (grid_idx - 1) % Nx
        idx1 = (grid_idx + 0) % Nx
        idx2 = (grid_idx + 1) % Nx

        thread_vecs[thread_id, idx0] += w0
        thread_vecs[thread_id, idx1] += w1
        thread_vecs[thread_id, idx2] += w2

    # Final reduction
    for t in range(num_threads):
        for j in range(Nx):
            vec[j] += thread_vecs[t, j]

@nb.njit
def p2g_B3(pos, vec, dx):
    """
    Computes grid indices and weights for charge deposition
    using cubic B-splines.

    Args:
        pos: Particle positions in the domain (array of shape [N_particles])
        vec: Pre-allocated array for grid vector (shape [Nx])
        dx: Grid spacing
    """

    Nx = len(vec)
    vec[:] = 0.0

    for i in range(len(pos)):
        grid_idx = int(np.floor(pos[i] / dx))
        r = (pos[i] - grid_idx * dx) / dx

        w0 = (1/6) * (1 - r)**3
        w1 = (1/6) * (3 * r**3 - 6 * r**2 + 4)
        w2 = (1/6) * (-3 * r**3 + 3 * r**2 + 3 * r + 1)
        w3 = (1/6) * r**3

        vec[(grid_idx - 1) % Nx] += w0
        vec[grid_idx % Nx]       += w1
        vec[(grid_idx + 1) % Nx] += w2
        vec[(grid_idx + 2) % Nx] += w3

@nb.njit(parallel=True)
def p2g_B3_par(pos, vec, dx):
    """
    Computes grid indices and weights for charge deposition
    using cubic B-splines with threadwise parallelism.

    Args:
        pos: Particle positions in the domain (array of shape [N_particles])
        vec: Pre-allocated array for grid vector (shape [Nx])
        dx: Grid spacing
    """

    Nx = len(vec)
    vec[:] = 0.0

    num_threads = nb.get_num_threads()
    thread_vecs = np.zeros((num_threads, Nx), dtype=np.float64)

    for i in nb.prange(len(pos)):
        thread_id = nb.get_thread_id()

        grid_idx = int(np.floor(pos[i] / dx))
        r = (pos[i] - grid_idx * dx) / dx

        w0 = (1/6) * (1 - r)**3
        w1 = (1/6) * (3 * r**3 - 6 * r**2 + 4)
        w2 = (1/6) * (-3 * r**3 + 3 * r**2 + 3 * r + 1)
        w3 = (1/6) * r**3

        idx0 = (grid_idx - 1) % Nx
        idx1 = grid_idx % Nx
        idx2 = (grid_idx + 1) % Nx
        idx3 = (grid_idx + 2) % Nx

        thread_vecs[thread_id, idx0] += w0
        thread_vecs[thread_id, idx1] += w1
        thread_vecs[thread_id, idx2] += w2
        thread_vecs[thread_id, idx3] += w3

    # Final reduction
    for t in range(num_threads):
        for j in range(Nx):
            vec[j] += thread_vecs[t, j]

@nb.njit
def g2p_dB1(pos, grid, val, dx):
    """
    Evaluates derivative of linear B-splines from grid to particles.

    Args:
        pos: Particle positions (1D array)
        grid: Grid vector (1D array of shape [Nx])
        val: Output array to accumulate results (same shape as pos)
        dx: Grid spacing
    """
    
    Nx = len(grid)
    val[:] = 0

    for i in range(len(pos)):
        # Nearest left grid point
        grid_idx = int(np.floor(pos[i] / dx))
        #frac = (pos[i] - grid_idx * dx) / dx

        # Derivatives of B1 shape functions
        dw0 = -1.0 / dx
        dw1 =  1.0 / dx

        idx0 = grid_idx % Nx
        idx1 = (grid_idx + 1) % Nx

        val[i] += grid[idx0] * dw0 + grid[idx1] * dw1

@nb.njit(parallel=True)
def g2p_dB1_par(pos, grid, val, dx):
    """
    Thread-parallel g2p interpolation of derivative of B1 splines.

    Args:
        pos: Particle positions (1D array)
        grid: Grid vector (1D array of shape [Nx])
        val: Output array to accumulate results (same shape as pos)
        dx: Grid spacing
    """

    Nx = len(grid)
    val[:] = 0

    for i in nb.prange(len(pos)):
        grid_idx = int(np.floor(pos[i] / dx))
        #frac = (pos[i] - grid_idx * dx) / dx

        dw0 = -1.0 / dx
        dw1 =  1.0 / dx

        idx0 = grid_idx % Nx
        idx1 = (grid_idx + 1) % Nx

        val[i] += grid[idx0] * dw0 + grid[idx1] * dw1

@nb.njit
def g2p_dB2(pos, grid, val, dx):
    """
    Interpolates derivative of quadratic B-splines from grid to particles.

    Args:
        pos: Particle positions (1D array)
        grid: Grid vector (1D array of shape [Nx])
        val: Output array to accumulate results (same shape as pos)
        dx: Grid spacing
    """
    Nx = len(grid)
    val[:] = 0

    for i in range(len(pos)):
        grid_idx = int(np.floor(pos[i] / dx))
        r = (pos[i] - grid_idx * dx) / dx

        # Quadratic interpolation
        dw0 = (r - 1) / dx
        dw1 = (1 - 2 * r) / dx
        dw2 = r / dx

        idx0 = (grid_idx - 1) % Nx
        idx1 = (grid_idx + 0) % Nx
        idx2 = (grid_idx + 1) % Nx

        val[i] += grid[idx0] * dw0 + grid[idx1] * dw1 + grid[idx2] * dw2

@nb.njit(parallel=True)
def g2p_dB2_par(pos, grid, val, dx):
    """
    Thread-parallel g2p interpolation of derivative of B2 splines.

    Args:
        pos: Particle positions (1D array)
        grid: Grid vector (1D array of shape [Nx])
        val: Output array to accumulate results (same shape as pos)
        dx: Grid spacing
    """
    Nx = len(grid)
    val[:] = 0

    for i in nb.prange(len(pos)):
        grid_idx = int(np.floor(pos[i] / dx))
        r = (pos[i] - grid_idx * dx) / dx

        # Quadratic interpolation
        dw0 = (r - 1) / dx
        dw1 = (1 - 2 * r) / dx
        dw2 = r / dx

        idx0 = (grid_idx - 1) % Nx
        idx1 = (grid_idx + 0) % Nx
        idx2 = (grid_idx + 1) % Nx

        val[i] += grid[idx0] * dw0 + grid[idx1] * dw1 + grid[idx2] * dw2

@nb.njit
def g2p_dB3(pos, grid, val, dx):
    """
    Interpolates derivative of cubic B-splines from grid to particles.

    Args:
        pos: Particle positions (1D array)
        grid: Grid vector (1D array of shape [Nx])
        val: Output array to accumulate results (same shape as pos)
        dx: Grid spacing
    """
    Nx = len(grid)
    val[:] = 0

    for i in range(len(pos)):
        grid_idx = int(np.floor(pos[i] / dx))
        r = (pos[i] - grid_idx * dx) / dx

        dw0 = -0.5 * (1 - r)**2 / dx
        dw1 = 1/6 * (9 * r**2 - 12 * r) / dx
        dw2 = 1/6 * (-9 * r**2 + 6 * r + 3) / dx
        dw3 = 0.5 * r**2 / dx

        idx0 = (grid_idx - 1) % Nx
        idx1 = grid_idx % Nx
        idx2 = (grid_idx + 1) % Nx
        idx3 = (grid_idx + 2) % Nx

        val[i] += (
            grid[idx0] * dw0 +
            grid[idx1] * dw1 +
            grid[idx2] * dw2 +
            grid[idx3] * dw3
        )

@nb.njit(parallel=True)
def g2p_dB3_par(pos, grid, val, dx):
    """
    Thread-parallel g2p interpolation of derivative of cubic B-splines.

    Args:
        pos: Particle positions (1D array)
        grid: Grid vector (1D array of shape [Nx])
        val: Output array to accumulate results (same shape as pos)
        dx: Grid spacing
    """
    Nx = len(grid)
    val[:] = 0

    for i in nb.prange(len(pos)):
        grid_idx = int(np.floor(pos[i] / dx))
        r = (pos[i] - grid_idx * dx) / dx

        dw0 = -0.5 * (1 - r)**2 / dx
        dw1 = 1/6 * (9 * r**2 - 12 * r) / dx
        dw2 = 1/6 * (-9 * r**2 + 6 * r + 3) / dx
        dw3 = 0.5 * r**2 / dx

        idx0 = (grid_idx - 1) % Nx
        idx1 = grid_idx % Nx
        idx2 = (grid_idx + 1) % Nx
        idx3 = (grid_idx + 2) % Nx

        val[i] += (
            grid[idx0] * dw0 +
            grid[idx1] * dw1 +
            grid[idx2] * dw2 +
            grid[idx3] * dw3
        )

def construct_G(Nx):
    """
    Constructs the derivative matrix for B-splines (of any order) on a periodic grid.

    Args:
        Nx: Number of grid points.

    Returns:
        G: Sparse mass matrix in CSR format.
    """

    G = sp.lil_matrix((Nx, Nx))
    for i in range(Nx):
        G[i, i] = 1
        G[i, (i - 1) % Nx] = - 1
    G = G.tocsr()
    return G

def construct_M_linear(Nx):
    """
    Constructs the mass matrix for linear B-splines on a periodic grid.

    Args:
        Nx: Number of grid points.

    Returns:
        M: Sparse mass matrix in CSR format.
    """
    M = sp.lil_matrix((Nx, Nx))

    for i in range(Nx):
        M[i, i] = 2 / 3  # Diagonal elements
        M[i, (i - 1) % Nx] = M[i, (i + 1) % Nx] = 1 / 6  # Off-diagonal elements
    
    M = M.tocsr()
    return M

def construct_M_quadratic(Nx):
    """
    Constructs the mass matrix for quadratic B-splines on a periodic grid.

    Args:
        Nx: Number of grid points.

    Returns:
        M: Sparse mass matrix in CSR format.
    """

    M = sp.lil_matrix((Nx, Nx))
    for i in range(Nx):
        M[i, i] = 11 / 20
        M[i, (i - 1) % Nx] = M[i, (i + 1) % Nx] = 13 / 60
        M[i, (i - 2) % Nx] = M[i, (i + 2) % Nx] = 1 / 120 
    M = M.tocsr()
    return M

def construct_laplacian(L, Nx, order):
    """
    Constructs the Laplacian matrix on a periodic grid.

    Args:
        L: Length of the domain.
        Nx: Number of grid points.
        order: Order of the B-spline (0, 1, or 2).

    Returns:
        Lap: Sparse mass matrix in CSR format.
    """
    dx = L / Nx

    # Construct the gradient matrix G as a circulant matrix with stencil (-1, 1)
    G = construct_G(Nx)

    # Construct the mass matrix M 
    if order == 1:
        M = sp.eye(Nx)
    elif order == 2:
        M = construct_M_linear(Nx)
    elif order == 3:
        M = construct_M_quadratic(Nx)

    # Compute the Laplacian matrix L using M and G
    Lap = G.T @ (M @ G)
    Lap /= dx**2

    return Lap

def binary_filter(Nx):

    B = sp.lil_matrix((Nx, Nx))
    for i in range(Nx):
        B[i, i] = 1/2
        B[i, (i + 1) % Nx] = 1/4
        B[i, (i - 1) % Nx] = 1/4
    B = B.tocsr()

    return B

"""
TESTS
"""

def test_p2g_deposition(order, serial=True, plotting=False):

    L  = 1.0
    A  = 0.5
    Nx = 64
    dx = L / Nx
    x_grid = np.linspace(0, 1, Nx, endpoint=False)

    # Define continuous charge density: ρ(x) = sin(2πx)
    def rho(x): return 1 + A * np.sin(2 * np.pi * x / L)
    max_rho = 1 + A

    # Particle positions generated via inclusion exclusion
    np.random.seed(0)
    Np = 100000
    pos = []

    while len(pos) < Np:
        x = np.random.rand() * L
        if np.random.rand() < rho(x) / max_rho:
            pos.append(x)

    pos = np.array(pos)
    
    # Deposit into grid
    grid = np.zeros(Nx)

    if serial:
        if order == 1:
            p2g_B1(pos, grid, dx)
        elif order == 2:
            p2g_B2(pos, grid, dx)
        elif order == 3:
            p2g_B3(pos, grid, dx)
    else:
        if order == 1:
            p2g_B1_par(pos, grid, dx)
        elif order == 2:
            p2g_B2_par(pos, grid, dx)
        elif order == 3:
            p2g_B3_par(pos, grid, dx)

    # Normalize
    grid *= Nx / Np

    # Exact projection of ρ onto grid
    exact = rho(x_grid)

    # Compare
    error = np.abs(grid - exact)
    print(f"Max error:   {np.max(error):.3e}")
    print(f"L2  error:   {np.linalg.norm(error) / np.sqrt(Nx):.3e}")

    if plotting:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(x_grid, exact, 'k-', lw=1.5, label='Exact ρ(x)')
        plt.plot(x_grid, grid, 'ro', ms=3, label='Deposited')
        plt.title(f"p2g_B{order}: Deposited vs. Exact")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def test_g2p_derivatives(order, serial=True, plotting=False):

    Nx = 128
    dx = 1.0 / Nx
    x_grid = np.linspace(0, 1, Nx, endpoint=False)

    # Grid function: sin(2πx)
    grid = np.sin(2 * np.pi * x_grid)

    # Analytical derivative: 2π cos(2πx)
    def fprime(x): return 2 * np.pi * np.cos(2 * np.pi * x)

    # Particle positions
    np.random.seed(0)
    pos = np.random.rand(200)
    val = np.zeros_like(pos)

    # Choose method here:
    if serial:
        if order == 1:
            g2p_dB1(pos, grid, val, dx)
        elif order == 2:
            g2p_dB2(pos, grid, val, dx)
        elif order == 3:
            g2p_dB3(pos, grid, val, dx)
    else:
        if order == 1:
            g2p_dB1_par(pos, grid, val, dx)
        elif order == 2:
            g2p_dB2_par(pos, grid, val, dx)
        elif order == 3:
            g2p_dB3_par(pos, grid, val, dx)

    # Compare with exact values
    exact = fprime(pos)
    error = np.abs(val - exact)

    print(f"Max error:   {np.max(error):.3e}")
    print(f"L2  error:   {np.linalg.norm(error) / np.sqrt(len(pos)):.3e}")

    # Optional: plot
    if plotting:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(pos, exact, 'k.', label='Exact derivative')
        plt.plot(pos, val, 'ro', ms=3, label='Interpolated')
        plt.title("Comparison of interpolated vs. exact derivative")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def test_full_force_calculation(order, serial=True, plotting=False):

    if plotting:
        import matplotlib.pyplot as plt

    print("Testing full force calculation")
    print("-----------------------------")
    print("Using B-splines of order:", order)
    if serial:
        print("Serial implementation")
    else:
        print("Parallel implementation")

    #================
    # Problem set up
    
    L  = 10.0
    A  = 0.75
    Nx = 51
    dx = L / Nx
    x_grid = np.linspace(0, L, Nx, endpoint=False)

    #================
    # Generate particles

    # Define continuous charge density: ρ(x) = sin(2πx)
    def rho(x): return 1 + A * np.sin(2 * np.pi * x / L)
    max_rho = 1 + A

    # Particle positions generated via inclusion exclusion
    np.random.seed(0)
    Np = 500000
    pos = []

    while len(pos) < Np:
        x = np.random.rand() * L
        if np.random.rand() < rho(x) / max_rho:
            pos.append(x)

    pos = np.array(pos)
    
    #================
    # Deposition
    
    charge = np.zeros(Nx)

    if serial:
        if order == 1:
            p2g_B1(pos, charge, dx)
        elif order == 2:
            p2g_B2(pos, charge, dx)
        elif order == 3:
            p2g_B3(pos, charge, dx)
    else:
        if order == 1:
            p2g_B1_par(pos, charge, dx)
        elif order == 2:
            p2g_B2_par(pos, charge, dx)
        elif order == 3:
            p2g_B3_par(pos, charge, dx)

    # Normalize
    charge *= Nx / Np

    error = np.abs(charge - rho(x_grid))

    print(f"Max error rho: {np.max(error)/np.max(rho(x_grid)):.3e}")
    print(f"L2  error rho: {np.linalg.norm(error) / np.linalg.norm(rho(x_grid)):.3e}")

    # Optional: plot
    if plotting:
        plt.figure(figsize=(8, 4))
        plt.plot(x_grid, rho(x_grid), 'k.', label='Exact charge density')
        plt.plot(x_grid, charge, 'ro', ms=3, label='Interpolated')
        plt.title("Comparison of interpolated vs. exact charge density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    #================
    # Solve the Poisson equation

    Lap = construct_laplacian(L, Nx, order)

    phi_grid = sp.linalg.spsolve(Lap, 1 - charge, permc_spec="MMD_AT_PLUS_A")
    phi_grid -= np.mean(phi_grid) # Remove mean of potential

    def phi(x): return - A * L**2 * np.sin(2 * np.pi * x / L) / (4 * np.pi**2)

    phi_exact = phi(x_grid)

    error = np.abs(phi_grid - phi_exact)

    print(f"Max error phi: {np.max(error)/np.max(phi_exact):.3e}")
    print(f"L2  error phi: {np.linalg.norm(error) / np.linalg.norm(phi_exact):.3e}")

    # Optional: plot
    if plotting:
        plt.figure(figsize=(8, 4))
        plt.plot(x_grid, phi_exact, 'k.', label='Exact potential')
        plt.plot(x_grid, phi_grid, 'ro', ms=3, label='Interpolated')
        plt.title("Comparison of interpolated vs. exact potential")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    #================
    # Force calculation

    force = np.zeros(Np)

    if serial:
        if order == 1:
            g2p_dB1(pos, phi_grid, force, dx)
        elif order == 2:
            g2p_dB2(pos, phi_grid, force, dx)
        elif order == 3:
            g2p_dB3(pos, phi_grid, force, dx)
    else:
        if order == 1:
            g2p_dB1_par(pos, phi_grid, force, dx)
        elif order == 2:
            g2p_dB2_par(pos, phi_grid, force, dx)
        elif order == 3:
            g2p_dB3_par(pos, phi_grid, force, dx)

    #================
    # Compare with exact values

    def force_exact(x): return - A * L * np.cos(2 * np.pi * x / L) / (2 * np.pi)

    error = np.abs(force - force_exact(pos))

    print(f"Max error force:   {np.max(error)/np.max(force_exact(pos)):.3e}")
    print(f"L2  error force:   {np.linalg.norm(error) / np.linalg.norm(force_exact(pos)):.3e}")
    print("")

    # Optional: plot
    if plotting:
        plt.figure(figsize=(8, 4))
        plt.plot(pos, force_exact(pos), 'k.', label='Exact force')
        plt.plot(pos, force, 'ro', ms=3, label='Interpolated')
        plt.title("Comparison of interpolated vs. exact force")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plotting = True
    """
    test_p2g_deposition(1)
    test_p2g_deposition(1, serial=False)
    test_p2g_deposition(2)
    test_p2g_deposition(2, serial=False)
    test_p2g_deposition(3)
    test_p2g_deposition(3, serial=False)
    test_g2p_derivatives(1)
    test_g2p_derivatives(1, serial=False)
    test_g2p_derivatives(2)
    test_g2p_derivatives(2, serial=False)
    test_g2p_derivatives(3)
    test_g2p_derivatives(3, serial=False)
    """
    test_full_force_calculation(1, plotting=plotting)
    test_full_force_calculation(1, serial=False, plotting=plotting)
    test_full_force_calculation(2, plotting=plotting)
    test_full_force_calculation(2, serial=False, plotting=plotting)
    test_full_force_calculation(3, plotting=plotting)
    test_full_force_calculation(3, serial=False, plotting=plotting)
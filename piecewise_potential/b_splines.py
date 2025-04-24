import numpy as np
from numba import njit, prange
from typing import Tuple
import scipy.sparse as sp

# Define potential and deriv functions
@njit
def V(q):
    return np.sin(q) + np.cos(2 * q + 1)**2
@njit
def dVdq(q):
    return np.cos(q) - 2 * np.sin(2 + 4 * q)

@njit
def VB1(q, dq, V_coeffs):

    N = len(V_coeffs)

    q_grid_idx = int(np.floor(q / dq))
    q_grid_frac = (q - q_grid_idx * dq) / dq

    # Linear interpolation
    w0 = 1 - q_grid_frac
    w1 = q_grid_frac

    idx0 = q_grid_idx 
    idx1 = (q_grid_idx + 1) % N

    return (w0 * V_coeffs[idx0] + 
            w1 * V_coeffs[idx1])

@njit
def VB2(q, dq, V_coeffs):

    N = len(V_coeffs)

    grid_idx = int(np.floor(q / dq))
    r = (q - grid_idx * dq) / dq

    # Quadratic interpolation
    w0 = 1/2 * (1 - r)**2
    w1 = 1/2 + r - r**2
    w2 = 1/2 * r**2
    idx0 = (grid_idx - 1) % N
    idx1 = (grid_idx + 0) % N
    idx2 = (grid_idx + 1) % N

    return (w0 * V_coeffs[idx0] + 
            w1 * V_coeffs[idx1] +
            w2 * V_coeffs[idx2])

@njit
def VB3(q, dq, V_coeffs):

    N = len(V_coeffs)
    grid_idx = int(np.floor(q / dq))
    r = (q - grid_idx * dq) / dq

    w0 = (1/6) * (1 - r)**3
    w1 = (1/6) * (3 * r**3 - 6 * r**2 + 4)
    w2 = (1/6) * (-3 * r**3 + 3 * r**2 + 3 * r + 1)
    w3 = (1/6) * r**3

    idx0 = (grid_idx - 1) % N
    idx1 = grid_idx % N
    idx2 = (grid_idx + 1) % N
    idx3 = (grid_idx + 2) % N

    return (w0 * V_coeffs[idx0] + 
            w1 * V_coeffs[idx1] +
            w2 * V_coeffs[idx2] +
            w3 * V_coeffs[idx3])

@njit
def dVB1(q, dq, V_coeffs):

    N = len(V_coeffs)

    q_grid_idx = int(np.floor(q / dq))

    # Linear interpolation
    dw0 = -1/dq
    dw1 = 1/dq

    idx0 = q_grid_idx 
    idx1 = (q_grid_idx + 1) % N

    return (dw0 * V_coeffs[idx0] + 
            dw1 * V_coeffs[idx1])

@njit
def dVB2(q, dq, V_coeffs):

    N = len(V_coeffs)

    grid_idx = int(np.floor(q / dq))
    r = (q - grid_idx * dq) / dq

    # Quadratic interpolation
    dw0 = (r - 1)/dq
    dw1 = (1 - 2 * r)/dq
    dw2 = r/dq

    idx0 = (grid_idx - 1) % N
    idx1 = (grid_idx + 0) % N
    idx2 = (grid_idx + 1) % N

    return (dw0 * V_coeffs[idx0] + 
            dw1 * V_coeffs[idx1] +
            dw2 * V_coeffs[idx2])

@njit
def dVB3(q, dq, V_coeffs):    

    N = len(V_coeffs)
    grid_idx = int(np.floor(q / dq))
    r = (q - grid_idx * dq) / dq

    w0 = -1/2 * (1-r)**2 / dq
    w1 = (1/6) * (9 * r**2 - 12 * r) / dq
    w2 = (1/6) * (-9 * r**2 + 6 * r + 3) / dq
    w3 = (1/2) * r**2 / dq

    idx0 = (grid_idx - 1) % N
    idx1 = grid_idx % N
    idx2 = (grid_idx + 1) % N
    idx3 = (grid_idx + 2) % N

    return (w0 * V_coeffs[idx0] + 
            w1 * V_coeffs[idx1] +
            w2 * V_coeffs[idx2] +
            w3 * V_coeffs[idx3])

def compute_bspline_coefficients(V_samples, N, order):
    """Compute B-spline coefficients to approximate V(q)."""

    if order == 1:
        # Linear: Coefficients are just the samples
        return V_samples

    elif order == 2:
        # Quadratic B-splines Vandermonde matrix
        M = sp.diags([1/2, 1/2], [-1, 0], shape=(N, N), format='lil')
        # Periodic boundary condition
        M[0, -1] = 1/2
        M = M.tocsc()
        return sp.linalg.spsolve(M, V_samples)

    elif order == 3:
        # Cubic B-splines Vandermonde matrix
        M = sp.diags([1/6, 2/3, 1/6], [-1, 0, 1], shape=(N, N), format='lil')
        # Periodic boundary conditions
        M[0, -1] = 1/6
        M[-1, 0] = 1/6
        M = M.tocsc()
        return sp.linalg.spsolve(M, V_samples)

    else:
        raise ValueError("Unsupported B-spline order.")

class BSplinePotentialInterpolator:
    def __init__(self, V_func, domain, num_gridpoints, order):
        """
        Initialize the B-spline interpolator.

        Args:
            V_func: Callable function V(q) returning the potential.
            domain: Tuple (q_min, q_max) specifying the domain boundaries.
            num_gridpoints: Number of gridpoints (excluding the endpoint).
            order: Order of B-spline interpolation (1, 2, or 3).
        """
        self.V_func = V_func
        self.q_min, self.q_max = domain
        self.num_gridpoints = num_gridpoints
        self.order = order

        # Set up grid (omit endpoint)
        self.dq = (self.q_max - self.q_min) / self.num_gridpoints
        self.grid = np.linspace(self.q_min, self.q_max, 
                                self.num_gridpoints, endpoint=False)

        # Sample the potential at the grid points
        self.V_samples = self.V_func(self.grid)

        # Compute B-spline coefficients
        self.coeffs = self.compute_bspline_coefficients()

        # Interpolation function
        if self.order == -1:
            self.evaluate_potential = V
            self.evaluate_deriv = dVdq
        if self.order == 1:
            self.evaluate_potential = self.evaluate_potential_linear
            self.evaluate_deriv = self.evaluate_deriv_linear
        elif self.order == 2:
            self.evaluate_potential = self.evaluate_potential_quadratic
            self.evaluate_deriv = self.evaluate_deriv_quadratic
        elif self.order == 3:
            self.evaluate_potential = self.evaluate_potential_cubic
            self.evaluate_deriv = self.evaluate_deriv_cubic

    def compute_bspline_coefficients(self):
        """Compute B-spline coefficients to approximate V(q)."""
        N = self.num_gridpoints

        if self.order == 1:
            # Linear: Coefficients are just the samples
            return self.V_samples

        elif self.order == 2:
            # Quadratic B-splines Vandermonde matrix
            M = sp.diags([1/2, 1/2], [-1, 0], shape=(N, N), format='lil')
            # Periodic boundary condition
            M[0, -1] = 1/2
            M = M.tocsc()
            return sp.linalg.spsolve(M, self.V_samples)

        elif self.order == 3:
            # Cubic B-splines Vandermonde matrix
            M = sp.diags([1/6, 2/3, 1/6], [-1, 0, 1], shape=(N, N), format='lil')
            # Periodic boundary conditions
            M[0, -1] = 1/6
            M[-1, 0] = 1/6
            M = M.tocsc()
            return sp.linalg.spsolve(M, self.V_samples)

        else:
            raise ValueError("Unsupported B-spline order.")

    def evaluate_potential_linear(self, q):
        """Evaluate the interpolated potential at position q."""
        
        return VB1(q, self.dq, self.coeffs)
    
    def evaluate_potential_quadratic(self, q):
        """Evaluate the interpolated potential at position q."""
        
        return VB2(q, self.dq, self.coeffs)
    
    def evaluate_potential_cubic(self, q):
        """Evaluate the interpolated potential at position q."""
        
        return VB3(q, self.dq, self.coeffs)
    
    def evaluate_potential(self, q):
        """Evaluate the interpolated potential at position q."""
        return V(q)

    def evaluate_deriv_linear(self, q):
        """Evaluate the derivative at position q."""
        
        return dVB1(q, self.dq, self.coeffs)
    
    def evaluate_deriv_quadratic(self, q):
        """Evaluate the derivative at position q."""
        
        return dVB2(q, self.dq, self.coeffs)
    
    def evaluate_deriv_cubic(self, q):
        """Evaluate the derivative at position q."""
        
        return dVB3(q, self.dq, self.coeffs)
    
    def evaluate_deriv(self, q):
        """Evaluate the derivative at position q."""
        return dVdq(q)

def plot_bspline_interpolation(order: int, num_points=10, domain=(0, 2 * np.pi)):
    import matplotlib.pyplot as plt

    # Evaluation points
    q_vals = np.linspace(domain[0], domain[1], 10 * num_points, endpoint=False)

    # Build B-spline interpolator object
    potential_interpolator = BSplinePotentialInterpolator(V, domain, num_points, order)
    
    # Interpolated and exact values
    V_interp  = np.array([potential_interpolator.evaluate_potential(q) for q in q_vals])
    dV_interp = np.array([potential_interpolator.evaluate_deriv(q) for q in q_vals])
    V_exact  = V(q_vals)
    dV_exact = dVdq(q_vals)
    
    # Compute L2 errors
    V_error = np.sqrt(np.mean((V_interp - V_exact) ** 2))
    dV_error = np.sqrt(np.mean((dV_interp - dV_exact) ** 2))

    print(f"Interpolation order: {order}")
    print(f"L2 error in potential     = {V_error:.3e}")
    print(f"L2 error in deriv (dV/dq) = {dV_error:.3e}")

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(q_vals, V_exact, label="Exact V(q)", color="black", linewidth=2)
    plt.plot(q_vals, V_interp, label=f"Interpolated V(q) (order={order})", linestyle="--")
    plt.xlabel("q")
    plt.ylabel("Potential V(q)")
    plt.title("Potential Comparison")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(q_vals, dV_exact, label="Exact dV/dq", color="black", linewidth=2)
    plt.plot(q_vals, dV_interp, label=f"Interpolated dV/dq (order={order})", linestyle="--")
    plt.xlabel("q")
    plt.ylabel("deriv dV/dq")
    plt.title("deriv Comparison")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(q_vals, np.abs(V_exact - V_interp), color="black", linewidth=2)
    plt.xlabel("q")
    plt.ylabel("Error in Potential V(q)")
    plt.title("Potential Error")

    plt.subplot(2, 2, 4)
    plt.plot(q_vals, np.abs(dV_exact - dV_interp), color="black", linewidth=2)
    plt.xlabel("q")
    plt.ylabel("Error in deriv dV/dq")
    plt.title("deriv Error")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    plot_bspline_interpolation(1, 51)
    plot_bspline_interpolation(2, 51)
    plot_bspline_interpolation(3, 51)
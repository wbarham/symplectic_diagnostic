import numpy as np
from numba import njit, prange
from typing import Tuple

from b_splines import (compute_bspline_coefficients,
                                       VB1, VB2, VB3,
                                       dVB1, dVB2, dVB3)

# Define potential and force functions
@njit
def V(q):
    return np.sin(q) + np.cos(2 * q + 1)**2

@njit
def dVdq(q):
    return np.cos(q) - 2 * np.sin(2 + 4 * q)

@njit(parallel=True)
def strang_step(q: np.ndarray, p: np.ndarray, dVdq: callable, period: float, dt: float):
    """Parallel Strang-splitting for multiple trajectories """
    for i in prange(q.shape[0]):
         q[i] = (q[i] + p[i] * dt/2) % period
         p[i] -= dVdq(q[i]) * dt
         q[i] = (q[i] + p[i] * dt/2) % period

@njit(parallel=True)
def rk2_step(q: np.ndarray, p: np.ndarray, dVdq: callable, period: float, dt: float):
    """Parallel RK2 for multiple trajectories"""
    for i in prange(q.shape[0]):
        # Stage 1
        F1 = -dVdq(q[i])
        q_temp = (q[i] + p[i] * dt/2) % period
        p_temp = p[i] + F1 * dt/2
        
        # Stage 2
        F2 = -dVdq(q_temp)
        q[i] = (q[i] + p_temp * dt) % period
        p[i] = p[i] + F2 * dt

@njit(parallel=True)
def compute_energy(q: np.ndarray, p: np.ndarray, V: callable, H: np.ndarray):
    """Parallel energy computation for multiple trajectories"""
    for i in prange(q.shape[0]):
        H[i] = 0.5 * p[i]**2 + V(q[i])

def create_wrapper(f_eval, dq, coeffs):
    @njit
    def f_wrapper(q):
        return f_eval(q, dq, coeffs)
    return f_wrapper

class PiecewisePotential:
    def __init__(self, V_func, dVdq_func,order: int, domain=(0, 2 * np.pi), num_gridpoints=2000):

        # Require order = 1, 2, or 3
        if order not in [0, 1, 2, 3]:
            raise ValueError("Interpolation order must be 0, 1, 2, or 3")

        # Store internals
        self.V_func = V_func
        self.order = order
        
        # Set up domain
        self.q_min, self.q_max = domain
        self.period = self.q_max - self.q_min
        self.num_gridpoints = num_gridpoints
        self.dq = (self.q_max - self.q_min) / self.num_gridpoints
        self.grid = np.linspace(self.q_min, self.q_max, 
                                self.num_gridpoints, endpoint=False)

        # Sample the potential at the grid points
        self.V_samples = self.V_func(self.grid)

        # Compute B-spline coefficients
        if order != 0:
            self.coeffs = compute_bspline_coefficients(self.V_samples, self.num_gridpoints, self.order)

        # Set up evaluation methods
        if order == 0:
            self.V_eval    = V_func
            self.dVdq_eval = dVdq_func
        elif order == 1:
            self.V_eval    = VB1
            self.dVdq_eval = dVB1
        elif order == 2:
            self.V_eval    = VB2
            self.dVdq_eval = dVB2
        elif order == 3:
            self.V_eval    = VB3
            self.dVdq_eval = dVB3

        # Create the dVdq wrapper function
        if order != 0:
            self.dVdq_func = create_wrapper(self.dVdq_eval, self.dq, self.coeffs)
            self.V_func    = create_wrapper(self.V_eval, self.dq, self.coeffs)
        else:
            self.dVdq_func = self.dVdq_eval
            self.V_func    = self.V_eval
        
    def strang_step(self, q, p, dt):
        """
        Perform a Strang splitting step.

        Args:
            q: Position (Ns,)
            p: Momentum (Ns,)
            dt: Time step
        """

        strang_step(q, p, self.dVdq_func, self.period, dt)

    def rk2_step(self, q, p, dt):
        """
        Perform a Runge-Kutta 2 step.

        Args:
            q: Position (Ns,)
            p: Momentum (Ns,)
            dt: Time step
        """

        rk2_step(q, p, self.dVdq_func, self.period, dt)

    def compute_trajectory(self, q_init, p_init, dt, num_steps, method="strang"):
        """
        Compute a trajectory from initial conditions.

        Args:
            q_init: Initial q values (N,)
            p_init: Initial p values (N,)
            dt: Time step
            num_steps: Number of steps
        Returns:
            q_traj: (N, num_steps+1) array of q trajectories
            p_traj: (N, num_steps+1) array of p trajectories
            energy: (N, num_steps+1) array of energies
        """

        # Choose integrator
        if method == "strang":
            self.stepper = self.strang_step
        elif method == "rk2":
            self.stepper = self.rk2_step
        else:
            raise ValueError("Method must be 'strang' or 'rk2'")

        num_trajectories = q_init.shape[0]
        q_traj = np.zeros((num_trajectories, num_steps + 1))
        p_traj = np.zeros((num_trajectories, num_steps + 1))
        energy = np.zeros((num_trajectories, num_steps + 1))
        
        # Initialize
        q_traj[:, 0] = q_init
        p_traj[:, 0] = p_init

        # Compute initial energy
        compute_energy(q_init, p_init, self.V_func, energy[:,0])

        # Time stepping
        for step in range(1, num_steps + 1):
            # Make copies of current state
            q_current = q_traj[:, step-1].copy()
            p_current = p_traj[:, step-1].copy()
            
            # Perform step (modifies arrays in-place)
            self.stepper(q_current, p_current, dt)

            # Compute energy
            compute_energy(q_current, p_current, self.V_func, energy[:,step])
            
            # Store results
            q_traj[:, step] = q_current
            p_traj[:, step] = p_current
        
        return q_traj, p_traj, energy

# Example usage
def main():
    # Simulation parameters
    import matplotlib.pyplot as plt

    num_trajectories = 5     # Fewer trajectories for clearer visualization
    num_gridpoints   = 101    # Number of grid points
    dt = 0.1                # Time step
    num_steps = 500          # Number of steps
    method = "strang"        # Set to "strang" or "rk2"
    
    # Initial conditions (circular in phase space)
    center = (np.pi/2, 0)      # Center point (q, p)
    radius = 1.0             # Radius of initial circle
    angles = 2 * np.pi * np.arange(num_trajectories) / num_trajectories
    q_init = center[0] + radius * np.cos(angles)
    p_init = center[1] + radius * np.sin(angles)
    
    # Colors and labels for plotting
    colors = ['b', 'g', 'r', 'm']
    labels = ["Exact", "Linear", "Quadratic", "Cubic"]
    
    # Run simulations for each interpolation order
    results = []
    for order in [0, 1, 2, 3]:
        # Create integrator
        integrator = PiecewisePotential(V, dVdq, order, num_gridpoints=num_gridpoints)
        q_traj, p_traj, energy = integrator.compute_trajectory(q_init, p_init, dt, num_steps, method=method)
        results.append((q_traj, p_traj, energy))
    
    # Create time array
    time = np.arange(num_steps + 1) * dt
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # 1. Phase Space Trajectories
    plt.subplot(2, 2, 1)
    for i, (color, label, (q, p, _)) in enumerate(zip(colors, labels, results)):
        # Plot just the first trajectory for clarity
        plt.plot(np.unwrap(q[0]), p[0], color, label=label, alpha=0.7)
    plt.xlabel("Position (q)")
    plt.ylabel("Momentum (p)")
    plt.title("Phase Space Trajectories")
    plt.grid(True)
    plt.legend()
    
    # 2. Energy Conservation
    plt.subplot(2, 2, 2)
    for color, label, (_, _, energy) in zip(colors, labels, results):
        # Plot energy deviation for first trajectory
        energy_dev = np.abs(energy[0] - energy[0, 0])
        plt.semilogy(time, energy_dev, color, label=label)
    plt.xlabel("Time")
    plt.ylabel("Energy Deviation")
    plt.title("Energy Conservation (Semilog)")
    plt.grid(True)
    plt.legend()
    
    # 3. Position Time Series
    plt.subplot(2, 2, 3)
    for color, label, (q, _, _) in zip(colors, labels, results):
        plt.plot(time, np.unwrap(q[0]), color, label=label, alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Position (q)")
    plt.title("Position vs Time")
    plt.grid(True)
    plt.legend()
    
    # 4. Final Energy Errors
    plt.subplot(2, 2, 4)
    final_errors = []
    for (_, _, energy) in results:
        # Compute max relative error across all trajectories
        rel_error = np.max(np.abs(energy - energy[:, 0:1]) / np.abs(energy[:, 0:1]))
        final_errors.append(rel_error)
    plt.bar(labels, final_errors, color=colors)
    plt.yscale('log')
    plt.ylabel("Max Relative Energy Error")
    plt.title("Final Energy Errors")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("pendulum_comparison.png", dpi=150)
    plt.show()
    
    # Print summary statistics
    print("\n=== Energy Conservation Summary ===")
    print(f"Method: {method.upper()} | dt: {dt} | dq: {2 * np.pi / num_gridpoints}")
    print("----------------------------------")
    for label, error in zip(labels, final_errors):
        print(f"{label:<9}: {error:.3e}")

if __name__ == "__main__":
    main()

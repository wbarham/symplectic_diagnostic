import numpy as np
from numba import njit, prange
from typing import Tuple

# Define potential and force functions
V = np.sin
dVdq = np.cos

# Core single-point computations (serial)
@njit
def compute_potential(q: float, dq: float, interpolation_order: int) -> float:
    """Compute potential at single point"""

    if interpolation_order == 0 or dq == 0:
        return V(q)
    
    q_grid_idx = int(np.floor(q / dq))
    q_grid_frac = (q - q_grid_idx * dq) / dq
    
    if interpolation_order == 1:
        # Linear interpolation
        w0 = 1 - q_grid_frac
        w1 = q_grid_frac
        return w0 * V(q_grid_idx * dq) + w1 * V((q_grid_idx + 1) * dq)
    
    elif interpolation_order == 2:
        # Quadratic interpolation
        w0 = q_grid_frac * (q_grid_frac - 1) / 2
        w1 = -(q_grid_frac + 1) * (q_grid_frac - 1)
        w2 = (q_grid_frac + 1) * q_grid_frac / 2
        return (w0 * V((q_grid_idx - 1) * dq) + 
                w1 * V(q_grid_idx * dq) + 
                w2 * V((q_grid_idx + 1) * dq))
    
    elif interpolation_order == 3:
        # Cubic interpolation
        w0 = -(q_grid_frac + 1) * q_grid_frac * (q_grid_frac - 1) / 6
        w1 = (q_grid_frac + 2) * q_grid_frac * (q_grid_frac - 1) / 2
        w2 = -(q_grid_frac + 2) * (q_grid_frac + 1) * (q_grid_frac - 1) / 2
        w3 = (q_grid_frac + 2) * (q_grid_frac + 1) * q_grid_frac / 6
        return (w0 * V((q_grid_idx - 2) * dq) +
                w1 * V((q_grid_idx - 1) * dq) +
                w2 * V(q_grid_idx * dq) +
                w3 * V((q_grid_idx + 1) * dq))
    else:
        raise ValueError("Invalid interpolation order")

@njit
def compute_force(q: float, dq: float, interpolation_order: int) -> float:
    """Compute force at single point"""

    if interpolation_order == 0 or dq == 0:
        return -(dVdq(q))
    
    q_grid_idx = int(np.floor(q / dq))
    q_grid_frac = (q - q_grid_idx * dq) / dq
    
    if interpolation_order == 1:
        # Linear interpolation derivative
        return -(V((q_grid_idx + 1) * dq) - V(q_grid_idx * dq)) / dq
    
    elif interpolation_order == 2:
        # Quadratic interpolation derivative
        dw0 = (2 * q_grid_frac - 1) / (2 * dq)
        dw1 = -2 * q_grid_frac / dq
        dw2 = (2 * q_grid_frac + 1) / (2 * dq)
        return -(dw0 * V((q_grid_idx - 1) * dq) +
                dw1 * V(q_grid_idx * dq) +
                dw2 * V((q_grid_idx + 1) * dq))
    
    elif interpolation_order == 3:
        # Cubic interpolation derivative
        dw0 = (3 * q_grid_frac**2 - 1) / (6 * dq)
        dw1 = (2 - 2 * q_grid_frac - 3 * q_grid_frac**2) / (2 * dq)
        dw2 = -(1 - 4 * q_grid_frac - 3 * q_grid_frac**2) / (2 * dq)
        dw3 = -(2 + 6 * q_grid_frac + 3 * q_grid_frac**2) / (6 * dq)
        return (dw0 * V((q_grid_idx - 2) * dq) +
                dw1 * V((q_grid_idx - 1) * dq) +
                dw2 * V(q_grid_idx * dq) +
                dw3 * V((q_grid_idx + 1) * dq))
    else:
        raise ValueError("Invalid interpolation order")

@njit
def kinetic_update(q: float, p: float, dt: float) -> Tuple[float, float]:
    """Update single point using kinetic part"""
    return q + p * dt, p

@njit
def potential_update(q: float, p: float, dt: float, 
                   dq: float, interpolation_order: int) -> Tuple[float, float]:
    """Update single point using potential part"""
    F_q = compute_force(q, dq, interpolation_order)
    return q, p + F_q * dt

# Parallelized time steppers
@njit(parallel=True)
def strang_step_batch(q: np.ndarray, p: np.ndarray, dt: float, 
                     dq: float, interpolation_order: int):
    """Parallel Strang splitting for multiple trajectories"""
    for i in prange(q.shape[0]):
        # Half-step kinetic
        q[i], p[i] = kinetic_update(q[i], p[i], dt/2)
        # Full-step potential
        q[i], p[i] = potential_update(q[i], p[i], dt, dq, interpolation_order)
        # Half-step kinetic
        q[i], p[i] = kinetic_update(q[i], p[i], dt/2)

@njit(parallel=True)
def rk2_step_batch(q: np.ndarray, p: np.ndarray, dt: float,
                  dq: float, interpolation_order: int):
    """Parallel RK2 for multiple trajectories"""
    for i in prange(q.shape[0]):
        # Stage 1
        F1 = compute_force(q[i], dq, interpolation_order)
        q_temp = q[i] + p[i] * dt/2
        p_temp = p[i] + F1 * dt/2
        
        # Stage 2
        F2 = compute_force(q_temp, dq, interpolation_order)
        q[i] = q[i] + p_temp * dt
        p[i] = p[i] + F2 * dt

@njit(parallel=True)
def compute_energy(q: np.ndarray, p: np.ndarray, H: np.ndarray, 
                   dq: float, interpolation_order: int):

    for i in prange(q.shape[0]):
        Vi = compute_potential(q[i], dq, interpolation_order)
        H[i] = 0.5 * p[i]**2 + Vi

@njit
def compute_trajectories_batch(q_init: np.ndarray, p_init: np.ndarray,
                             dt: float, num_steps: int,
                             dq: float = 0.0, interpolation_order: int = 0,
                             method: str = "strang"):
    """
    Compute multiple trajectories in parallel with optional interpolation.
    
    Args:
        q_init: Initial q values (N,)
        p_init: Initial p values (N,)
        dt: Time step
        num_steps: Number of steps
        dq: Grid spacing
        interpolation_order: 0=exact, 1=linear, 2=quadratic, 3=cubic
        method: "strang" or "rk2"
        
    Returns:
        q_traj: (N, num_steps+1) array of q trajectories
        p_traj: (N, num_steps+1) array of p trajectories
        energy: (N, num_steps+1) array of energies
    """
    num_trajectories = q_init.shape[0]
    q_traj = np.zeros((num_trajectories, num_steps + 1))
    p_traj = np.zeros((num_trajectories, num_steps + 1))
    energy = np.zeros((num_trajectories, num_steps + 1))
    
    # Initialize
    q_traj[:, 0] = q_init
    p_traj[:, 0] = p_init
    
    # Compute initial energy
    compute_energy(q_init, p_init, energy[:,0], dq, interpolation_order)
    
    # Choose integrator
    if method == "strang":
        step_func = strang_step_batch
    elif method == "rk2":
        step_func = rk2_step_batch
    else:
        raise ValueError("Method must be 'strang' or 'rk2'")
    
    # Time stepping
    for step in range(1, num_steps + 1):
        # Make copies of current state
        q_current = q_traj[:, step-1].copy()
        p_current = p_traj[:, step-1].copy()
        
        # Perform step (modifies arrays in-place)
        step_func(q_current, p_current, dt, dq, interpolation_order)

        # Compute energy
        compute_energy(q_current, p_current, energy[:,step], dq, interpolation_order)
        
        # Store results
        q_traj[:, step] = q_current
        p_traj[:, step] = p_current
    
    return q_traj, p_traj, energy

# Example usage
def main():
    # Simulation parameters
    import matplotlib.pyplot as plt

    num_trajectories = 5      # Fewer trajectories for clearer visualization
    dt = 0.01                 # Time step
    num_steps = 500          # Number of steps
    dq = 0.1                 # Coarse-graining spacing
    method = "strang"        # Integration method ("strang" or "rk2")
    
    # Initial conditions (circular in phase space)
    center = (np.pi, 0)      # Center point (q, p)
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
        q_traj, p_traj, energy = compute_trajectories_batch(
            q_init, p_init, dt, num_steps, 
            dq=dq, interpolation_order=order, method=method
        )
        results.append((q_traj, p_traj, energy))
    
    # Create time array
    time = np.arange(num_steps + 1) * dt
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # 1. Phase Space Trajectories
    plt.subplot(2, 2, 1)
    for i, (color, label, (q, p, _)) in enumerate(zip(colors, labels, results)):
        # Plot just the first trajectory for clarity
        plt.plot(q[0], p[0], color, label=label, alpha=0.7)
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
        plt.plot(time, q[0], color, label=label, alpha=0.7)
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
    print(f"Method: {method.upper()} | dt: {dt} | dq: {dq}")
    print("----------------------------------")
    for label, error in zip(labels, final_errors):
        print(f"{label:<9}: {error:.3e}")

if __name__ == "__main__":
    main()
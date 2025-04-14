import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_forces_batch_parallel(positions, forces, alpha, k, N, M):
    """
    Compute forces for all trajectories in parallel.
    """     
    for i in prange(N): 
        for j in prange(M):
            left  = (j - 1) % N
            right = (j + 1) % N
            forces[i, j] = k * (positions[i, right] - 2 * positions[i, j] + positions[i, left]) \
                         + alpha * np.sin(positions[i, j])

@njit(parallel=True)
def symplectic_step_batch_parallel(positions, velocities, forces, dt, alpha, k, N, M):
    """
    Perform symplectic steps for all trajectories in parallel.
    """
    # Half-step velocity update and full position update
    for i in prange(N):
        for j in prange(M):
            velocities[i,j] += 0.5 * forces[i,j] * dt
            positions[i,j]  += velocities[i,j] * dt
    
    # Compute new forces
    compute_forces_batch_parallel(positions, forces, alpha, k, N, M)
    
    # Another half-step velocity update
    for i in prange(N):
        for j in prange(M):
            velocities[i,j] += 0.5 * forces[i,j] * dt

@njit(parallel=True)
def euler_step_batch_parallel(positions, velocities, forces, dt, alpha, k, N, M):
    """
    Perform Euler steps for all trajectories in parallel.
    """
    # Full-step position update
    for i in prange(N):
        for j in prange(M):
            positions[i,j]  += velocities[i,j] * dt
            velocities[i,j] += forces[i,j] * dt
    
    # Compute new forces
    compute_forces_batch_parallel(positions, forces, alpha, k, N, M)

@njit(parallel=True)
def rk2_step_batch_parallel(positions, velocities, forces, dt, alpha, k, N, M):
    """
    Perform RK2 steps for all trajectories in parallel.
    """
    # Create temporary arrays
    temp_pos = np.empty_like(positions)
    temp_vel = np.empty_like(velocities)
    k1_force = np.empty_like(forces)
    k2_force = np.empty_like(forces)
    
    # First stage (k1)
    compute_forces_batch_parallel(positions, k1_force, alpha, k, N, M)
    
    for i in prange(N):
        for j in prange(M):
            # Compute intermediate position and velocity
            temp_pos[i,j] = positions[i,j] + 0.5 * velocities[i,j] * dt
            temp_vel[i,j] = velocities[i,j] + 0.5 * k1_force[i,j] * dt
    
    # Second stage (k2)
    compute_forces_batch_parallel(temp_pos, k2_force, alpha, k, N, M)
    
    for i in prange(N):
        for j in prange(M):
            # Final update
            positions[i,j] += velocities[i,j] * dt
            velocities[i,j] += k2_force[i,j] * dt
    
    # Update forces for next step
    compute_forces_batch_parallel(positions, forces, alpha, k, N, M)

# Example usage if main script
if __name__ == "__main__":

    # Import matplotlib for visualization
    import matplotlib.pyplot as plt 

    import sys
    from pathlib import Path
    import argparse

    # Import from parent directory
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from loop_integrals import LoopIntegral     # The class you provided

    # System parameters
    N = 2**10  # Number of masses per trajectory
    M = 2**10  # Number of trajectories
    rad = 0.5
    k = 1.0    # Spring constant
    alpha = 1.0 # Pendulum coefficient
    dt = 0.1   # Time step
    nsteps = 100 # Number of integration steps

    # Set up initial conditions
    np.random.seed(42)
    random_init = np.random.rand(N, 2)

    # Create identical initial conditions for both integrators
    def initialize():
        loop_q = np.array([
            [random_init[i, 0] + rad * np.cos(2 * np.pi * j / M) for j in range(M)] 
            for i in range(N)
        ])
        loop_p = np.array([
            [random_init[i, 1] + rad * np.sin(2 * np.pi * j / M) for j in range(M)]
            for i in range(N)
        ])
        forces = np.zeros((N, M))
        compute_forces_batch_parallel(loop_q, forces, alpha, k, N, M)
        return loop_q.copy(), loop_p.copy(), forces.copy()

    # Initialize both systems
    q_symp, p_symp, f_symp = initialize()
    q_rk2, p_rk2, f_rk2 = initialize()

    # Setup loop integral calculator
    loop_integral = LoopIntegral(M)
    t = np.linspace(0, nsteps * dt, nsteps)
    
    # Arrays to store results
    int_symp = np.zeros(nsteps)
    int_rk2 = np.zeros(nsteps)

    # Compute initial loop integrals
    int_symp[0] = loop_integral.compute(q_symp, p_symp)
    int_rk2[0] = loop_integral.compute(q_rk2, p_rk2)

    # Run both integrators
    for i in range(1, nsteps):
        # Symplectic integration
        symplectic_step_batch_parallel(q_symp, p_symp, f_symp, dt, alpha, k, N, M)
        int_symp[i] = loop_integral.compute(q_symp, p_symp)
        
        # RK2 integration
        rk2_step_batch_parallel(q_rk2, p_rk2, f_rk2, dt, alpha, k, N, M)
        int_rk2[i] = loop_integral.compute(q_rk2, p_rk2)

        if i % 10 == 0:
            print(f"Completed {i}/{nsteps} steps")

    # Calculate relative errors with small offset
    tiny = 1e-16 # Added to make log-scale plot more readable
    rel_error_symp = np.abs((int_symp - int_symp[0]) / (np.abs(int_symp[0]))) + tiny
    rel_error_rk2 = np.abs((int_rk2 - int_rk2[0]) / (np.abs(int_rk2[0]))) + tiny

    # Set up plot
    plt.figure()
    plt.semilogy(t, rel_error_symp, 'b-', label='Strang', linewidth=2)
    plt.semilogy(t, rel_error_rk2, 'r--', label='RK2', linewidth=2)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Relative Error in Loop Integral', fontsize=12)
    plt.title(rf'$N=${N}, $M=${M}, $\Delta t=${dt}', fontsize=14)
    
    #plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save and show
    plt.savefig('nl_pend_array_time_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
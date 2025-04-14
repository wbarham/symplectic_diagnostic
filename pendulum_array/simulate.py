#!/usr/bin/env python3
import numpy as np
from numba import njit, prange
import argparse
import sys
from pathlib import Path

# Import integrator functions from parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from pendulum_array import (compute_forces_batch_parallel,
                              symplectic_step_batch_parallel,
                              rk2_step_batch_parallel)
from loop_integrals import LoopIntegral

def initialize_system(N, M, rad, k, alpha, dt, seed=42):
    """Initialize system with given parameters"""
    np.random.seed(seed)
    random_init = np.random.rand(N, 2)
    
    loop_q = np.array([
        [random_init[i, 0] + rad * np.sin(2 * np.pi * j / M) for j in range(M)] 
        for i in range(N)
    ])
    loop_p = np.array([
        [random_init[i, 1] + rad * np.cos(2 * np.pi * j / M) for j in range(M)]
        for i in range(N)
    ])
    forces = np.zeros((N, M))
    compute_forces_batch_parallel(loop_q, forces, alpha, k, N, M)
    return loop_q, loop_p, forces

def run_simulation(params):
    """Run single step simulation with given parameters"""
    # Initialize systems
    q_symp, p_symp, f_symp = initialize_system(**params)
    q_rk2, p_rk2, f_rk2 = initialize_system(**params)
    
    # Setup calculator
    loop_integral = LoopIntegral(params['M'])
    
    # Compute initial integrals
    init_symp = loop_integral.compute(q_symp, p_symp)
    init_rk2 = loop_integral.compute(q_rk2, p_rk2)
    
    # Single step integration
    symplectic_step_batch_parallel(q_symp, p_symp, f_symp, params['dt'], 
                                 params['alpha'], params['k'], params['N'], params['M'])
    rk2_step_batch_parallel(q_rk2, p_rk2, f_rk2, params['dt'],
                           params['alpha'], params['k'], params['N'], params['M'])
    
    # Compute final integrals
    final_symp = loop_integral.compute(q_symp, p_symp)
    final_rk2 = loop_integral.compute(q_rk2, p_rk2)
    
    # Calculate relative errors
    rel_err_symp = np.abs((final_symp - init_symp) / (np.abs(init_symp)))
    rel_err_rk2 = np.abs((final_rk2 - init_rk2) / (np.abs(init_rk2)))
    
    return {
        'rel_err_symp': rel_err_symp,
        'rel_err_rk2': rel_err_rk2,
        'init_integral': init_symp
    }

def main():
    parser = argparse.ArgumentParser(description='Single-step pendulum array simulation')
    parser.add_argument('--N', type=int, default=1024, help='Number of masses')
    parser.add_argument('--M', type=int, default=1024, help='Number of trajectories')
    parser.add_argument('--rad', type=float, default=1.0, help='Initial radius')
    parser.add_argument('--k', type=float, default=1.0, help='Spring constant')
    parser.add_argument('--alpha', type=float, default=1.0, help='Nonlinearity coefficient')
    parser.add_argument('--dt', type=float, default=0.5, help='Time step size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Convert args to dict
    params = vars(args)
    
    # Run simulation
    results = run_simulation(params)
    
    # Format output for easy regex parsing
    output = f"""
=== Simulation Parameters ===
N: {params['N']}
M: {params['M']}
radius: {params['rad']}
k: {params['k']}
alpha: {params['alpha']}
dt: {params['dt']}
seed: {params['seed']}

=== Results ===
Initial loop integral: {results['init_integral']}
Relative error (Symplectic): {results['rel_err_symp']}
Relative error (RK2): {results['rel_err_rk2']}
=== End of Simulation ===
"""
    print(output)

if __name__ == "__main__":
    main()
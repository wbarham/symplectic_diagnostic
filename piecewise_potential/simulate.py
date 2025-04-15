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
from piecewise_potential import (strang_step_batch, rk2_step_batch)
from loop_integrals import LoopIntegral

def initialize_system(q0, p0, rad, M, dq, interpolation_order, dt):
    """Initialize system with given parameters"""
    
    loop_q = np.array([q0 + rad * np.sin(2 * np.pi * j / M) for j in range(M)])
    loop_p = np.array([p0 + rad * np.cos(2 * np.pi * j / M) for j in range(M)])

    return loop_q, loop_p

def run_simulation(params):
    """Run single step simulation with given parameters"""
    # Initialize systems
    q_symp, p_symp = initialize_system(**params)
    q_rk2, p_rk2 = initialize_system(**params)
    
    # Setup calculator
    loop_integral = LoopIntegral(params['M'])
    
    # Compute initial integrals
    init_symp = loop_integral.compute(q_symp, p_symp)
    init_rk2 = loop_integral.compute(q_rk2, p_rk2)
    
    # Single step integration
    strang_step_batch(q_symp, p_symp, params['dt'], 
                                 params['dq'], params['interpolation_order'])
    rk2_step_batch(q_rk2, p_rk2, params['dt'], 
                                 params['dq'], params['interpolation_order'])
    
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
    parser = argparse.ArgumentParser(description='Single-step interpolated pendulum simulation')
    parser.add_argument('--q0', type=float, default=np.pi, help='Initial q position')
    parser.add_argument('--p0', type=float, default=0.5, help='Initial p momentum')
    parser.add_argument('--rad', type=float, default=1.0, help='Initial radius')
    parser.add_argument('--M', type=int, default=1024, help='Number of trajectories')
    parser.add_argument('--dq', type=float, default=0.1, help='Interpolation grid size')
    parser.add_argument('--interpolation_order', type=int, default=2, help='Interpolation order')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size')
    args = parser.parse_args()
    
    # Convert args to dict
    params = vars(args)
    
    # Run simulation
    results = run_simulation(params)
    
    # Format output for easy regex parsing
    output = f"""
=== Simulation Parameters ===
q0: {params['q0']}
p0: {params['p0']}
radius: {params['rad']}
M: {params['M']}
dq: {params['dq']}
interpolation_order: {params['interpolation_order']}
dt: {params['dt']}

=== Results ===
Initial loop integral: {results['init_integral']}
Relative error (Symplectic): {results['rel_err_symp']}
Relative error (RK2): {results['rel_err_rk2']}
=== End of Simulation ===
"""
    print(output)

if __name__ == "__main__":
    main()
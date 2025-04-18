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
from pic_1d import PICSimulation, landau_IC, two_stream_IC
from loop_integrals_par import LoopIntegral

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
@njit(parallel=True)
def generate_loop(pos, vel, rad, boxsize, Np, Ntheta):
    
    loop_q = np.zeros((Np, Ntheta), dtype=np.float64)
    loop_p = np.zeros((Np, Ntheta), dtype=np.float64)

    for i in prange(Np):
        for j in prange(Ntheta):
            theta = 2 * np.pi * j / Ntheta  # Angle for the current point
            loop_q[i, j] = np.mod(pos[i] + rad * np.sin(theta), boxsize)  
            loop_p[i, j] = vel[i] + rad * np.cos(theta) 

    return loop_q, loop_p

def initialize_system(**params):
    """Initialize system with given parameters"""

    IC = params["IC"]
    rad = params["rad"]
    Ntheta  = params["Ntheta"]
    Np = params["Np"]
    boxsize = params["boxsize"]
    vb = params["vb"]
    vth = params["vth"]
    A = params["A"]
    k = params["k"]
    seed = params["seed"]

    if IC == "two_stream":
        pos, vel = two_stream_IC(Np, boxsize, vb, vth, A, seed=seed)
    elif IC == "landau":
        pos, vel = landau_IC(Np, boxsize, vth, A, k, seed=seed)
    else:
        raise ValueError("Invalid initial condition specified.")

    # Fill loop
    loop_q, loop_p = generate_loop(pos, vel, rad, boxsize, Np, Ntheta)

    return loop_q, loop_p

def run_simulation(params):

    # Set initial conditions
    loop_q0, loop_p0 = initialize_system(**params)

    # Create PICSimulation object
    sim_symp = PICSimulation(
        Np=params["Np"], Nx=params["Nx"], dt=params["dt"], 
        boxsize=params["boxsize"], n0=params["n0"], 
        initial_pos=None, initial_vel=None,
        order_time=params["order_time"], order_space=params["order_space"], 
        filter=params["filter"],
        symplectic=True, 
        verbose=params["verbose"], 
    )

    sim_non = PICSimulation(
        Np=params["Np"], Nx=params["Nx"], dt=params["dt"], 
        boxsize=params["boxsize"], n0=params["n0"], 
        initial_pos=None, initial_vel=None,
        order_time=params["order_time"], order_space=params["order_space"], 
        filter=params["filter"],
        symplectic=False, 
        verbose=params["verbose"], 
    )

    # Initialize loop integral storage arrays
    loop_q_symp = np.zeros((params["Np"], params["Ntheta"]), dtype=np.float64)
    loop_p_symp = np.zeros((params["Np"], params["Ntheta"]), dtype=np.float64)
    loop_q_non  = np.zeros((params["Np"], params["Ntheta"]), dtype=np.float64)
    loop_p_non  = np.zeros((params["Np"], params["Ntheta"]), dtype=np.float64)

    # Iterate over each point on the loop
    for i in range(params["Ntheta"]):
        
        # Set the simulation internal state for the current loop point
        sim_symp.initialize(loop_q0[:, i].squeeze(), loop_p0[:, i].squeeze())
        sim_non.initialize(loop_q0[:, i].squeeze(), loop_p0[:, i].squeeze())

        # Run the simulation
        sim_symp.step()
        loop_q_symp[:, i] = sim_symp.pos.squeeze()
        loop_p_symp[:, i] = sim_symp.vel.squeeze()
        sim_non.step()
        loop_q_non[:, i] = sim_non.pos.squeeze()
        loop_p_non[:, i] = sim_non.vel.squeeze()

    # Initialize LoopIntegral
    loop_integral = LoopIntegral(params["Ntheta"], period=params["boxsize"])

    # Compute loop integrals
    loop_int_init = loop_integral.compute(loop_q0, loop_p0)
    loop_int_symp = loop_integral.compute(loop_q_symp, loop_p_symp)
    loop_int_non  = loop_integral.compute(loop_q_non, loop_p_non)

    rel_err_symp = np.abs((loop_int_symp - loop_int_init) / (np.abs(loop_int_init)))
    rel_err_non  = np.abs((loop_int_non  - loop_int_init) / (np.abs(loop_int_init)))

    return {
        'rel_err_symp': rel_err_symp,
        'rel_err_non': rel_err_non,
        'init_integral': loop_int_init,
    }

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="1D Plasma PIC Loop Integral Diagnostic")
    # Loop parameters
    parser.add_argument('--Ntheta', type=int, default=2**12, help="Number of points on the loop")
    parser.add_argument('--rad', type=float, default=1.0, help="Loop radius")
    # Discretization parameters
    parser.add_argument('--Np', type=int, default=2**10, help="Number of particles")
    parser.add_argument('--Nx', type=int, default=2**7, help="Number of mesh cells")
    parser.add_argument('--boxsize', type=float, default=50.0, help="Periodic domain size")
    # Time-stepping parameters
    parser.add_argument('--dt', type=float, default=0.5, help="Time step")
    parser.add_argument('--symplectic', type=str2bool, default="True", help="Enable symplectic integration")
    # Display parameters
    parser.add_argument('--verbose', type=str2bool, default="False", help="Enable verbose output")
    # Initial data parameters
    parser.add_argument('--IC', type=str, default="landau", help="Initial condition (two_stream or landau)")
    parser.add_argument('--n0', type=float, default=1.0, help="Electron number density")
    parser.add_argument('--vb', type=float, default=3.0, help="Beam velocity (for two-stream IC)")
    parser.add_argument('--vth', type=float, default=1.0, help="Thermal velocity")
    parser.add_argument('--A', type=float, default=0.5, help="Perturbation amplitude")
    parser.add_argument('--k', type=int, default=1, help="Wavenumber of perturbation (for Landau IC)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    # Order parameters
    parser.add_argument('--order_time', type=int, default=2, help="Time-stepper order")
    parser.add_argument('--order_space', type=int, default=3, help="Spatial interpolation order (must be 2 or 3)")
    parser.add_argument('--filter', type=int, default=0, help="Number of times to apply filter")

    # Parse command-line arguments
    args = parser.parse_args()
    # Convert args to dict
    params = vars(args)

    # Run the simulation
    results = run_simulation(params)

    # Format output for easy regex parsing
    output = f"""
=== Simulation Parameters ===
Ntheta: {args.Ntheta}
Np: {args.Np}
Nx: {args.Nx}
boxsize: {args.boxsize}
dt: {args.dt}
symplectic: {args.symplectic}
verbose: {args.verbose}
IC: {args.IC}
n0: {args.n0}
vb: {args.vb}
vth: {args.vth}
A: {args.A}
k: {args.k}
seed: {args.seed}
order_time: {args.order_time}
order_space: {args.order_space}
filter: {args.filter}

=== Results ===
Initial loop integral: {results['init_integral']}
Relative error (Symplectic): {results['rel_err_symp']}
Relative error (RK2): {results['rel_err_non']}
=== End of Simulation ===
"""
    print(output)

if __name__ == "__main__":
    
    main()
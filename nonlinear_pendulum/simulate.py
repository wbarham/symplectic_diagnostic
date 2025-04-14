#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import from parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from nonlinear_pendulum import strang_step  # Your time-stepping function
from loop_integrals import LoopIntegral     # The class you provided

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Pendulum loop integral simulation')
    parser.add_argument('--q0', type=float, default=np.pi, help='Initial q position')
    parser.add_argument('--p0', type=float, default=0.5, help='Initial p momentum')
    parser.add_argument('--radius', type=float, default=1.0, help='Loop radius')
    parser.add_argument('--nsteps', type=int, default=1000, help='Number of time steps')
    parser.add_argument('--ntheta', type=int, default=2048, help='Number of theta points')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size')
    args = parser.parse_args()

    # Initialize loop
    theta = np.linspace(0, 2*np.pi, args.ntheta, endpoint=False)
    loop = np.array([
        args.q0 + args.radius * np.sin(theta),
        args.p0 + args.radius * np.cos(theta)
    ])

    # Initialize loop integral calculator
    loop_int_calc = LoopIntegral(args.ntheta)

    # Storage for results
    times = np.arange(args.nsteps) * args.dt
    loop_integrals = np.zeros(args.nsteps)

    # Compute initial loop integral
    initial_integral = loop_int_calc.compute(loop[0], loop[1])
    loop_integrals[0] = initial_integral

    # Time stepping loop
    for i in range(1, args.nsteps):
        # Perform time step
        strang_step(loop, args.dt)

        # Compute and store loop integral
        loop_integrals[i] = loop_int_calc.compute(loop[0], loop[1])

        # Progress reporting
        if i % 100 == 0:
            print(f"Step {i}/{args.nsteps}")

    # Calculate relative error
    rel_error = np.abs((loop_integrals - initial_integral) / initial_integral)

    # Set up plot parameters
    plt.rcParams['lines.markersize'] = 0.1
    plt.rcParams.update({'font.size': 14})

    # Create figure
    plt.figure()
    plt.semilogy(times, rel_error)
    plt.xlabel(r'Time')
    plt.ylabel(r'Relative error in loop integral')
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()

    # Generate descriptive filename
    params = {
        'q0': args.q0,
        'p0': args.p0,
        'rad': args.radius,
        'dt': args.dt,
        'ntheta': args.ntheta,
        'nsteps': args.nsteps
    }
    
    # Format values appropriately (replace π with pi for filename)
    q0_str = f"pi" if np.isclose(params['q0'], np.pi) else f"{params['q0']:.2f}"
    filename = (
        f"pendulum_q0_{q0_str}_p0_{params['p0']:.2f}_"
        f"rad_{params['rad']:.2f}_dt_{params['dt']:.3f}_"
        f"ntheta_{params['ntheta']}_nsteps_{params['nsteps']}"
    )
    
    # Replace any remaining dots with underscores for cleaner filenames
    filename = filename.replace('.', '_')
    
    plt.savefig(filename + ".png", bbox_inches='tight', dpi=300)
    print(f"Plot saved as: {filename}")
    plt.close()

if __name__ == "__main__":
    main()
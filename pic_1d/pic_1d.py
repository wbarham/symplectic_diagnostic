import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from   scipy.sparse.linalg import spsolve

from b_splines import (
    p2g_B1_par,
    p2g_B2_par,
    p2g_B3_par,
    g2p_dB1_par,
    g2p_dB2_par,
    g2p_dB3_par,
    construct_M_linear,
    construct_M_quadratic,
    construct_laplacian,
    binary_filter
)

class PICSimulation:
    """
    Class for 1D Plasma PIC Simulation.
    Handles only data and stepping.
    """

    def __init__(self, Np, Nx, dt, boxsize, n0, 
                 initial_pos, initial_vel, 
                 order_time=2, order_space=2, filter=0,
                 symplectic=True, verbose=False):
        
        # Simulation parameters
        self.Np = Np
        self.Nx = Nx
        self.boxsize = boxsize
        self.n0 = n0
        self.dt = dt

        # Particle weight
        self.w = self.n0 * self.Nx / self.Np

        # Choose time integration method
        if order_time == 1:
            self.step = self.lie_trotter_step if symplectic else self.euler_step
        elif order_time == 2:
            if symplectic:
                self.step = self.strang_step
            else:
                self.x0 = np.zeros(Np)
                self.v0 = np.zeros(Np)
                self.force0 = np.zeros(Np)
                self.step = self.rk2_step

        # Set up filtering if needed
        self.filter_num = filter
        if filter:
            self.binary_filter = binary_filter(Nx)

        # Grid operators
        self.Lap = construct_laplacian(boxsize, Nx, order_space)
        self.dx = boxsize / Nx
        self.order_space = order_space

        if order_space == 1:
            self.p2g = p2g_B1_par
            self.g2p = g2p_dB1_par
            self.M   = sp.eye(Nx)
        elif order_space == 2:
            self.p2g = p2g_B2_par
            self.g2p = g2p_dB2_par
            self.M   = construct_M_linear(Nx)
        elif order_space == 3:
            self.p2g = p2g_B3_par
            self.g2p = g2p_dB3_par
            self.M   = construct_M_quadratic(Nx)
        else:
            raise ValueError(f"Invalid interpolation order: {order_space}")
        
        # Initialize particle data
        self.initialize(initial_pos, initial_vel)

        if verbose:
            vmax = np.max(np.abs(self.vel))
            print(f"CFL condition: {vmax * self.dt / self.dx} (should be < 1)")

    def initialize(self, pos, vel):
        if pos is not None:
            self.pos = np.copy(pos)
            self.density = np.zeros(self.Nx)
            self.phi     = np.zeros(self.Nx)
            self.force   = np.zeros(self.Np)
            self.compute_charge_density()
            self.compute_force()
        if vel is not None:
            self.vel = np.copy(vel)
        self.t = 0

    def compute_charge_density(self):
        self.p2g(self.pos, self.density, self.dx)
        self.density *= self.w

        if self.filter_num > 0:
            for i in range(self.filter_num):
                self.n = self.binary_filter @ self.density

    def compute_force(self):
        self.compute_charge_density()
        self.phi = spsolve(self.Lap, self.density - self.n0, permc_spec="MMD_AT_PLUS_A")
        self.phi -= np.mean(self.phi)
        self.g2p(self.pos, self.phi, self.force, self.dx)
        self.force *= -1

    def euler_step(self):
        self.compute_force()
        self.pos = np.mod(self.pos + self.vel * self.dt, self.boxsize)
        self.vel += self.force * self.dt
        self.t += self.dt

    def lie_trotter_step(self):
        self.vel += self.force * self.dt
        self.pos = np.mod(self.pos + self.vel * self.dt, self.boxsize)
        self.compute_force()
        self.t += self.dt

    def rk2_step(self):
        self.compute_force()
        self.x0[:] = self.pos
        self.v0[:] = self.vel
        self.force0[:] = self.force

        self.pos = np.mod(self.pos + self.vel * self.dt, self.boxsize)
        self.vel += self.force * self.dt
        self.compute_force()

        self.pos = np.mod(self.x0 + 0.5 * self.dt * (self.v0 + self.vel), self.boxsize)
        self.vel = self.v0 + 0.5 * self.dt * (self.force0 + self.force)
        self.t += self.dt

    def strang_step(self):
        self.vel += 0.5 * self.force * self.dt
        self.pos = np.mod(self.pos + self.vel * self.dt, self.boxsize)
        self.compute_force()
        self.vel += 0.5 * self.force * self.dt
        self.t += self.dt

    def compute_energy(self):
        KE = 0.5 * self.w * np.sum(self.vel**2)
        EE = 0.5 * self.phi.T @ self.density
        return KE, EE
    
"""
Initialization functions for Landau and Two-stream simulations
"""

def two_stream_IC(N, boxsize, vb, vth, A, seed=None):
    """
    This function generates particle positions and velocities for 
    simulating the two-stream instability.

    Parameters:
        N (int): Number of particles.
        boxsize (float): Length of the simulation domain.
        vb (float): Bulk velocity of each stream.
        vth (float): Thermal velocity spread of particles around the bulk velocity.
        A (float): Amplitude of the sinusoidal perturbation applied to the 
                   velocity distribution.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pos (numpy.ndarray): Particle positions.
        vel (numpy.ndarray): Particle velocities.
    """

    # Set random seed if specified
    if seed is not None:
        np.random.seed(seed)

    # Initialize random data for two-stream instability test
    pos = np.random.rand(N) * boxsize
    vel = vth * np.random.randn(N) + vb
    Nh = int(N / 2)
    vel[Nh:] *= -1
    vel *= (1 + A * np.sin(2 * np.pi * pos / boxsize))

    return pos, vel

def landau_IC(N, boxsize, vth, A, k, seed=None):
    """
    This function generates particle positions and velocities for 
    simulating Landau damping.

    Parameters:
        N (int): Number of particles.
        boxsize (float): Size of the simulation domain.
        vth (float): Thermal velocity.
        A (float): Amplitude of density perturbation.
        k (int): Wavenumber of the perturbation.
        seed (int, optional): Random seed.

    Returns:
        pos (numpy.ndarray): Particle positions.
        vel (numpy.ndarray): Particle velocities.
    """

    # Set random seed if specified
    if seed is not None:
        np.random.seed(seed)

    # Initialize positions with rejection sampling
    pos = []
    while len(pos) < N:
        x = np.random.rand() * boxsize
        if np.random.rand() < (1 + A * np.cos(2 * np.pi * k * x / boxsize)) / (1 + A):
            pos.append(x)

    pos = np.array(pos).reshape(N)

    # Initialize velocities as Maxwellian
    vel = vth * np.random.randn(N)

    return pos, vel

"""
Helper function for plotting
"""

def phase_space_plot(pos, vel, boxsize, n, phi, ax, mode="landau"):
    """
    General plot function for phase space, charge density, and electric potential.

    Parameters:
    pos (ndarray): Array of particle positions.
    vel (ndarray): Array of particle velocities.
    boxsize (float): Size of the simulation box.
    n (ndarray): Charge density array.
    phi (ndarray): Electric potential array.
    ax (ndarray): Array of Matplotlib Axes objects for the plots.
    mode (str): "landau" or "two-stream" to determine phase space plotting style.
    """
    # Clear previous plots
    for axis in ax:
        axis.cla()

    # Phase space plot
    if mode == "two-stream":
        Nh = int(len(pos) / 2)
        ax[0].scatter(pos[0:Nh], vel[0:Nh], s=0.4, color='blue', alpha=0.5)
        ax[0].scatter(pos[Nh:], vel[Nh:], s=0.4, color='red', alpha=0.5)
        ax[0].axis([0, boxsize, -8, 8])
    else:  # default to "landau"
        ax[0].scatter(pos, vel, s=0.4, color='blue', alpha=0.5)
        ax[0].axis([0, boxsize, -10, 10])

    ax[0].set_title("Phase Space Distribution")
    ax[0].set_xlabel(r'Position ($x$)')
    ax[0].set_ylabel(r'Velocity ($v$)')

    # Charge density plot
    x = np.linspace(0, boxsize, len(n))
    ax[1].plot(x, n, color='green')
    ax[1].set_title("Charge Density")
    ax[1].set_xlabel(r'Position ($x$)')
    ax[1].set_ylabel(r'Density ($\rho$)')
    ax[1].grid(True)
    ax[1].axis([0, boxsize, 0, 2])

    # Electric potential plot
    phi_centered = phi - np.mean(phi)
    ax[2].plot(x, phi_centered, color='blue')
    ax[2].set_title("Electric Potential")
    ax[2].set_xlabel(r'Position ($x$)')
    ax[2].set_ylabel(r'Potential ($\phi$)')
    ax[2].grid(True)

    if mode == "two-stream":
        ax[2].axis([0, boxsize, -16, 16])
    else:
        ax[2].axis([0, boxsize, -20, 20])

    plt.tight_layout()

def plot_energy(time_array, energy_data, save_path=None):
    """
    Plot kinetic, electrostatic, and total energy evolution, along with the relative error.

    Parameters:
    - time_array (ndarray): Array of time points.
    - energy_data (dict): Dictionary containing "KE", "EE", "total_energy", and "total_charge" arrays.
    - save_path (str, optional): If provided, path to save the plot image.
    """
    KE = energy_data["KE"]
    EE = energy_data["EE"]
    total_energy = energy_data["total_energy"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=80)

    # Energy evolution
    ax1.plot(time_array, KE, label="Kinetic Energy")
    ax1.plot(time_array, EE, label="Electrostatic Energy")
    ax1.plot(time_array, total_energy, label="Total Energy")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Energy")
    ax1.legend()
    ax1.set_title("Energy Evolution")
    ax1.grid()

    # Relative energy error
    relative_error = np.abs((total_energy - total_energy[0]) / total_energy[0])
    ax2.plot(time_array, relative_error)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Relative Error of Total Energy")
    ax2.grid()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=240)
    plt.show()

"""
Helper function for running the simulation
"""

def run_simulation(sim: PICSimulation, nsteps=2, plot_mode=None, verbose=True):
    """
    Run the simulation using the provided PICSimulation object.

    Parameters:
    - sim (PICSimulation): The simulation object.
    - nsteps (int): Number of time steps to simulate.
    - plot_function (function): Optional function to plot the current state.
    - make_plots (bool): Whether to update plots during the run.
    - verbose (bool): Print step information.
    """

    if plot_mode: # Make plot if a mode is specified
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=80)

    energy_kinetic = []
    energy_electrostatic = []
    total_energy = []
    total_charge = []

    # Save initial energy and charge history
    KE, EE = sim.compute_energy()
    energy_kinetic.append(KE)
    energy_electrostatic.append(EE)
    total_energy.append(KE + EE)
    total_charge.append(np.sum(sim.density) - sim.n0)

    for i in range(nsteps):
        if verbose:
            print(f"Step {i+1} / {nsteps}")
        sim.step()

        # Save energy and charge history
        KE, EE = sim.compute_energy()
        energy_kinetic.append(KE)
        energy_electrostatic.append(EE)
        total_energy.append(KE + EE)
        total_charge.append(np.sum(sim.density) - sim.n0)

        # Plot if needed
        if plot_mode:
            phase_space_plot(sim.pos, sim.vel, sim.boxsize, 
                          sim.density, sim.phi, ax, mode=plot_mode)
            plt.pause(0.001)

    return {
        "KE": np.array(energy_kinetic),
        "EE": np.array(energy_electrostatic),
        "total_energy": np.array(total_energy),
        "total_charge": np.array(total_charge)
    }

# Example run
if __name__ == "__main__":
    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="1D Plasma PIC Simulation")
    # Discretization parameters
    parser.add_argument('--Np', type=int, default=10000, help="Number of particles")
    parser.add_argument('--Nx', type=int, default=200, help="Number of mesh cells")
    parser.add_argument('--boxsize', type=float, default=50.0, help="Periodic domain size")
    # Time-stepping parameters
    parser.add_argument('--nsteps', type=int, default=1000, help="Number of simulation steps")
    parser.add_argument('--dt', type=float, default=0.01, help="Time step")
    parser.add_argument('--symplectic', type=str2bool, default="True", help="Enable symplectic integration")
    # Display parameters
    parser.add_argument('--verbose', type=str2bool, default="False", help="Enable verbose output")
    # Initial data parameters
    parser.add_argument('--IC', type=str, default="two_stream", help="Initial condition (two_stream or landau)")
    parser.add_argument('--n0', type=float, default=1.0, help="Electron number density")
    parser.add_argument('--vb', type=float, default=3.0, help="Beam velocity (for two-stream IC)")
    parser.add_argument('--vth', type=float, default=1.0, help="Thermal velocity")
    parser.add_argument('--A', type=float, default=0.5, help="Perturbation amplitude")
    parser.add_argument('--k', type=int, default=1, help="Wavenumber of perturbation (for Landau IC)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    # Order parameters
    parser.add_argument('--order_time', type=int, default=2, help="Time-stepper order")
    parser.add_argument('--order_space', type=int, default=2, help="Spatial interpolation order (must be 2 or 3)")
    parser.add_argument('--filter', type=int, default=0, help="Number of times to apply filter")

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.IC == "two_stream":
        pos, vel = two_stream_IC(args.Np, args.boxsize, args.vb, args.vth, args.A, seed=args.seed)
    elif args.IC == "landau":
        pos, vel = landau_IC(args.Np, args.boxsize, args.vth, args.A, args.k, seed=args.seed)
    else:
        raise ValueError("Invalid initial condition specified.")

    # Ensure that pos and vel have been initialized
    if pos is None or vel is None:
        raise RuntimeError("Particle positions and velocities must be initialized before running the simulation.")

    # Initialize the simulation
    sim = PICSimulation(
        Np=args.Np, Nx=args.Nx, dt=args.dt, 
        boxsize=args.boxsize, n0=args.n0, 
        initial_pos=pos, initial_vel=vel, 
        order_time=args.order_time, order_space=args.order_space, 
        filter=args.filter,
        symplectic=args.symplectic, 
        verbose=args.verbose 
    )

    # Run the simulation and plot
    plot_mode = None
    # plot_mode = args.IC
    results = run_simulation(sim, nsteps=args.nsteps, plot_mode=plot_mode, 
                             verbose=True)
    time_array = np.arange(len(results["KE"])) * sim.dt
    save_path = None
    # save_path = f"energy_plot_order_{args.order_space}.png"
    plot_energy(time_array, results, 
                save_path=save_path)
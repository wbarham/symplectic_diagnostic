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
        alpha (float): Amplitude of density perturbation.
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

# Define the plotting function
def two_stream_plot(pos, vel, boxsize, n, phi, ax):
    """
    Plot the phase space distribution of particles in a two-stream instability 
    simulation, and also the charge density.

    Parameters:
    pos (ndarray): Array of particle positions.
    vel (ndarray): Array of particle velocities.
    boxsize (float): Size of the simulation box.
    Nh (int): Number of particles in one of the streams.
    n (ndarray): Charge density array.
    phi (ndarray): Potential array.
    E (ndarray): Electric field array.
    ax1 (matplotlib.axes.Axes): Matplotlib Axes object for the phase space plot.
    ax2 (matplotlib.axes.Axes): Matplotlib Axes object for the charge density plot.
    """
    # Clear the previous plots
    ax[0].cla()
    ax[1].cla()
    ax[2].cla()

    Nh = int(len(pos) / 2)

    # Phase space plot for two-stream particles
    ax[0].scatter(pos[0:Nh], vel[0:Nh], s=0.4, color='blue', alpha=0.5)
    ax[0].scatter(pos[Nh:], vel[Nh:], s=0.4, color='red', alpha=0.5)
    ax[0].set_title("Phase Space Distribution")
    ax[0].set_xlabel(r'Position ($x$)')
    ax[0].set_ylabel(r'Velocity ($v$)')
    #ax[0].axis([0, boxsize, np.min(vel), np.max(vel)])
    ax[0].axis([0, boxsize, -8, 8])

    # Charge density plot
    ax[1].plot(np.linspace(0, boxsize, len(n)), n, color='green')
    ax[1].set_title("Charge Density")
    ax[1].set_xlabel(r'Position ($x$)')
    ax[1].set_ylabel(r'Density ($\rho$)')
    ax[1].grid(True)
    #ax[1].axis([0, boxsize, np.min(n) - 1, np.max(n) + 1])
    ax[1].axis([0, boxsize, 0, 2])

    # Electric potential plot
    ax[2].plot(np.linspace(0, boxsize, len(phi)), phi - np.mean(phi), color='blue')
    ax[2].set_title("Electric Potential")
    ax[2].set_xlabel(r'Position ($x$)')
    ax[2].set_ylabel(r'Potential ($\phi$)')
    ax[2].grid(True)
    #ax[2].axis([0, boxsize, np.min(phi - np.mean(phi)) - 1, 
    #              np.max(phi - np.mean(phi)) + 1])
    ax[2].axis([0, boxsize, -16, 16])

    #plt.pause(0.001)

def landau_plot(pos, vel, boxsize, n, phi, ax):
    """
    Plot the phase space distribution of particles in a Landau damping simulation,
    and also the charge density, electric potential, and electric field.

    Parameters:
    pos (ndarray): Array of particle positions.
    vel (ndarray): Array of particle velocities.
    boxsize (float): Size of the simulation box.
    n (ndarray): Charge density array.
    phi (ndarray): Electric potential array.
    E (ndarray): Electric field array.
    ax (ndarray): Array of Matplotlib Axes objects for the plots.
    """
    # Clear the previous plots
    ax[0].cla()
    ax[1].cla()
    ax[2].cla()

    # Phase space plot for Landau damping particles
    ax[0].scatter(pos, vel, s=0.4, color='blue', alpha=0.5)
    ax[0].set_title("Phase Space Distribution")
    ax[0].set_xlabel(r'Position ($x$)')
    ax[0].set_ylabel(r'Velocity ($v$)')
    #ax[0].axis([0, boxsize, np.min(vel), np.max(vel)])
    ax[0].axis([0, boxsize, -10, 10])

    # Charge density plot
    ax[1].plot(np.linspace(0, boxsize, len(n)), n, color='green')
    ax[1].set_title("Charge Density")
    ax[1].set_xlabel(r'Position ($x$)')
    ax[1].set_ylabel(r'Density ($\rho$)')
    ax[1].grid(True)
    #ax[1].axis([0, boxsize, np.min(n) - 1, np.max(n) + 1])
    ax[1].axis([0, boxsize, 0, 2])

    # Electric potential plot
    ax[2].plot(np.linspace(0, boxsize, len(phi)), phi - np.mean(phi), color='blue')
    ax[2].set_title("Electric Potential")
    ax[2].set_xlabel(r'Position ($x$)')
    ax[2].set_ylabel(r'Potential ($\phi$)')
    ax[2].grid(True)
    #ax[2].axis([0, boxsize, np.min(phi - np.mean(phi)) - 1, 
    #              np.max(phi - np.mean(phi)) + 1])
    ax[2].axis([0, boxsize, -20, 20])

    #plt.pause(0.001)

class PICSimulation:
    """
    Class for 1D Plasma PIC Simulation with Cubic B-Spline Interpolation
    """

    def __init__(self, Np, Nx, nsteps, dt, boxsize, n0, 
                 initial_pos, initial_vel, 
                 order_time=2, order_space=2, filter=0,
                 make_plots=True, make_movie=False, symplectic=True, 
                 verbose=False, plot_function=None):
        
        # Store simulation parameters
        self.Np = Np
        self.Nx = Nx
        self.t = 0
        self.dt = dt
        self.nsteps = nsteps
        self.tEnd = dt * nsteps
        self.boxsize = boxsize
        self.n0 = n0
        self.make_plots = make_plots
        self.make_movie = make_movie
        self.plot_function = plot_function

        # Initialize position and velocity
        self.pos = initial_pos
        self.vel = initial_vel

        # Weight on each particle
        self.w = self.n0 * self.Nx / self.Np

        # Set up time integration
        if order_time == 1:
            if symplectic:
                self.step = self.lie_trotter_step
            else:
                self.step = self.euler_step
        elif order_time == 2:
            if symplectic:
                self.step = self.strang_step
            else:
                self.x0   = np.zeros(Np)
                self.v0   = np.zeros(Np)
                self.force0 = np.zeros(Np)
                self.step = self.rk2_step

        # Set up filtering
        self.filter_num = filter
        if filter:
            self.binary_filter = binary_filter(Nx)

        # Set grid operators
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

        # Initialize charge density, electric potential, and force
        self.density = np.zeros(Nx)
        self.phi     = np.zeros(Nx)
        self.force   = np.zeros(Np)
        self.compute_charge_density()
        self.compute_force()

        if verbose:
            self.vmax = np.max(np.abs(self.vel))
            print(f"CFL condition: {self.vmax * self.dt / self.dx} (must be < 1)")

    def compute_charge_density(self):
        self.p2g(self.pos, self.density, self.dx)
        self.density *= self.w

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
        # Store initial values
        self.compute_force()
        self.x0[:]   = self.pos
        self.v0[:]   = self.vel
        self.force0[:] = self.force

        # Euler step for intermediate result
        self.pos = np.mod(self.pos + self.vel * self.dt, self.boxsize)
        self.vel += self.force * self.dt
        self.compute_force()

        # Final update using RK2
        self.pos = np.mod(self.x0 + 0.5 * self.dt * (self.v0 + self.vel), self.boxsize)
        self.vel = self.v0 + 0.5 * self.dt * (self.force0 + self.force )
        self.t  += self.dt

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

    def run(self):
        Nt = int(np.ceil(self.tEnd / self.dt))

        if self.make_plots and self.plot_function:
            fig, ax = plt.subplots(1, 3, dpi=80)

        self.energy_kinetic = []
        self.energy_electrostatic = []
        self.total_energy = []
        self.total_charge = []

        for i in range(Nt):
            print(f"Step {i} of {Nt}")
            self.step()

            KE, EE = self.compute_energy()
            self.energy_kinetic.append(KE)
            self.energy_electrostatic.append(EE)
            self.total_energy.append(KE + EE)
            self.total_charge.append(np.sum(self.density))

            if self.make_movie and self.plot_function:
                self.plot_function(self.pos, self.vel, self.boxsize, self.density,
                                   self.phi, ax)
                plt.tight_layout()
                plt.xlabel('x')
                plt.ylabel('v')
                plt.savefig(f'plots_movie/pic{i}.png', dpi=240)

        if self.make_plots and self.plot_function:
            plt.tight_layout()
            self.plot_function(self.pos, self.vel, self.boxsize, self.density, 
                               self.phi, ax)
            plt.xlabel('x')
            plt.ylabel('v')
            plt.show()

        if self.make_plots:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=80)

            # Plot energies over time on the first subplot (ax1)
            ax1.plot(np.arange(Nt) * self.dt, self.energy_kinetic, label="Kinetic Energy")
            ax1.plot(np.arange(Nt) * self.dt, self.energy_electrostatic, 
                     label="Electrostatic Energy")
            ax1.plot(np.arange(Nt) * self.dt, self.total_energy, label="Total Energy")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Energy")
            ax1.legend()
            ax1.set_title("Energy Evolution")
            ax1.grid()

            # Plot relative error of total energy on the second subplot (ax2)
            ax2.plot(np.arange(Nt) * self.dt, abs((np.array(self.total_energy) \
                                                   - self.total_energy[0]) \
                                                  / self.total_energy[0]))
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Relative Error")
            ax2.set_title("Relative Error of Total Energy")
            ax2.grid()

            # Adjust layout to avoid overlap
            plt.tight_layout()

            # Save the combined plot as an image
            plt.savefig(f"energy_plots_order_{self.order_space}.png", dpi=240)

            # Show the plot
            plt.show()

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
    parser.add_argument('--Np', type=int, default=100000, help="Number of particles")
    parser.add_argument('--Nx', type=int, default=150, help="Number of mesh cells")
    parser.add_argument('--nsteps', type=int, default=1000, help="Number of simulation steps")
    parser.add_argument('--dt', type=float, default=0.05, help="Time step")
    parser.add_argument('--boxsize', type=float, default=50.0, help="Periodic domain size")
    parser.add_argument('--symplectic', type=str2bool, default="True", help="Enable symplectic integration")
    # Display parameters
    parser.add_argument('--make_plots', type=str2bool, default="True", help="Enable real-time plotting")
    parser.add_argument('--make_movie', type=str2bool, default="False", help="Enable movie making")
    parser.add_argument('--verbose', type=str2bool, default="False", help="Enable verbose output")
    # Initial data parameters
    parser.add_argument('--IC', type=str, default="two_stream", help="Initial condition (two_stream or landau)")
    parser.add_argument('--n0', type=float, default=1.0, help="Electron number density")
    parser.add_argument('--vb', type=float, default=3.0, help="Beam velocity (for two-stream IC)")
    parser.add_argument('--vth', type=float, default=1.0, help="Thermal velocity")
    parser.add_argument('--A', type=float, default=0.5, help="Perturbation amplitude")
    parser.add_argument('--k', type=int, default=1, help="Wavenumber of perturbation (for Landau IC)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--order_time', type=int, default=2, help="Time-stepper order")
    parser.add_argument('--order_space', type=int, default=2, help="Spatial interpolation order (must be 2 or 3)")
    parser.add_argument('--filter', type=int, default=0, help="Number of times to apply filter")

    args = parser.parse_args()

    pos, vel = None, None
    plot_function = None

    if args.IC == "two_stream":
        pos, vel = two_stream_IC(args.Np, args.boxsize, args.vb, args.vth, args.A, seed=args.seed)
        plot_function = two_stream_plot
    elif args.IC == "landau":
        pos, vel = landau_IC(args.Np, args.boxsize, args.vth, args.A, args.k, seed=args.seed)
        plot_function = landau_plot
    else:
        raise ValueError("Invalid initial condition specified.")

    # Ensure that pos and vel have been initialized
    if pos is None or vel is None:
        raise RuntimeError("Particle positions and velocities must be initialized before running the simulation.")

    # Initialize the simulation
    sim = PICSimulation(
        Np=args.Np, Nx=args.Nx, nsteps=args.nsteps, dt=args.dt, boxsize=args.boxsize, 
        n0=args.n0, initial_pos=pos, initial_vel=vel, 
        order_time=args.order_time, order_space=args.order_space, filter=args.filter,
        make_plots=bool(args.make_plots), make_movie=bool(args.make_movie),
        symplectic=bool(args.symplectic), 
        verbose=bool(args.verbose), plot_function=plot_function
    )

    sim.run()

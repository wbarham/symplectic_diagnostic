import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.njit()
def strang_step(z, dt):
    q = z[0]
    p = z[1]
    
    q += dt/2 * p
    p += dt * np.sin(q)
    q += dt/2 * p

@nb.njit()
def hamiltonian(z):
    q = z[0]
    p = z[1]
    return 0.5 * p**2 + np.cos(q)

def generate_plot_data(dt, nsteps, nsoln):
    data = np.zeros((nsteps, 2, nsoln))
    z = np.array([ [np.pi] * nsoln, [8 * (i/nsoln - 1/2) for i in range(nsoln)] ])
    data[0, :, :] = z[:, :]

    for i in range(1, nsteps):
        strang_step(z, dt)
        data[i, :, :] = z[:, :]
    return data

if __name__ == "__main__":
    # Set simulation parameters
    tend = 10.0
    dt = 0.1
    nsteps = int(tend/dt)
    t = np.linspace(0, tend, nsteps)

    # Loop parameters
    q0 = np.pi
    p0 = 1.0
    rad = 1.0

    # Generate phase space data
    data = generate_plot_data(dt, 5001, 101)

    # Set up loop
    ntheta = 2**16
    theta = np.linspace(0, 1, ntheta)
    loop = np.array([[q0 + rad * np.sin(2 * np.pi * theta[i]) for i in range(ntheta)], 
                    [p0 + rad * np.cos(2 * np.pi * theta[i]) for i in range(ntheta)]])

    # Arrays to store energy and loop samples
    energies = np.zeros(nsteps)
    sample_loops = []
    sample_times = []
    nsamples = nsteps // 5

    # Compute initial energy
    energies[0] = hamiltonian(loop[:,0])  # Energy of first point

    for i in range(nsteps):
        if i % nsamples == 0:
            sample_loops.append(loop.copy())
            sample_times.append(t[i])

        if i % 50 == 0:
            print(f"Stepping solution {i} of {nsteps}")

        # Step loop and compute energy
        strang_step(loop, dt)
        energies[i] = hamiltonian(loop[:,0])  # Track energy of first point

    sample_loops.append(loop.copy())
    sample_times.append(t[-1])

    # Create plots
    # Set up plot parameters
    plt.rcParams['lines.markersize'] = 0.1
    # Set font size
    plt.rcParams.update({'font.size': 14})

    # Phase Space Plot
    plt.figure()
    ax1 = plt.gca()
    ax1.set_xlabel(r'$q$ (mod $2\pi$)')
    ax1.set_ylabel(r'$p$')
    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_ylim(-3, 3)
    ax1.plot(data[::6, 0, :] % (2 * np.pi), data[::6, 1, :], '.k', markersize=0.1)

    # Plot loops with color progression
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(sample_loops)))
    for i, (loop, tsample, color) in enumerate(zip(sample_loops, sample_times, colors)):
        ax1.plot(loop[0] % (2 * np.pi), loop[1], '-', color=color, label=f't={tsample:.0f}')
    ax1.legend(loc='best')
    plt.tight_layout()
    plt.savefig('nl_pendulum_phase_space.png', bbox_inches='tight')
    plt.show()

    # Energy Plot
    plt.figure()
    ax2 = plt.gca()
    ax2.semilogy(t, np.abs((energies - energies[0])/energies[0]))
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy Relative Error')
    ax2.grid(True)
    plt.tight_layout()
    #plt.savefig('nl_pendulum_energy.png', bbox_inches='tight')
    plt.show()
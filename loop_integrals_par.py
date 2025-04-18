import numpy as np
import pyfftw
pyfftw.interfaces.cache.enable()

# Use pyfftw's aligned arrays and fft/ifft
from pyfftw.interfaces.numpy_fft import fft, ifft
import os
import cProfile
import pstats

class LoopIntegral:
    def __init__(self, N, period=None, num_threads=None):
        """
        Initialize the loop integral calculator with optimizations using pyFFTW's parallelism.

        Parameters:
        - N (int): Number of grid points in the theta direction.
        - period (float, optional): Periodicity of the q values. If None, no unwrapping is applied.
        - num_threads (int, optional): Number of threads to use for pyFFTW. If None, uses all available CPUs.

        """
        self.M = N
        self.period = period
        self.dtheta = 1 / N
        self.num_threads = num_threads or os.cpu_count()

        # Define wave numbers in Fourier space (precomputed and aligned)
        k = np.fft.fftfreq(N, d=self.dtheta)
        k_imag_unaligned = 1j * k.reshape(1, N)
        self.k_imag = pyfftw.empty_aligned(k_imag_unaligned.shape, dtype=k_imag_unaligned.dtype)
        self.k_imag[:] = k_imag_unaligned

        # Initialize aligned arrays for q and p (placeholder shape and dtype)
        self.q_aligned = pyfftw.empty_aligned((1, N), dtype=np.complex128)
        self.p_aligned = pyfftw.empty_aligned((1, N), dtype=np.float64)

    def compute(self, q, p):
        """
        Compute the loop integral ∑ p * dq/dtheta scaled by dtheta using pyFFTW parallelism.

        Parameters:
        - p (np.ndarray): Array of shape (D, N) where D is the dimension of the system
        - q (np.ndarray): Array of shape (D, N) where D is the dimension of the system

        Returns:
        - loop_integral (float): Value of loop integral

        """
        # Unwrap q prior to differentiating if it is periodic
        if self.period is not None:
            q = np.unwrap(q, axis=-1, period=self.period)

        # Resize aligned arrays if the input shape changes
        if self.q_aligned.shape != q.shape:
            self.q_aligned = pyfftw.empty_aligned(q.shape, dtype=q.dtype)
        if self.p_aligned.shape != p.shape:
            self.p_aligned = pyfftw.empty_aligned(p.shape, dtype=p.dtype)

        self.q_aligned[:] = q
        self.p_aligned[:] = p

        # Perform FFT with specified number of threads
        q_hat = fft(self.q_aligned, axis=-1, threads=self.num_threads)
        q_prime_hat = self.k_imag * q_hat

        # Perform inverse FFT with specified number of threads
        q_prime = ifft(q_prime_hat, axis=-1, threads=self.num_threads).real

        # Compute loop integral using optimized summation
        loop_integral = np.sum(self.p_aligned * q_prime) * self.dtheta
        return loop_integral

# Main test
if __name__ == "__main__":
    # Initialize parameters
    N = 4096  # Number of grid points in the theta direction
    D = 1000    # Number of dimensions (number of loops)

    # Random initialization of q and p
    q0 = np.random.random(D)  # q with shape (D,)
    p0 = np.random.random(D)  # p with shape (D,)

    # Vary q and p periodically to form D distinct loops
    q = np.zeros((D, N))
    p = np.zeros((D, N))
    for i in range(D):
        for j in range(N):
            q[i, j] = q0[i] + np.sin(2 * np.pi * j / N)
            p[i, j] = p0[i] + np.cos(2 * np.pi * j / N)

    # Create LoopIntegral instance
    loop_integral_calculator = LoopIntegral(N)

    # Compute the loop integral and profile it
    with cProfile.Profile() as pr:
        computed_loop_integral = loop_integral_calculator.compute(q, p)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(20) # Show the top 20 time-consuming functions

    # Expected value: sum of D integrals of cos^2(theta) over [0, 2pi] which is 0.5 per loop
    expected_value = 0.5 * D

    # Calculate the relative error
    relative_error = np.abs(computed_loop_integral - expected_value) / expected_value

    # Print results
    print(f"\nComputed Loop Integral: {computed_loop_integral}")
    print(f"Expected Loop Integral: {expected_value}")
    print(f"Relative Error: {relative_error}")
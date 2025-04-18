import numpy as np
import pyfftw
pyfftw.interfaces.cache.enable()

# Replace np.fft functions with pyfftw for performance:
from pyfftw.interfaces.numpy_fft import fft, ifft

class LoopIntegral:
    def __init__(self, N, period=None):
        """
        Initialize the loop integral calculator with optimizations.

        Parameters:
        - N (int): Number of grid points in the theta direction.
        - period (float, optional): Periodicity of the q values. If None, no unwrapping is applied.

        """
        self.M = N
        self.period = period
        self.dtheta = 1 / N

        # Define wave numbers in Fourier space (precomputed and optimized)
        k = np.fft.fftfreq(N, d=self.dtheta) # Proper wave numbers
        self.k_imag = 1j * k.reshape(1, N)

    def compute(self, q, p):
        """
        Compute the loop integral ∑ p * dq/dtheta scaled by dtheta.

        Parameters:
        - p (np.ndarray): Array of shape (D, N) where D is the dimension of the system
        - q (np.ndarray): Array of shape (D, N) where D is the dimension of the system

        Returns:
        - loop_integral (float): Value of loop integral

        """
        # Unwrap q prior to differentiating if it is periodic
        if self.period is not None:
            q = np.unwrap(q, axis=-1, period=self.period)

        # Compute spectral derivative 
        q_hat = fft(q, axis=-1)  
        q_prime_hat = self.k_imag * q_hat  
        q_prime = ifft(q_prime_hat, axis=-1).real  

        # Compute loop integral using optimized summation
        loop_integral = np.sum(p * q_prime) * self.dtheta
        return loop_integral
    
# Main test
if __name__ == "__main__":
    # Initialize parameters
    N = 4096  # Number of grid points in the theta direction
    D = 100    # Number of dimensions (number of loops)

    # Random initialization of q and p
    q0 = np.random.random(D)  # q with shape (D, N)
    p0 = np.random.random(D)  # p with shape (D, N)

    # Vary q and p periodically to form D distinct loops
    q = np.zeros((D, N))
    p = np.zeros((D, N))
    for i in range(D):
        for j in range(N):
            q[i, j] = q0[i] + np.sin(2 * np.pi * j / N) 
            p[i, j] = p0[i] + np.cos(2 * np.pi * j / N) 

    # Create LoopIntegral instance
    loop_integral_calculator = LoopIntegral(N)

    # Compute the loop integral
    computed_loop_integral = loop_integral_calculator.compute(q, p)

    # Expected value: sum of D integrals of cos^2(theta) over [0, 2pi] which is 0.5 per loop
    expected_value = 0.5 * D

    # Calculate the relative error
    relative_error = np.abs(computed_loop_integral - expected_value) / expected_value

    # Print results
    print(f"Computed Loop Integral: {computed_loop_integral}")
    print(f"Expected Loop Integral: {expected_value}")
    print(f"Relative Error: {relative_error}")
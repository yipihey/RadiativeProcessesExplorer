import numpy as np
from scipy.special import wofz

# Physical constants
R_H = 2.18e-11  # Rydberg constant [erg]
h = 6.626e-27   # Planck's constant [erg·s]
alpha = 1 / 137 # Fine-structure constant
a0 = 5.29e-9    # Bohr radius [cm]
c = 2.998e10  # Speed of light [cm/s]
k_B = 1.38e-16  # Boltzmann constant [erg/K]

def voigt_profile(nu, nu_0, A21, T, m, v_shift):
    """Compute Voigt profile for absorption line modeling."""
    delta_nu_D = (nu_0 / c) * np.sqrt(2 * k_B * T / m)
    gamma_L = A21 / (4 * np.pi)
    x = (nu - nu_0 - v_shift * nu_0 / c) / delta_nu_D
    a = gamma_L / delta_nu_D
    return np.real(wofz(x + 1j * a)) / (delta_nu_D * np.sqrt(np.pi))

def calculate_absorption(nu_0, lambda_delta, A21, T, m, v_shift, N):
    """Calculate absorption parameters."""
    nu = np.linspace(nu_0*(1 - lambda_delta), nu_0*(1 + lambda_delta), 500)
    phi_V = voigt_profile(nu, nu_0, A21, T, m, v_shift*1e5)
    
    B12 = (c**2 / (2 * h * nu_0**3)) * A21  # g2/g1 = 1 assumed
    sigma_nu = (h * nu_0 / (4 * np.pi)) * B12 * phi_V
    return nu, sigma_nu, np.exp(-N * sigma_nu)

def photoionization_cross_section_cgs(nu, Z):
    """
    Calculate the photoionization cross-section for a hydrogenic atom in CGS units.
    Parameters:
        nu (float or np.ndarray): Frequency of the incident photon [Hz].
        Z (int): Atomic charge number (e.g., 1 for hydrogen, 2 for He+).
    Returns:
        sigma (float or np.ndarray): Photoionization cross-section [cm²].
    """
            
    # Threshold frequency
    nu_0 = (Z**2 * R_H) / h    
    # Cross-section at threshold
    sigma_0 = (64 * np.pi * alpha * a0**2) / (3 * np.sqrt(3) * Z**2)
    # Dimensionless parameter
    tau = np.sqrt(nu / nu_0 - 1)
    # Cross-section formula
    sigma = sigma_0 * (nu_0 / nu)**3 * np.exp(-4 * np.arctan(tau) / tau) / (1 - np.exp(-2 * np.pi / tau))

    if isinstance(nu, np.ndarray):
        sigma[nu<nu_0] = 0
    else:
        if nu<nu_0:
            sigma = 0.0

    
    return sigma


import numpy as np
from scipy.special import wofz
from scipy.integrate import trapezoid


# Physical constants
R_H =  2.1798723611030e-11  # Rydberg constant [erg]
h = 6.626e-27   # Planck's constant [erg·s]
alpha = 1 / 137.035999177 # Fine-structure constant
a0 = 5.291772105e-9    # Bohr radius [cm]
c = 2.99792458e10  # Speed of light [cm/s]
k_B = 1.380649e-16  # Boltzmann constant [erg/K]
m_e = 9.10938188e-28

def voigt_profile(nu, nu_0, A21, T, m, v_shift):
    """Compute Voigt profile for absorption line modeling."""
    delta_nu_D = (nu_0 / c) * np.sqrt(2 * k_B * T / m)
    gamma_L = A21 / (4 * np.pi)
    x = (nu - nu_0 - v_shift * nu_0 / c) / delta_nu_D
    a = gamma_L / delta_nu_D
    return np.real(wofz(x + 1j * a)) / (delta_nu_D * np.sqrt(np.pi))

def calculate_absorption(nu_0, lambda_delta, A21, T, m, v_shift, N):
    """Calculate absorption parameters."""
    nu = np.linspace(nu_0*(1 - lambda_delta), nu_0*(1 + lambda_delta), 1500)
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
    sigma = sigma_0 * (nu_0 / nu)**3 * np.exp( -4 * np.arctan(tau) / tau) / (1 - np.exp(-2 * np.pi / tau))

    if isinstance(nu, np.ndarray):
        sigma[nu<nu_0] = 0
    else:
        if nu<nu_0:
            sigma = 0.0

    
    return sigma

def bb(nu, T):
    return 2*h*nu**3/c**2 / (np.exp(h*nu/k_B/T) - 1)

def stellar_bb_spectrum(nu,T,R):
    return 4*np.pi * R**2 * bb(nu,T)

from scipy.integrate import quad

def number_ionizing_photons(thresh, T, R):
    def integrand(nu):
        return 1/(h*nu) * stellar_bb_spectrum(nu,T,R)
    return quad(integrand, thresh, 1000*thresh)


# Add these new functions to your existing physics.py

def recombination_coefficient(T_e, n):
    """Hydrogenic recombination coefficient to level n (cm³/s)"""
    return 2.07e-11 * n**(-3.) * (T_e/1e4)**-0.75

def einstein_A_hydrogen(n):
    """Einstein A coefficient for hydrogen transitions (s⁻¹)"""
    return 4.7e9 * n**(-3.)

def two_photon_profile(nu, nu_Lyα):
    """Two-photon decay spectral profile (Hz⁻¹)"""
    y = nu / nu_Lyα
    mask = (y > 0) & (y < 1)
    phi = np.zeros_like(y)
    phi[mask] = (6/(5*nu_Lyα)) * y[mask] * (1 - y[mask])
    return phi

def recombination_spectrum(T_e, n_max, n_e, nu_grid, 
                          case_b=True, two_photon=True):
    """
    Compute hydrogen recombination spectrum
    
    Parameters:
    T_e: Electron temperature [K]
    n_max: Maximum principal quantum number
    n_e: Electron density [cm⁻³]
    nu_grid: Frequency grid [Hz]
    
    Returns:
    spectrum: Emission spectrum [erg s⁻¹ Hz⁻¹]
    wavelength: Corresponding wavelength grid [Å]
    """
    # Use existing constants from physics.py
    global R_H, h, c, k_B
    
    # Level populations
    n_levels = np.arange(2, n_max+1)
    alpha_n = np.array([recombination_coefficient(T_e, n) for n in n_levels])
    N_n = alpha_n / alpha_n.sum()
    
    spectrum = np.zeros_like(nu_grid)
    nu_Lyα = R_H/h * (1 - 1/2**2)  # 3/4 R_H/h
    
    # Line emission using existing voigt_profile
    for i, n in enumerate(n_levels):
        for n_prime in range(1, n):
            if case_b and n_prime == 1:
                continue  # Skip Lyman series for Case B
                
            # Transition parameters
            nu_ij = R_H/h * (1/n_prime**2 - 1/n**2)
            A = einstein_A_hydrogen(n)
            
            # Use existing voigt_profile with electron mass
            profile = voigt_profile(
                nu=nu_grid,
                nu_0=nu_ij,
                A21=A,
                T=T_e,
                m=9.109e-28,  # Electron mass in grams
                v_shift=0
            )
            
            spectrum += h*nu_ij * A * N_n[i] * n_e * profile

    # Add two-photon continuum using existing constants
    if two_photon and 2 in n_levels:
        A_2γ = 8.22  # Two-photon decay rate [s⁻¹]
        N_2s = N_n[0] * 0.1  # Assume 10% population in 2s
        spectrum += h*nu_Lyα * A_2γ * N_2s * n_e * two_photon_profile(nu_grid, nu_Lyα)

    return spectrum, c/nu_grid * 1e8  # spectrum, wavelength in Å


def compton_cross_section(E_eV):
    """Calculate Compton scattering cross-section using Klein-Nishina formula"""
    sigma_T = 6.652458732e-25  # Thomson cross-section (cm²)
    m_e_c2_eV = 511e3  # Electron rest mass in eV
    
    x = np.asarray(E_eV) / m_e_c2_eV
    
    
    # Corrected Klein-Nishina formula implementation
    term1 = (1 + x)/x**3
    term2a = (2*x*(1 + x))/(1 + 2*x)
    term2b = np.log(1 + 2*x)  # Removed erroneous /x
    
    term3 = np.log(1 + 2*x)/(2*x)
    term4 = (1 + 3*x)/((1 + 2*x)**2)
    
    ratio = 0.75 * (term1*(term2a - term2b) + term3 - term4)

    ratio = np.where(x < 1e-3, 1, ratio)
    
    return sigma_T * ratio


from scipy.integrate import trapezoid
import numpy as np

def calculate_ic_spectrum(nu_input, I_input, gamma, n_e, T_bb):
    """Stable inverse Compton calculation using trapezoidal integration"""
    # Constants
    mec2 = m_e * c**2  # erg
    h_nu_input = h * nu_input  # erg
    
    # Create output frequency grid (log spaced)
    nu_output = np.logspace(
        np.log10(nu_input.min()) - 2,
        np.log10(nu_input.max()) + np.log10(gamma.max()**2) + 2,
        500
    )
    
    # Initialize output spectrum
    I_output = np.zeros_like(nu_output)
    
    # Create integration grids in log space
    log_gamma = np.log(gamma)
    log_nu_input = np.log(nu_input)
    
    # Precompute boost factors and cross sections
    gamma_2d, nu_2d = np.meshgrid(gamma, nu_input, indexing='ij')
    with np.errstate(divide='ignore', invalid='ignore'):
        boost = (4/3) * gamma_2d**2 * (h_nu_input / mec2)
        valid = (boost > 1e-6) & (boost < 1e6)
        nu_scat = nu_2d * boost
        x = h_nu_input / (gamma_2d * mec2)
    
    sigma = compton_cross_section(x * 511e3)  # x in eV
    integrand = np.where(valid, n_e[:, None] * sigma * I_input * boost, 0.0)
    
    # Main integration loop
    for i, nu_out in enumerate(nu_output):
        # Frequency bin with 5% tolerance
        mask = (nu_scat >= 0.95*nu_out) & (nu_scat <= 1.05*nu_out)
        
        if not np.any(mask):
            continue
            
        # Apply mask and integrate
        masked = np.where(mask, integrand/nu_scat, 0)
        
        # Integrate over gamma dimension first
        int_gamma = trapezoid(masked, x=log_gamma, axis=0)
        
        # Then integrate over input frequencies
        I_output[i] = trapezoid(int_gamma, x=log_nu_input)

    # Apply smoothing filter
    I_output = np.convolve(I_output, np.ones(3)/3, mode='same')
    
    return nu_output, I_output
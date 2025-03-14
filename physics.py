import numpy as np
from scipy.special import wofz
from scipy.integrate import trapezoid

import pandas as pd

# Physical constants
R_H =  2.1798723611030e-11  # Rydberg constant [erg]
h = 6.626e-27   # Planck's constant [erg·s]
alpha = 1 / 137.035999177 # Fine-structure constant
a0 = 5.291772105e-9    # Bohr radius [cm]
c = 2.99792458e10  # Speed of light [cm/s]
k_B = 1.380649e-16  # Boltzmann constant [erg/K]
m_e = 9.10938188e-28
m_p = 1.6726e-24


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


def hydrogen_energy(n):
    """Energy of hydrogen level n in erg"""
    return -R_H*h*c/n**2

def hydrogen_radiative_rates(n_upper, n_lower):
    """Einstein A coefficient for transition n_upper -> n_lower"""
    if n_upper <= n_lower: return 0
    return 6.67e8 * (n_upper - n_lower)**2 / (n_upper**2 * n_lower**3)

def hydrogen_oscillator_strength(n_upper, n_lower):
    """Quantum mechanical oscillator strength"""
    if n_upper <= n_lower: return 0
    delta_n = n_upper - n_lower
    return (32/3*np.sqrt(3)/(3*np.pi)) * (n_lower**2 * n_upper**2)/(n_upper**2 - n_lower**2)**3

def hydrogen_collisional_rate(n_lower, n_upper, T, ne):
    """Accurate collisional rate coefficient (cm³/s)"""
    delta_E = hydrogen_energy(n_upper) - hydrogen_energy(n_lower)
    return 8.63e-6 * ne * hydrogen_oscillator_strength(n_upper, n_lower) / \
           (T**0.5 * (n_lower**2)) * np.exp(-delta_E/(k_B*T))

def planck_function(nu, T):
    """Radiation field intensity (erg/cm²/s/Hz/sr)"""
    return (2*h*nu**3/c**2) / (np.exp(h*nu/(k_B*T)) - 1)    

def photoionization_cross_section_hydrogen(E_photon, n):
    """Hydrogenic photoionization cross-section (cm²)"""
    E_ion = -hydrogen_energy(n)
    x = E_photon/E_ion
    return 6.3e-18 * n**-5 * (x >= 1) * (x**-3.5)

########################

# Li and Draine 2001 (PAH Absorption Code)


filename = "./drude_params.csv"

class Blackbody:
    def __init__(self, T):
        self.T = T

    def spectrum(self, lam):
        """
        - lam: wavelength in microns

        Returns:
        - intensity at the given wavelength
        """

        lam *= 1e-6  # convert microns to meters

        h = 6.62607015e-34  # Planck constant in J*s
        c = 3.0e8  # speed of light in m/s
        k = 1.380649e-23  # Boltzmann constant in J/K

        return (2.0 * h * c**2) / (lam**5 * (np.exp(h * c / (lam * k * self.T)) - 1.0))

class PAHSpectrum:

    def __init__(self, NC, HC_ratio, ionized=False):
        self.H_C = HC_ratio
        self.NC = NC
        self.table = pd.read_csv(filename)
        self.ionized = ionized

        self.lam_min = 1/17.25
        self.lam_max = 1/3.3


    def check_lam_bounds(self, lam):
        

        if lam < self.lam_min:
            raise ValueError("Wavelength is out of bounds for model")

    def get_sigma(self, j, ionized=False):

        
        sigma = 0.0

        if ionized:
            sigma = self.table["sigma_ion"][j]
        else:
            sigma = self.table["sigma"][j]
        
        # Modify cross sections that depend linearly on
        # the H/C ratio.
        if j in (3, 6, 7, 8, 9):
            return sigma * self.H_C * 1e-20
        
        return sigma * 1e-20

    def S(self, lam, j):
        """
        - lam: wavelength in microns
        - j: feature number

        Returns:
        - absorption cross-section per carbon atom in cm2 / C
        """

        # self.check_lam_bounds(lam)

        lam /= 10000 # convert micron to cm

        gam_j = self.table["gamma_j"][j]
        lam_j = self.table["lam_j"][j] / 10000 # convert micron to cm
        sigma = self.get_sigma(j, self.ionized)

        return (2 / np.pi) * (gam_j * lam_j * sigma) / ((lam / lam_j - lam_j / lam)**2 + gam_j**2)

    def absorption_cross_section(self, lam):
        """
        absorption cross section per carbon atom
        """
        self.check_lam_bounds(lam)
        x = (1./lam)

        if x > 17.25:
            raise ValueError("x is out of bounds")
        elif (15. < x) and (x < 17.25):
            return (126. - 6.4943 * x) * 1e-18
        elif (10. < x) and (x < 15.):
            return self.S(lam, 1) + (-3.0 + 1.35*x) * 1e-18
        elif (7.7 < x) and (x < 10.):
            return (66.302 - 24.367*x + 2.950*x**2 - 0.1057*x**3) * 1e-18
        elif (5.9 < x) and (x < 7.7):
            return self.S(lam, 2) + (1.8687 + 0.1905*x + 0.4175 * (x - 5.9)**2 + 0.04370 * (x - 5.9)**3) * 1e-18
        elif (3.3 < x) and (x < 5.9):
            return self.S(lam, 2)  + (1.8687 + 0.1905*x) * 1e-18 
        elif x < 3.3:

            S_3_14 = np.sum([self.S(lam, i) for i in range(2, 14)])

            return 34.58**(-18-(3.431 / x)) * self.cutoff(lam) + S_3_14
    
    # def sum_S(self, lam):
    #     return np.sum([self.S(lam, j) for j in range(0, 5)])

    def cutoff(self, lam):
        # Desert+1990

        self.check_lam_bounds(lam)

        M = 0.

        if self.NC >= 40:
            M = 0.4 * self.NC
        elif self.NC < 40:
            M = 0.3 * self.NC
        
        lam_c = 0.

        if self.ionized:
            # PAH Cations
            lam_c = 1 / (2.282 * M**(-0.5) + 0.889)
        else:
            lam_c = 1 / (3.804 * M**(-0.5) + 1.052)
            

        y = lam_c / lam

        return (1/np.pi) * np.arctan((10 * (y-1))**3 / y) + (1/2)

    def optical_depth(self, lam):
        self.NC * self.absorption_cross_section(lam)

    def attenuate(self, intensity, lam):
        return intensity * np.exp(self.optical_depth(lam))
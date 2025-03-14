import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.constants import e

# Physical constants (in CGS)
R_H = 2.1798723611030e-11  # Rydberg constant [erg]
h = 6.626e-27   # Planck's constant [erg·s]
c = 2.99792458e10  # Speed of light [cm/s]
k_B = 1.380649e-16  # Boltzmann constant [erg/K]
m_p = 1.6726e-24  # Proton mass [g]
R_sun = 6.957e10  # Solar radius [cm]

def hydrogen_energy(n):
    """Energy of hydrogen level n in erg"""
    return -R_H/n**2

def planck_function(nu, T):
    """Radiation field intensity (erg/cm²/s/Hz/sr)"""
    return (2*h*nu**3/c**2) / (np.exp(h*nu/(k_B*T)) - 1)

def voigt_profile(nu, nu_0, A21, T, m, v_shift):
    """Compute Voigt profile for absorption line modeling."""
    from scipy.special import wofz
    delta_nu_D = (nu_0 / c) * np.sqrt(2 * k_B * T / m)
    gamma_L = A21 / (4 * np.pi)
    x = (nu - nu_0 - v_shift * nu_0 / c) / delta_nu_D
    a = gamma_L / delta_nu_D
    return np.real(wofz(x + 1j * a)) / (delta_nu_D * np.sqrt(np.pi))

def calculate_einstein_A(n_upper, n_lower):
    """Calculate Einstein A coefficient for hydrogen transition"""
    if n_upper <= n_lower:
        return 0.0
    
    # Energy difference
    delta_E = abs(hydrogen_energy(n_upper) - hydrogen_energy(n_lower))
    nu = delta_E/h
    
    # Calculate reduced matrix element (approximate)
    matrix_element = 2.0 * n_lower**2 * n_upper**2 / (n_upper**2 - n_lower**2)**3
    
    # Einstein A coefficient
    return 64 * np.pi**4 * e**2 / (3 * h * c**3) * nu**3 * matrix_element

def calculate_level_populations(n_levels, T_gas, n_H, T_rad, R_star_pc):
    """Calculate non-LTE level populations for hydrogen atom"""
    
    # Convert parsec to cm
    R_star_cm = R_star_pc * 3.086e18
    
    # Energy levels and statistical weights
    energies = np.array([hydrogen_energy(n) for n in range(1, n_levels+1)])
    g = np.array([2*n**2 for n in range(1, n_levels+1)])
    
    # Initialize rate matrix
    rate_matrix = np.zeros((n_levels, n_levels))
    
    # Geometric dilution factor for radiation field
    W = 0.5 * (1 - np.sqrt(1 - min((R_sun/R_star_cm)**2, 1.0)))
    
    # Calculate rates between all levels
    for i in range(n_levels):
        for j in range(i+1, n_levels):
            # Energy and frequency of transition
            delta_E = abs(energies[j] - energies[i])
            nu_ij = delta_E/h
            
            # Einstein coefficients
            A_ji = calculate_einstein_A(j+1, i+1)
            B_ij = A_ji * c**2 / (2*h*nu_ij**3)  # Einstein B for absorption
            B_ji = g[i]/g[j] * B_ij  # Einstein B for stimulated emission
            
            # Radiation field
            J_nu = W * planck_function(nu_ij, T_rad)
            
            # Collisional rates (more accurate approximation)
            C_ij = 5.465e-4 * n_H / np.sqrt(T_gas) * \
                   g[j]/g[i] * np.exp(-delta_E/(k_B*T_gas))
            C_ji = 5.465e-4 * n_H / np.sqrt(T_gas)
            
            # Add to rate matrix
            rate_matrix[i,j] = -(B_ij * J_nu + C_ij)  # i -> j (upward)
            rate_matrix[j,i] = A_ji + B_ji * J_nu + C_ji  # j -> i (downward)
    
    # Fix diagonal elements (total probability out)
    for i in range(n_levels):
        rate_matrix[i,i] = -np.sum(rate_matrix[:,i])
    
    # Replace last equation with particle conservation
    rate_matrix[-1,:] = 1.0
    
    # RHS vector (all zeros except last element = 1 for normalization)
    rhs = np.zeros(n_levels)
    rhs[-1] = 1.0
    
    # Solve system
    try:
        populations = np.linalg.solve(rate_matrix, rhs)
        # Ensure positivity and normalization
        populations = np.abs(populations)
        populations = populations / np.sum(populations)
    except np.linalg.LinAlgError:
        # Fallback to LTE populations if matrix is singular
        populations = g * np.exp(energies/(k_B*T_gas))
        populations = populations / np.sum(populations)
    
    return populations

def calculate_spectrum(wavelengths, populations, n_levels, T_gas, N_H):
    """Calculate absorption spectrum including lines and bound-free transitions"""
    cross_section = np.zeros_like(wavelengths)
    frequencies = c/(wavelengths*1e-8)  # Convert Å to Hz
    
    # Calculate statistical weights
    g = np.array([2*n**2 for n in range(1, n_levels+1)])
    
    # Loop over all possible transitions
    for n_lower in range(1, n_levels):
        for n_upper in range(n_lower+1, n_levels+1):
            # Line transition energy
            E_transition = abs(hydrogen_energy(n_upper) - hydrogen_energy(n_lower))
            nu_0 = E_transition/h
            
            # Einstein A coefficient
            A21 = calculate_einstein_A(n_upper, n_lower)
            
            # Add line absorption
            profile = voigt_profile(frequencies, nu_0, A21, T_gas, m_p, 0)
            
            # Include stimulated emission correction
            correction = (1 - populations[n_upper-1]*g[n_lower-1]/(populations[n_lower-1]*g[n_upper-1]))
            if correction > 0:  # Only include absorption
                cross_section += populations[n_lower-1] * correction * profile
            
        # Add bound-free absorption
        E_ion = abs(hydrogen_energy(n_lower))
        nu_threshold = E_ion/h
        mask = frequencies > nu_threshold
        
        if np.any(mask):
            # Approximate photoionization cross-section
            sigma_bf = 6.3e-18 * n_lower**(-5) * (frequencies[mask]/nu_threshold)**(-3)
            cross_section[mask] += populations[n_lower-1] * sigma_bf
    
    return cross_section * N_H

st.title("Hydrogen Atom Level Populations and Spectrum")

# Sidebar controls
st.sidebar.header("Physical Parameters")
n_levels = st.sidebar.slider("Number of Levels", 3, 10, 5)
T_gas = st.sidebar.number_input("Gas Temperature (K)", 100., 1e6, 5000.)
n_H = st.sidebar.number_input("Hydrogen Density (cm⁻³)", 1e0, 1e10, 1e4, format="%.1e")
T_rad = st.sidebar.number_input("Radiation Temperature (K)", 1000., 1e5, 10000.)
R_star = st.sidebar.number_input("Distance from Star (pc)", 0.1, 100.0, 1.0)
N_H = st.sidebar.number_input("Column Density N_H (cm⁻²)", 1e10, 1e25, 1e18, format="%.1e")

# Calculate populations
populations = calculate_level_populations(n_levels, T_gas, n_H, T_rad, R_star)

# Calculate spectrum
wavelengths = np.logspace(2.7, 4, 1000)  # 1000-10000 Å
cross_section = calculate_spectrum(wavelengths, populations, n_levels, T_gas, N_H)

# Plotting
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=[f"n={i+1}" for i in range(n_levels)],
    y=populations,
    name="Level Populations"
))
fig1.update_layout(
    title="Hydrogen Level Populations",
    yaxis_type="log",
    yaxis_title="Population Fraction",
    yaxis=dict(range=[-5, 1])  # Set y-axis range from 10^-5 to 10^1
)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=wavelengths,
    y=cross_section,
    mode='lines',
    name="Total Optical Depth"
))
fig2.update_layout(
    title="Absorption Features",
    xaxis_title="Wavelength (Å)",
    yaxis_title="τ(λ)",
    xaxis_type="log",
    yaxis_type="log"
)

st.plotly_chart(fig1)
st.plotly_chart(fig2)
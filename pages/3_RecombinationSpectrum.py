import numpy as np
import streamlit as st
import plotly.graph_objects as go
from physics import recombination_spectrum, R_H, h, c

st.title("🌌 Hydrogen Radiative Recombination Spectrum")

# Sidebar inputs
st.sidebar.header("Recombination Parameters")
T_e = st.sidebar.number_input("Electron Temperature (K)", 100, 50000, 10000)
n_max = st.sidebar.number_input("Max Quantum Level", 2, 100, 30)
n_e = st.sidebar.number_input("Electron Density (cm⁻³)", 1e1, 1e8, 1e4, format="%.1e")
case_b = st.sidebar.checkbox("Case B (exclude Lyman)", True)
two_photon = st.sidebar.checkbox("Include two-photon decay", True)
unit = st.sidebar.radio("X-axis Unit", ["Wavelength (Å)", "Frequency (THz)"])

# Frequency grid
nu_min = R_H/h * (1/n_max**2)
nu_max = R_H/h * 5
nu = np.logspace(np.log10(nu_min), np.log10(nu_max), 5000)

# Calculate spectrum
spectrum, wavelength = recombination_spectrum(
    T_e=T_e,
    n_max=n_max,
    n_e=n_e,
    nu_grid=nu,
    case_b=case_b,
    two_photon=two_photon
)

# Dynamic x-axis values
x = wavelength if unit.startswith("Wavelength") else nu/1e12  # Convert to THz
x_label = "Wavelength (Å)" if unit.startswith("Wavelength") else "Frequency (THz)"

# Lyman limit marker
lyman_limit_nu = R_H/h
lyman_limit_x = c/lyman_limit_nu*1e8 if unit.startswith("Wavelength") else lyman_limit_nu/1e12

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, 
    y=spectrum,
    mode='lines', 
    name='Spectrum',
    line=dict(color='blue', width=2)
))

# Add Lyman limit marker
fig.add_vline(
    x=lyman_limit_x,
    line=dict(color="red", width=1, dash="dot"),
    annotation_text="Lyman Limit",
    annotation_position="top right"
)

fig.update_layout(
    title="Hydrogen Recombination Spectrum",
    xaxis_title=x_label,
    yaxis_title="Intensity (erg s⁻¹ Hz⁻¹)",
    xaxis_type="log",
    yaxis_type="log",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)


# Theory section
st.markdown(r"""
# DRAFT PAGE 
## Recombination Physics

The spectrum is calculated using:

$$ \alpha_n(T_e) = 2.07 \times 10^{-11} n^{-3} \left(\frac{T_e}{10^4\,\text{K}}\right)^{-0.75} $$

Population distribution:
$$ N_n = \frac{\alpha_n}{\sum_k \alpha_k} $$

Line emission:
$$ L_\nu = h\nu A_{nn'} N_n n_e \phi(\nu) $$

With Voigt profile combining:
- Thermal broadening: $\Delta\nu_{\text{thermal}} = \nu_0 \sqrt{\frac{k_B T}{m_e c^2}}$
- Natural broadening: $\Delta\nu_{\text{natural}} = \frac{A}{4\pi}$
""")
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from physics import c, h, m_e


def polarized_compton_cross_section(ratio, Q, U, theta, phi): #ratio is omega_out / omega_in
    e = 0.30282212088 #electron charge in natural units
    m = .511*.001 #electron mass in GeV (natural units)

    #define polarization and wavenumber quantities
    s0 = np.sqrt(Q**2 + U**2)
    s1 = Q
    ex = np.sqrt((s0+s1)/(2*s0))
    ey = np.sqrt((s0-s1)/(2*s0))
    kxhat = np.sin(theta)*np.cos(phi)
    kyhat = np.sin(theta)*np.sin(phi)
    dot = ex*kxhat + ey*kyhat
    
    #define other factors for the cross section
    prefactor = e**4 * ratio**2 / (64*np.pi**2 * m**2)
    term1 = 1/ratio
    term2 = ratio

    conversion_to_si = 38940/(10**16)**2 #convert from GeV ^ -2 (in natural units) to cm^2 in SI

    return prefactor * (term1 + term2 - 2 * dot**2) * conversion_to_si

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


st.title("☢️ Polarized Compton Scattering Differential Cross-Section")

sigma_thomson = compton_cross_section(1e-3)  # Thomson limit


# Sidebar controls
st.sidebar.header("Range of Outgoing/Ingoing photon frequencies")
min_ratio = st.sidebar.number_input("Min Ratio", 1.0* 1e-3, 1.0, 0.1, format="%.1e")
max_ratio = st.sidebar.number_input("Max Ratio", 1.0* 1e-3, 1.0, 0.99, format="%.1e")

#now add some sliders for Q, U, theta, phi
st.sidebar.header("Stokes Parameters")
st.sidebar.write("Stokes parameters here are normalized such that I = 1, V = 0.")
Q = st.sidebar.slider("Stokes Q", min_value = 0.0001, max_value = 1.0, step = .01)
U = np.sqrt(1 - Q**2)
st.sidebar.write("Stokes U", U)
#U = st.sidebar.slider("Stokes U", min_value = 0.0001, max_value = 1.0, step = .01)
st.sidebar.header("Position of Source")
theta = st.sidebar.slider("theta of source", min_value = 0.0, max_value = np.pi, step = .1)
phi = st.sidebar.slider("phi of source", min_value = 0.0, max_value = 2*np.pi, step = .1)

# ratio grid
ratios = np.logspace(np.log10(min_ratio), np.log10(max_ratio), 1000)

# Calculate cross-sections
sigma = [polarized_compton_cross_section(ratio, Q, U, theta, phi) for ratio in ratios]

# Unit conversions
x = ratios
x_label = f"Ratio Outgoing/Ingoing Photon Frequency"

# Create plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, 
    y=sigma,
    mode='lines',
    name='Klein-Nishina',
    line=dict(color='royalblue', width=2)
))

# Add reference lines

# Format plot
fig.update_layout(
    title="Polarized Compton Scattering Differential Cross-Section",
    xaxis_title=x_label,
    yaxis_title="Cross Section (cm²)",
    xaxis_type="log",
    yaxis_type="log",
    yaxis_range=[np.log10(sigma_thomson/1e3), np.log10(sigma_thomson*1.1)],  # Focus on relevant range
    yaxis=dict(
        tickformat=".1e",
        dtick=1  # Logarithmic tick spacing
    ),
    template="plotly_white",
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

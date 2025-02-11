import numpy as np
import streamlit as st
import plotly.graph_objects as go
from physics import photoionization_cross_section_cgs, c, R_H, h, alpha, a0

# Title
st.title("☢️ Photoionization Modeling")

# Sidebar for user inputs
st.sidebar.header("Photoionization Parameters")
Z = st.sidebar.number_input("Atomic Charge Number (Z)", min_value=1, value=1, step=1)
N_H = st.sidebar.number_input("Column Density $N_H$ (cm⁻²)", min_value=1e10, max_value=1e26, value=1e20, format="%.1e")
unit = st.sidebar.radio("X-axis Unit", ["Wavelength (Å)", "Frequency (THz)"])

# Calculate threshold frequency
nu_0 = (Z**2 * R_H) / h  # Threshold frequency in Hz

# Convert threshold frequency to THz, eV, and Å
nu_0_THz = nu_0 / 1e12  # Threshold frequency in THz
E_0_eV = nu_0 * h / 1.60218e-12  # Threshold energy in eV (1 erg = 1e-7 J, 1 eV = 1.60218e-19 J)
lambda_0_A = c / nu_0 * 1e8  # Threshold wavelength in Å
sigma_0 = (64 * np.pi * alpha * a0**2) / (3 * np.sqrt(3) * Z**2)

# Display threshold frequency in sidebar
st.sidebar.markdown(f"""
### Threshold Frequency:
- **Frequency:** {nu_0_THz:.5g} THz
- **Energy:** {E_0_eV:.5g} eV
- **Wavelength:** {lambda_0_A:.5g} Å
### Threshold Cross Section:
- $\sigma_0 = $ {sigma_0:.4g} cm²
""")

# Markdown section with LaTeX
st.markdown(r"""
### Photoionization Cross-Section Formula

The photoionization cross-section for hydrogenic atoms is given by:

$$
\sigma_{\text{PI}}(\nu, Z) = \sigma_0 \left(\frac{\nu_0}{\nu}\right)^3 \frac{\exp\left[-4 \frac{\arctan(\tau)}{\tau}\right]}{1 - \exp(-2\pi/\tau)}
$$

where:
- $\sigma_0 = \frac{64 \pi \alpha a_0^2}{3 \sqrt{3} Z^2}$ is the cross-section at the threshold frequency $\nu_0$,
- $\nu_0 = \frac{Z^2 R_H}{h}$ is the threshold frequency,
- $\tau = \sqrt{\frac{\nu}{\nu_0} - 1}$,
- $Z$ is the atomic charge number,
- $\nu$ is the frequency of the incident photon.
""")

# Frequency range (in Hz)
nu_min = 0.9 * nu_0  # Start at 0.1 × threshold frequency
nu_max = 1e17  # Maximum frequency (Hz)
nu = np.logspace(np.log10(nu_min), np.log10(nu_max), 1500)  # Log-spaced frequency range

# Calculate cross-section
sigma = photoionization_cross_section_cgs(nu, Z)

# Calculate intensity
intensity = np.exp(-N_H * sigma)

st.markdown(fr"""{sigma[0]:.4g}  {intensity[-1]:.4g}""")

# Convert x-axis based on user selection
if unit == "Wavelength (Å)":
    x_values = c / nu * 1e8  # Convert frequency to wavelength in Å
    xlabel = "Wavelength (Å)"
    threshold_x = lambda_0_A  # Threshold wavelength in Å
else:
    x_values = nu / 1e12  # Convert frequency to THz
    xlabel = "Frequency (THz)"
    threshold_x = nu_0_THz  # Threshold frequency in THz

# Create Plotly figure
fig = go.Figure()

# Add cross-section trace
fig.add_trace(go.Scatter(
    x=x_values,
    y=sigma,
    mode='lines',
    name=f"Cross Section (Z = {Z})",
    line=dict(color='blue', width=2),
    yaxis="y1"
))

# Add intensity trace
fig.add_trace(go.Scatter(
    x=x_values,
    y=intensity,
    mode='lines',
    name=f"Intensity (N_H = {N_H:.1e} cm⁻²)",
    line=dict(color='red', width=2),
    yaxis="y2"
))

# Add vertical line at threshold frequency
fig.add_vline(
    x=threshold_x,
    line=dict(color="black", width=2, dash="dash"),
    annotation_text=f"Threshold ({threshold_x:.4g} {xlabel.split(' ')[0]})",
    annotation_position="top right"
)

# Update layout for log-log scale and dual y-axes
fig.update_layout(
    title="Photoionization Cross-Section and Intensity",
    xaxis_title=xlabel,
    yaxis_title="Cross Section (cm²)",
    yaxis2=dict(
        title="Intensity",
        overlaying="y",
        side="right",
        range=[0, 1.1]  # Intensity ranges from 0 to 1
    ),
    xaxis_type="log",
    yaxis_type="log",
    template="custom"  # Explicitly use the custom template
)


# Display the plot
st.plotly_chart(fig, use_container_width=True)
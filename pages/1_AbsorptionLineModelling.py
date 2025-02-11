import numpy as np
import streamlit as st
import plotly.graph_objects as go
from physics import voigt_profile, calculate_absorption, c, k_B, h

st.title("Absorption Line Modeling")
st.markdown(r"""
### Voigt profile $\phi(\nu)$ combines with Einstein coefficient $A_{21}$ to form the cross section
$$
\sigma(\nu) = \frac{h\nu_0}{4\pi}\frac{A_{21}\, c^2}{2h\nu_0^3}\phi(\nu)
$$
""")

# Sidebar controls
st.sidebar.header("Absorption Line Parameters")
lambda_0 = st.sidebar.number_input("λ₀ (Å)", 1e-5, 1e10,value=1215.67, format="%.2f")
lambda_delta = st.sidebar.number_input("δλ/λ₀", 1e-5, 5e-2, 1e-3, format="%.0e")
unit = st.sidebar.radio("X-axis Unit", ["Wavelength (Å)", "Frequency (THz)"])
A21 = st.sidebar.number_input("A₂₁ (s⁻¹)", 1e-20, 1e12, 4.69e8, format="%.2e")
T = st.sidebar.number_input("Temperature (K)", 1e0, 5e6, 1e4)
m = st.sidebar.number_input("Particle Mass (g)", 1e-25, 1e-20,1.67e-24, format="%.2e")
v_shift = st.sidebar.number_input("Velocity Shift (km/s)", -100000.0, 100000.0, 0.0)
N = st.sidebar.number_input("Column Density (cm⁻²)", 1e8, 1e26, 1e18, format="%.2e")

# Core calculations
nu_0 = c / (lambda_0 * 1e-8)
nu, sigma_nu, intensity = calculate_absorption(nu_0, lambda_delta, A21, T, m, v_shift, N)

# Visualization
fig = go.Figure()
x = c/nu*1e8 if unit.startswith("Wavelength") else nu/1e12
fig.add_trace(go.Scatter(x=x, y=sigma_nu, name="Cross Section", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=x, y=intensity, name="Intensity", yaxis="y2", line=dict(color='red')))

# Add vertical line at threshold frequency
linex = c/nu_0*1e8 if unit.startswith("Wavelength") else nu_0/1e12
fig.add_vline(
    x=linex,
    line=dict(color="grey", width=2, dash="dash"),
    annotation_text=f"line ({linex:.4g} )",
    annotation_position="top right"
)


fig.update_layout(
    title="Voigt Profile & Absorption Line",
    xaxis_title="Wavelength (Å)" if unit.startswith("Wavelength") else "Frequency (THz)",
    yaxis_title="Cross Section [cm²]",
    yaxis2=dict(title="Intensity", overlaying="y", side="right"))
st.plotly_chart(fig, use_container_width=True)

# Calculate Doppler broadening
dnu = (nu_0 / c) * np.sqrt(2 * k_B * T / m)  # Doppler broadening
# Display the text dynamically
st.sidebar.markdown(f"""
### Thermal Line Width:
- Doppler Width: **{dnu:.3e} Hz**
- Converted to Angstrom: **{(c*dnu/(nu_0*(nu_0+dnu))) * 1e8:.4g} Å**
- $$\delta \lambda/\lambda \sim $$ {(dnu/((nu_0+dnu))) * 1e8:.4g}
""")
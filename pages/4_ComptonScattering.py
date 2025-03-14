import numpy as np
import streamlit as st
import plotly.graph_objects as go
from physics import compton_cross_section, c, h, m_e

st.title("☢️ Compton Scattering Cross-Section")

sigma_thomson = compton_cross_section(1e-3)  # Thomson limit

st.markdown(r"""
### Klein-Nishina Cross Section
The total cross-section for Compton scattering:
$$
\sigma_{\text{KN}}(E) = \sigma_T \cdot \frac{3}{4} \left[
\frac{1+x}{x^3}\left(\frac{2x(1+x)}{1+2x} - \ln(1+2x)\right) + 
\frac{\ln(1+2x)}{2x} - \frac{1+3x}{(1+2x)^2}
\right]
$$
where:
- $\sigma_T = \frac{8\pi}{3}r_e^2 = $  (Thomson cross-section)
- $x = \frac{E_\gamma}{m_e c^2}$ (Dimensionless photon energy)
- $r_e = \frac{e^2}{m_e c^2}$ (Classical electron radius)
""")

# Sidebar controls
st.sidebar.header("Photon Energy Range")
min_energy = st.sidebar.number_input("Min Energy (eV)", 1e-3, 1e12, 1e-3, format="%.1e")
max_energy = st.sidebar.number_input("Max Energy (eV)", 1e-3, 1e12, 1e6, format="%.1e")
unit = st.sidebar.radio("Energy Unit", ["eV", "keV", "MeV", "GeV"],index=2)

# Energy grid
E_eV = np.logspace(np.log10(min_energy), np.log10(max_energy), 1000)

# Calculate cross-sections
sigma = [compton_cross_section(E) for E in E_eV]

# Unit conversions
conversion = {"eV": 1, "keV": 1e3, "MeV": 1e6, "GeV": 1e9}
x = E_eV / conversion[unit]
x_label = f"Photon Energy ({unit})"

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
m_e_c2 = m_e * c**2 / 1.602e-12  # Convert m_e c² to eV
fig.add_hline(y=sigma_thomson, line=dict(color='grey', dash='dash'),
             annotation_text=f"Thomson Limit ({sigma_thomson:.2e} cm²)")
fig.add_vline(x=m_e_c2/conversion[unit], line=dict(color='red', dash='dot'),
             annotation_text="m_e c²")

# Format plot
fig.update_layout(
    title="Compton Scattering Cross-Section",
    xaxis_title=x_label,
    yaxis_title="Cross Section (cm²)",
    xaxis_type="log",
    yaxis_type="log",
    yaxis_range=[np.log10(sigma_thomson/1e4), np.log10(sigma_thomson*1.1)],  # Focus on relevant range
    yaxis=dict(
        tickformat=".1e",
        dtick=1  # Logarithmic tick spacing
    ),
    template="plotly_white",
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

# Physics notes
st.markdown(r"""
## Energy Regimes

**Low Energy Limit (E ≪ m_ec²):**
$$
\sigma \approx \sigma_T \left(1 - 2x + \frac{26}{5}x^2 - \cdots\right)
$$

**High Energy Limit (E ≫ m_ec²):**
$$
\sigma \approx \frac{3}{8}\sigma_T \frac{1}{x}\left(\ln(2x) + \frac{1}{2}\right)
$$

**Key Transition:**
- $E = m_e c^2 \approx 511$ keV (electron rest mass energy)
- Cross-section decreases as ~1/E at high energies
""")

# Add to physics.py
st.sidebar.markdown(f"""
### Current Parameters:
- Energy range: {min_energy:.1e} - {max_energy:.1e} eV
- Electron rest mass: {m_e_c2/1e3:.1f} keV
- Thomson cross-section: {sigma_thomson:.4e} cm²
""")

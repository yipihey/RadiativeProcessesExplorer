#5_InverseComptonScattering.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from physics import compton_cross_section, bb, c, h, k_B, m_e, calculate_ic_spectrum

st.title("☢️ Inverse Compton Scattering")
st.markdown(r"""
### Non-thermal Radiation Mechanism
Inverse Compton scattering converts electron kinetic energy to photon energy:
$$
\frac{E_\text{final}}{E_\text{initial}} \sim \gamma^2 \quad \text{(Thomson regime)}
$$
where $\gamma$ is the Lorentz factor of relativistic electrons
""")

# Sidebar controls
st.sidebar.header("Parameters")
T_bb = st.sidebar.number_input("Blackbody Temperature (K)", 1e0, 1e9, 1e4)
E_e_min = st.sidebar.number_input("Min Electron Energy (γ)", 1.0, 1e6, 10.0)
E_e_max = st.sidebar.number_input("Max Electron Energy (γ)", 1.0, 1e6, 1e3)
dist_type = st.sidebar.radio("Electron Distribution", ["Power Law", "Maxwellian"])

# Distribution parameters
if dist_type == "Power Law":
    p = st.sidebar.slider("Spectral Index (p)", 1.5, 5.0, 2.0)
else:
    T_e = st.sidebar.number_input("Electron Temp (K)", 1e6, 1e12, 1e8)

# Energy grids (input photons)
nu_input = np.logspace(np.log10(k_B*T_bb/h)-5, np.log10(k_B*T_bb/h)+5, 1000)
gamma = np.logspace(np.log10(E_e_min), np.log10(E_e_max), 100)

# Initialize spectra using vectorized function
bb_spec = bb(nu_input, T_bb)


# Calculate electron distribution
if dist_type == "Power Law":
    n_e = gamma**-p
else:  # Relativistic Maxwellian
    theta = k_B*T_e/(m_e*c**2)
    n_e = gamma**2 * np.exp(-gamma/theta)
n_e /= np.trapz(n_e, gamma)  # Normalize


# Calculate IC spectrum using improved method
nu_output, ic_spec = calculate_ic_spectrum(nu_input, bb_spec, gamma, n_e, T_bb)

# Filter zeros and handle numerical stability
valid = ic_spec > 1e-40 * ic_spec.max()
nu_plot = nu_output[valid]
ic_plot = ic_spec[valid]

# Plotting
# Filter zeros using relative threshold
valid = ic_spec > 1e-6 * ic_spec.max()
nu_plot = nu_output[valid]
ic_plot = ic_spec[valid]

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=nu_input, y=bb_spec, name="Input Blackbody"))
fig.add_trace(go.Scatter(x=nu_plot, y=ic_plot, name="IC Spectrum",
                        line=dict(width=2)))

# Add characteristic energy lines (updated)
E_bb = k_B*T_bb/h
fig.add_vline(x=E_bb, line=dict(color='grey', dash='dash'),
             annotation_text=f"kT_bb: {E_bb:.1e} Hz")
fig.add_vline(x=E_bb * gamma[-1]**2, line=dict(color='red', dash='dot'),
             annotation_text=f"Max Boost: {gamma[-1]**2:.1e}x")

# Set y-axis limits dynamically
if len(ic_plot) > 0:
    y_min = max(1e-40, 0.1 * ic_plot.min())  # Prevent zero/negative values
    y_max = 10 * ic_plot.max()
else:
    y_min = 1e-40
    y_max = 1e-20

fig.update_layout(
    title="Inverse Compton Spectrum",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Specific Intensity (erg cm⁻² s⁻¹ Hz⁻¹)",
    xaxis_type="log",
    yaxis_type="log",
    template="plotly_white",
    yaxis=dict(
        range=[np.log10(y_min), 14+np.log10(y_max)],  # Log axis requires log values
        autorange=False
    )
)
st.plotly_chart(fig, use_container_width=True)

# Theory section
st.markdown(r"""
## Inverse Compton Formalism

### Electron Distributions
**Power Law:**
$$
N(\gamma) \propto \gamma^{-p} \quad (\gamma_{\text{min}} \leq \gamma \leq \gamma_{\text{max}})
$$

**Relativistic Maxwellian:**
$$
N(\gamma) \propto \gamma^2 e^{-\gamma/\theta} \quad (\theta = kT_e/m_e c^2)
$$

### Scattering Process
Photon energy gain per scattering:
$$
\frac{\nu_f}{\nu_i} \approx \frac{4}{3}\gamma^2 \frac{h\nu_i}{m_e c^2} \quad \text{(Thomson regime)}
$$

Total spectrum combines contributions from all electrons:
$$
I_{\text{IC}}(\nu) = \int N_e(\gamma) \sigma_{\text{KN}}(\gamma) I_{\text{bb}}(\nu/\gamma^2) d\gamma
$$
""")

# Add to physics.py
st.sidebar.markdown(f"""
### Current Parameters:
- Electron energy range: γ = {E_e_min:.1f} - {E_e_max:.1f}
- Max photon energy: {gamma[-1]**2*T_bb:.2e} keV
- Dominant scattering regime: {'Thomson' if np.max(boost) < 1 else 'Klein-Nishina'}
""")
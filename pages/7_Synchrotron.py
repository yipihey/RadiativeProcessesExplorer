import numpy as np
import streamlit as st
import plotly.graph_objects as go
from physics import c, nu_crit, single_particle_synchrotron, total_synchrotron_spectrum

st.title("🧲 Synchrotron Emission Modeling")
st.markdown(r"""
### Synchrotron Radiation - Radiation from a Particle Accelerated by a Uniform Magnetic Field
Charged particles radiate when accelerated by magnetic fields. Ultrarelatvistic electrons whose velocities are much smaller than the speed of light emit synchrotron radiation. Cosmic rays are celestial particles (e.g., electrons, protons, and heavier nuclei) with extremely high energies. Cosmic-ray electrons in the interstellar magnetic field emit the synchrotron radiation that accounts for most of the continuum emission from our Galaxy at frequencies below about 30 GHz.

### Synchrotron Radiation From a Single Electron
Assuming a particle of mass $m$ and charge $q$ in a uniform magnetic field of magnitude $B$ at a pitch angle, $\alpha$. The total radiated power is then:
$$
P = 2\sigma_Tc\beta^2\gamma^2U_Bsin^2\alpha
$$
where $U_B$ and $\sigma_T$ are the magnetic enegry density and the Thomson cross section, respectively, defined as:
$$
U_B = \frac{B^2}{8\pi}, \sigma_T = \frac{8\pi}{3}(\frac{e^2}{m_ec^2})^2
$$
For an electron, which we will use here, this comes out to be $\sigma_T = 6.65*10^{-25}$.

### Synchrotron Power Spectrum for a Single Electron
For a single particle with some Lorentz factor, $\gamma$, the synchrotron radiation power spectrum as a function of freqency is:
$$
P(\nu, \gamma) = \frac{\sqrt{3} e^3 B \sin \alpha}{m_e c^2} \left(\frac{\nu}{\nu_c} \right) \int_{\nu/\nu_c}^{\infty} K_{5/3}(\eta) \, d\eta
$$
where $K_{5/3}(\eta)$ is a modified Bessel function and the critical frequency is $\nu_c=3/2 \gamma^2 \nu_G sin\alpha$.

#### Synchtrotron Power Spectrum of a Distribution of Electrons
For a power law distribution of particle energies with index p over a sufficiently broad energy range given by:
$$
N(\gamma) \propto \gamma^{-p}
$$
The total power per unit volume per unit frequency is:
$$
P_{\text{total}}(\nu) = \int_{\gamma_{\min}}^{\gamma_{\max}} P(\nu, \gamma) N(\gamma) \, d\gamma
$$

It sounds like in the total spectrum, and absorption feature emerges for stronger magnetic fields. This spectral feature is thought to be a gyro-resonance absorption, in which case the frequency of this absorption line directly measures the magnetic field strength.

Credit to \url{https://www.cv.nrao.edu/~sransom/web/Ch5.html}
""")

# Sidebar inputs
st.sidebar.header("Parameters")
B = st.sidebar.number_input("Magnetic Field (Gauss)", 1e-20, 1e4, 1e-4, format="%.1e")
alpha = st.sidebar.number_input("Particle Pitch Angle (Radians)", 1e-20, 2*np.pi, .79e-1, format="%.1e")
gamma_fixed = st.sidebar.number_input("Lorentz Factor of Single Particle", 1e0, 1e5, 1e2, format="%.1e")
p = st.sidebar.number_input("Energy Spectrum Power Law Index", 1e-20, 1e2, 2e0, format="%.1e")

nu_min = 1e1  # Hz
nu_max = 1e12  # Hz
nu = np.logspace(np.log10(nu_min), np.log10(nu_max), 500)

gamma_min = 10
gamma_max = 1e5

gamma_char = np.sqrt(gamma_min * gamma_max)  # Geometric mean
nu_crit_char = nu_crit(gamma_char, B, alpha)

nu_crit_fixed = nu_crit(gamma_fixed, B, alpha)
spectrum_single = np.array([single_particle_synchrotron(n, B, gamma_fixed, alpha) for n in nu])

spectrum_total = np.array([total_synchrotron_spectrum(n, B, alpha, gamma_min, gamma_max, p) for n in nu])

x_single = nu / nu_crit_fixed
x_total = nu / nu_crit_char

x_label = "ν/νcrit"

# Plotting
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_single, 
    y=spectrum_single,
    mode='lines', 
    name='Single Particle Spectrum',
    line=dict(color='magenta', width=2),
    yaxis='y1'
))

fig.add_trace(go.Scatter(
    x=x_total, 
    y=spectrum_total,
    mode='lines', 
    name='Total Particle Spectrum',
    line=dict(color='green', width=2),
    xaxis='x2',
    yaxis='y2'
))

fig.update_layout(
    title="Synchrotron Emission Power",
    
    xaxis=dict(
        title=x_label,
        type="log",
        range=[-2, 1] 
    ),
    yaxis=dict(
        title="Single Particle Power (erg s⁻¹ Hz⁻¹ cm⁻³)",
        type="log",
        range=[-28, -25]
    ),

    xaxis2=dict(
        title=x_label,
        overlaying="x",
        type="log",
        side="top",
        range=[-5, 1]  # Fixed range
    ),
    yaxis2=dict(
        title="Total Power (erg s⁻¹ Hz⁻¹ cm⁻³)",
        overlaying="y",
        type="log",
        side="right",
        range=[-31,None],
    ),

    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
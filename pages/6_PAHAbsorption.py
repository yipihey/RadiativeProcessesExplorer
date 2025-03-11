import numpy as np
import streamlit as st
import plotly.graph_objects as go

from physics import Blackbody, PAHSpectrum

st.title("PAH Absorption Line Modeling")

st.markdown("""
This is a model of the PAH absorption spectrum against a blackbody. The PAH absorption model is taken from 
            [Li & Draine 2001](https://arxiv.org/abs/astro-ph/0011319). The model generates absorption from a single PAH population described by the
            carbon atom column number density of the PAH molecules, the hydrogen-carbon atom ratio of the molecule,
            and whether or not the population is ionized. The emission features of these lines are based on Drude profiles. Additionally, 
            the temperature of the blackbody can be set.

**NOTE**: Only the wavelengths greater than 0.6 micron are modeled accurately (PAHs behave like carbanaceous particles at those wavelengths, which
            are much harder to model).

The programmed model is included in `pah.py`, and the Drude profile model parameters from Draine and Li are stored in `drude_params.csv`. All of these are stored in the main directory.
""")
# Sidebar controls
st.sidebar.header("Absorption Line Parameters")


HC = st.sidebar.slider("PAH Hydrogen-Carbon atom ratio", 0.25, 0.5, 0.25, format="%.2f")
T = st.sidebar.slider("Blackbody Temperature (K)", 1.0, 2000.0, 290.0, format="%.2f")
N = st.sidebar.slider("Carbon Column Density (cm⁻²)", 1e18, 1e22, 1e20, format="%.2e")
ionization = st.sidebar.checkbox("Ionized PAH", value=True)

# Core calculations
# Visualization
fig = go.Figure()

# Generate wavelength range
lams = np.logspace(0, 2, 1000)

# Calculate absorption spectrum and blackbody spectrum
pah = PAHSpectrum(N, HC, ionization)
bb = Blackbody(T)
abs_spectrum = np.array([pah.absorption_cross_section(lam_i) for lam_i in lams])
bb_spectrum = np.array([bb.spectrum(lam_i) for lam_i in lams])

# Plot the spectra
fig.add_trace(go.Scatter(x=lams, y=bb_spectrum * np.exp(-N * abs_spectrum), name="Absorbed Spectrum", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=lams, y=bb_spectrum, name="Blackbody Spectrum", line=dict(color='red')))

# Set log scale for axes
fig.update_layout(
    xaxis_type="log",
    yaxis_type="log",
    xaxis_title="Wavelength (µm)",
    yaxis_title="Intensity (J s⁻¹ m⁻² µm⁻¹)"
)

st.plotly_chart(fig)

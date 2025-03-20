import streamlit as st
import plotly.io as pio

# Define global Plotly figure styling
pio.templates["custom"] = pio.templates["plotly_white"]

# Update layout defaults
pio.templates["custom"].layout.update(
    font=dict(size=28),  # Increase font size
    title_font=dict(size=32),  # Increase title font
    xaxis=dict(title_font=dict(size=28), tickfont=dict(size=26)),  # X-axis labels
    yaxis=dict(title_font=dict(size=28), tickfont=dict(size=26)),  # Y-axis labels
    legend=dict(font=dict(size=16)),  # Legend font size
    margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins
)
# Set global spline interpolation
pio.templates["custom"].data.scatter = [
    dict(line=dict(shape="spline", smoothing=0.6))  # Set default to spline
]

# Apply globally
pio.templates.default = "custom"

st.set_page_config(page_title="Radiative Processes Explorer", layout="wide")
st.title("🌌 Radiative Processes Explorer")
st.markdown("""
Welcome to the Radiative Processes Explorer! Select a page from the sidebar to explore different radiative processes.
""")
pg = st.navigation([st.Page("./pages/1_AbsorptionLineModelling.py"),
                    st.Page("./pages/2_PhotoIonizationModelling.py"),
                    st.Page("./pages/3_RecombinationSpectrum.py"),
                    st.Page("./pages/4_ComptonScattering.py"),
                    st.Page("./pages/5_InverseComptonScattering.py"),
                    st.Page("./pages/6_PAHAbsorption.py"),
                    st.Page("./pages/7_PolarizedComptonScattering.py"),
#                   ,st.Page("./pages/8_MultiLevelHydrogenAbsorption.py"), # doesn't work yet
                    st.Page("./pages/9_Synchrotron.py")
                    ])
pg.run()

import streamlit as st

st.set_page_config(
    page_title="CoralMD",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------- simple CSS to get a clean hero look ----------
st.markdown(
    """
    <style>
    .main {
        background-color: #fff9f0;  /* warm off-white like your Figma */
    }
    .coral-hero {
        padding: 4rem 3rem 3rem 3rem;
    }
    .coral-logo {
        font-size: 4.5rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        font-style: italic;
    }
    .coral-md {
        font-size: 2.2rem;
        vertical-align: super;
        margin-left: 0.15em;
    }
    .coral-tagline {
        font-size: 1.1rem;
        margin-top: 0.75rem;
    }
    .role-label {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- layout ----------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="coral-hero">', unsafe_allow_html=True)
    st.markdown(
        '<div class="coral-logo">CORAL<span class="coral-md">MD</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="coral-tagline">'
        'Reimagining healthcare: a precision medicine platform.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("")
    st.markdown('<div class="role-label">Continue as…</div>', unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.page_link(
            "pages/1_Patient_Home.py",
            label="🧍 Patient",
            help="Upload your own data and see the prototype risk view.",
        )
    with c2:
        st.page_link(
            "pages/2_Practitioner_Home.py",
            label="🩺 Practitioner",
            help="Browse patients and explore multimodal dashboards.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # Use your DNA / coral image here — external URL or a local file.
    # If you have a local image, drop it in the repo and change to st.image('dna_hero.png')
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/0/0a/Coral_Reef.jpg",
        use_column_width=True,
    )

st.markdown("---")
st.caption("CoralMD — Pomona College DS190 prototype. Not medical advice.")

"""
Material Fatigue Strength Predictor — SaaS Entry Point
Streamlit multi-page application
"""

import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MatPredict | Steel Fatigue Strength",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Brand palette ── */
    :root {
        --primary:   #1a73e8;
        --accent:    #ff6d00;
        --surface:   #0d1117;
        --card:      #161b22;
        --border:    #30363d;
        --text:      #e6edf3;
        --muted:     #8b949e;
    }

    /* ── Global typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: var(--text);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--card);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* ── Hide default Streamlit decoration ── */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ── Metric card polish ── */
    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 20px;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--primary);
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: .85; }

    /* ── Hero section ── */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a73e8, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: var(--muted);
        margin-top: .5rem;
    }

    /* ── Stat card ── */
    .stat-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .stat-num { font-size: 2.2rem; font-weight: 700; color: #1a73e8; }
    .stat-lbl { font-size: .85rem; color: var(--muted); margin-top: 4px; }

    /* ── Feature pill ── */
    .pill {
        display: inline-block;
        background: #1a73e820;
        border: 1px solid #1a73e8;
        color: #1a73e8;
        border-radius: 999px;
        padding: 4px 14px;
        font-size: .8rem;
        font-weight: 600;
        margin: 4px 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SESSION-STATE DEFAULTS
# ─────────────────────────────────────────────
_state_defaults = dict(
    prediction_done=False,
    fatigue=None,
    yield_strength=None,
    uts=None,
    hardness=None,
    steel_type=None,
    last_inputs={},
)
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# SIDEBAR  (shared across all pages)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ **MatPredict**")
    st.markdown(
        "<span style='color:#8b949e;font-size:.85rem;'>"
        "Steel Fatigue Strength Predictor"
        "</span>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Navigate**")
    st.page_link("app.py",                        label="🏠  Home",          icon=None)
    st.page_link("pages/1_Prediction.py",         label="🔮  Predict",       icon=None)
    st.page_link("pages/2_Analytics.py",          label="📊  Analytics",     icon=None)
    st.divider()
    st.markdown(
        "<span style='color:#8b949e;font-size:.78rem;'>"
        "Model: Random Forest · 100 estimators<br>"
        "Dataset: 437 steel specimens<br>"
        "Target: Fatigue Strength (MPa)"
        "</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
col_hero, col_img = st.columns([3, 2], gap="large")

with col_hero:
    st.markdown(
        """
        <div style='padding-top:2rem;'>
          <div class='hero-title'>Predict Steel Fatigue<br>Strength with ML</div>
          <div class='hero-sub'>
            Input chemical composition &amp; heat-treatment parameters
            to instantly obtain fatigue strength, yield strength, UTS,
            and hardness estimations — all powered by a Random Forest
            model trained on 437 real-world specimens.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🔮  Start Predicting", use_container_width=True):
            st.switch_page("pages/1_Prediction.py")
    with col_b2:
        if st.button("📊  View Analytics", use_container_width=True):
            st.switch_page("pages/2_Analytics.py")

with col_img:
    st.markdown(
        """
        <div style='padding-top:2.5rem; text-align:center;'>
          <svg viewBox="0 0 320 220" xmlns="http://www.w3.org/2000/svg" width="100%">
            <defs>
              <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#1a73e8" stop-opacity="0.9"/>
                <stop offset="100%" stop-color="#00c6ff" stop-opacity="0.7"/>
              </linearGradient>
            </defs>
            <!-- Background grid -->
            <rect width="320" height="220" rx="16" fill="#161b22" stroke="#30363d"/>
            <line x1="40" y1="180" x2="290" y2="180" stroke="#30363d" stroke-width="1"/>
            <line x1="40" y1="140" x2="290" y2="140" stroke="#30363d" stroke-width="1"/>
            <line x1="40" y1="100" x2="290" y2="100" stroke="#30363d" stroke-width="1"/>
            <line x1="40" y1="60"  x2="290" y2="60"  stroke="#30363d" stroke-width="1"/>
            <!-- Stress-strain curve -->
            <polyline points="40,178 80,130 110,100 140,88 170,82 200,78 230,76 260,75 285,76"
              fill="none" stroke="url(#g1)" stroke-width="3" stroke-linecap="round"/>
            <!-- Annotation dots -->
            <circle cx="80"  cy="130" r="5" fill="#1a73e8"/>
            <circle cx="200" cy="78"  r="5" fill="#ff6d00"/>
            <!-- Labels -->
            <text x="85"  y="125" fill="#1a73e8" font-size="10">Yield</text>
            <text x="205" y="73"  fill="#ff6d00" font-size="10">UTS</text>
            <text x="40"  y="198" fill="#8b949e" font-size="9">0</text>
            <text x="265" y="198" fill="#8b949e" font-size="9">Strain →</text>
            <text x="8"   y="80"  fill="#8b949e" font-size="9">σ</text>
          </svg>
          <div style='color:#8b949e;font-size:.78rem;margin-top:6px;'>
            Typical stress–strain profile generated by MatPredict
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────
# STATS ROW
# ─────────────────────────────────────────────
st.markdown("### Platform Highlights")
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("437", "Training specimens"),
    ("25",  "Input features"),
    ("R² ≈ 0.97", "Cross-val accuracy"),
    ("RF · 100", "Ensemble estimators"),
]
for col, (num, lbl) in zip([c1, c2, c3, c4], stats):
    with col:
        st.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-num'>{num}</div>"
            f"<div class='stat-lbl'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ─────────────────────────────────────────────
# FEATURE OVERVIEW
# ─────────────────────────────────────────────
st.markdown("### What Gets Predicted?")
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown("**Chemical Composition Inputs**")
    for pill in ["%C  Carbon", "%Si  Silicon", "%Mn  Manganese",
                 "%P  Phosphorus", "%S  Sulfur", "%Ni  Nickel",
                 "%Cr  Chromium", "%Cu  Copper", "%Mo  Molybdenum"]:
        st.markdown(f"<span class='pill'>{pill}</span>", unsafe_allow_html=True)

    st.markdown("<br>**Heat-Treatment Process Inputs**", unsafe_allow_html=True)
    for pill in ["NT  Normalizing Temp", "THT  Homogenisation Temp",
                 "CT  Cooling Temp", "TT  Tempering Temp",
                 "DT  Dipping Temp", "RedRatio"]:
        st.markdown(f"<span class='pill'>{pill}</span>", unsafe_allow_html=True)

with col_r:
    st.markdown("**Predicted Properties**")
    rows = [
        ("⚡ Fatigue Strength", "MPa", "Primary ML output — direct from model"),
        ("🔩 Yield Strength",   "MPa", "0.6 × Fatigue Strength"),
        ("💪 Ultimate Tensile", "MPa", "1.2 × Yield Strength"),
        ("🔵 Brinell Hardness", "BHN", "0.18 × Fatigue Strength"),
        ("🏭 Steel Grade",      "—",   "Classified from hardness range"),
    ]
    for name, unit, desc in rows:
        st.markdown(
            f"""
            <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;
                        padding:12px 16px;margin-bottom:8px;'>
              <span style='font-weight:600;'>{name}</span>
              <span style='color:#1a73e8;font-size:.8rem;margin-left:8px;'>{unit}</span><br>
              <span style='color:#8b949e;font-size:.82rem;'>{desc}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:.8rem;'>"
    "MatPredict · Built with Streamlit · Random Forest (scikit-learn 1.6.1) · "
    "Data: NIMS Steel Database"
    "</div>",
    unsafe_allow_html=True,
)

"""
pages/1_Prediction.py — Interactive Prediction Dashboard
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import load_assets, run_prediction, derive_properties, FEATURE_COLS

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MatPredict | Predict",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SHARED STYLES (duplicated to work standalone)
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root{--primary:#1a73e8;--accent:#ff6d00;--surface:#0d1117;
          --card:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;}
    html,body,[class*="css"]{font-family:'Inter','Segoe UI',sans-serif;color:var(--text);}
    [data-testid="stSidebar"]{background:var(--card);border-right:1px solid var(--border);}
    [data-testid="stSidebar"]*{color:var(--text)!important;}
    #MainMenu,footer{visibility:hidden;}
    [data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);
        border-radius:12px;padding:14px 20px;}
    [data-testid="stMetricValue"]{font-size:1.6rem!important;}
    .stButton>button{background:var(--primary);color:#fff;border:none;border-radius:8px;
        padding:.55rem 1.6rem;font-weight:600;transition:opacity .2s;}
    .stButton>button:hover{opacity:.85;}
    .section-hdr{font-size:1.05rem;font-weight:700;color:var(--text);
        border-left:4px solid var(--primary);padding-left:10px;margin:12px 0 8px;}
    .result-banner{background:linear-gradient(135deg,#1a73e820,#00c6ff15);
        border:1px solid #1a73e8;border-radius:14px;padding:18px 22px;margin-bottom:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SESSION-STATE DEFAULTS
# ─────────────────────────────────────────────
_defaults = dict(
    prediction_done=False, fatigue=None, yield_strength=None,
    uts=None, hardness=None, steel_type=None, badge=None, last_inputs={},
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ **MatPredict**")
    st.markdown("<span style='color:#8b949e;font-size:.85rem;'>Steel Fatigue Predictor</span>",
                unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",               label="🏠  Home")
    st.page_link("pages/1_Prediction.py",label="🔮  Predict")
    st.page_link("pages/2_Analytics.py", label="📊  Analytics")
    st.divider()
    if st.session_state.prediction_done:
        st.markdown("**Last Result**")
        st.markdown(f"🔸 Fatigue: **{st.session_state.fatigue:.1f} MPa**")
        st.markdown(f"🔹 Yield:   **{st.session_state.yield_strength:.1f} MPa**")
        st.markdown(f"🔹 UTS:     **{st.session_state.uts:.1f} MPa**")
        st.markdown(f"🔹 Hardness:**{st.session_state.hardness:.1f} BHN**")

# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────
assets = load_assets()
if assets["error"]:
    st.error(f"❌ Asset loading error: {assets['error']}")
    st.stop()

model  = assets["model"]
scaler = assets["scaler"]

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("## 🔮 Prediction Dashboard")
st.markdown(
    "<span style='color:#8b949e;'>Adjust the sliders below and click "
    "<b>Run Prediction</b> to compute material properties.</span>",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────
# INPUT FORM  (3-column layout)
# ─────────────────────────────────────────────
col_chem, col_proc1, col_proc2 = st.columns([1.1, 1, 1], gap="large")

# ── Chemical composition ────────────────────────────────────────────────
with col_chem:
    st.markdown("<div class='section-hdr'>Chemical Composition (%wt)</div>",
                unsafe_allow_html=True)
    C   = st.slider("Carbon  %C",       0.17, 0.63, 0.39, 0.01,
                    help="Carbon content by weight percent")
    Si  = st.slider("Silicon  %Si",     0.16, 2.05, 0.30, 0.01,
                    help="Silicon content")
    Mn  = st.slider("Manganese  %Mn",   0.37, 1.60, 0.82, 0.01,
                    help="Manganese content")
    P   = st.slider("Phosphorus  %P",   0.002, 0.031, 0.016, 0.001,
                    format="%.3f", help="Phosphorus (keep low for ductility)")
    S   = st.slider("Sulfur  %S",       0.003, 0.030, 0.015, 0.001,
                    format="%.3f", help="Sulfur (keep low for toughness)")
    Ni  = st.slider("Nickel  %Ni",      0.01, 2.78, 0.52, 0.01,
                    help="Nickel — improves toughness")
    Cr  = st.slider("Chromium  %Cr",    0.01, 1.17, 0.57, 0.01,
                    help="Chromium — improves hardness / corrosion resistance")
    Cu  = st.slider("Copper  %Cu",      0.01, 0.26, 0.07, 0.01,
                    help="Copper content")
    Mo  = st.slider("Molybdenum  %Mo",  0.00, 0.24, 0.07, 0.01,
                    help="Molybdenum — enhances hardenability")

# ── Processing parameters – block 1 ──────────────────────────────────
with col_proc1:
    st.markdown("<div class='section-hdr'>Heat Treatment — Stage 1</div>",
                unsafe_allow_html=True)
    NT     = st.slider("Normalizing Temp NT (°C)",    825, 930, 872,
                       help="Austenitizing / normalizing temperature")
    THT    = st.slider("Homogenisation Temp THT (°C)", 30, 865, 738,
                       help="High-temperature homogenisation soak")
    THt    = st.slider("Homogenisation Time THt (h)",  0, 30,  26,
                       help="Soak duration at THT")
    THQCr  = st.slider("Quench Cooling Rate THQCr",    0, 24,  11,
                       help="Cooling rate after homogenisation")
    CT     = st.slider("Cooling Temp CT (°C)",         30, 930, 129,
                       help="Intermediate cooling temperature")
    Ct     = st.slider("Cooling Time Ct (h)",          0, 540, 41,
                       help="Time held at CT")
    DT     = st.slider("Dipping Temp DT (°C)",         30, 903, 124,
                       help="Dip / immersion temperature")
    Dt     = st.slider("Dipping Time Dt (h)",          0.0, 70.2, 4.8, 0.1,
                       help="Duration of dip treatment")

# ── Processing parameters – block 2 ──────────────────────────────────
with col_proc2:
    st.markdown("<div class='section-hdr'>Heat Treatment — Stage 2</div>",
                unsafe_allow_html=True)
    QmT    = st.slider("Quench Medium Temp QmT (°C)",  30, 140, 35,
                       help="Temperature of quench medium")
    TT     = st.slider("Tempering Temp TT (°C)",       30, 680, 537,
                       help="Tempering temperature")
    Tt     = st.slider("Tempering Time Tt (h)",        0, 120, 65,
                       help="Duration at tempering temperature")
    TCr    = st.slider("Tempering Cooling Rate TCr",   0.0, 24.0, 20.8, 0.1,
                       help="Cooling rate after tempering")
    RedRatio = st.slider("Reduction Ratio",            240, 5530, 924,
                         help="Forging / rolling reduction ratio")

    st.markdown("<div class='section-hdr'>Defect Indicators</div>",
                unsafe_allow_html=True)
    dA = st.slider("Defect Area dA",   0.000, 0.130, 0.047, 0.001, format="%.3f")
    dB = st.slider("Defect Breadth dB",0.000, 0.050, 0.003, 0.001, format="%.3f")
    dC = st.slider("Defect Depth dC",  0.000, 0.058, 0.008, 0.001, format="%.3f")

# ─────────────────────────────────────────────
# PREDICTION BUTTON
# ─────────────────────────────────────────────
st.divider()
btn_col, _ = st.columns([1, 3])
with btn_col:
    predict_clicked = st.button("⚡  Run Prediction", use_container_width=True)

if predict_clicked:
    input_dict = {
        "NT": NT, "THT": THT, "THt": THt, "THQCr": THQCr,
        "CT": CT, "Ct": Ct, "DT": DT, "Dt": Dt,
        "QmT": QmT, "TT": TT, "Tt": Tt, "TCr": TCr,
        "C": C, "Si": Si, "Mn": Mn, "P": P, "S": S,
        "Ni": Ni, "Cr": Cr, "Cu": Cu, "Mo": Mo,
        "RedRatio": RedRatio, "dA": dA, "dB": dB, "dC": dC,
    }

    with st.spinner("Running Random Forest inference …"):
        fatigue, err = run_prediction(model, scaler, input_dict)

    if err:
        st.error(f"❌ Prediction error: {err}")
    else:
        props = derive_properties(fatigue)
        # Persist to session state
        for k, v in props.items():
            st.session_state[k] = v
        st.session_state.prediction_done = True
        st.session_state.last_inputs = input_dict.copy()

# ─────────────────────────────────────────────
# RESULTS PANEL
# ─────────────────────────────────────────────
if st.session_state.prediction_done:
    st.divider()
    st.markdown("## 📊 Prediction Results")

    fat  = st.session_state.fatigue
    ys   = st.session_state.yield_strength
    uts  = st.session_state.uts
    hrd  = st.session_state.hardness
    grade = st.session_state.steel_type
    badge = st.session_state.badge

    # ── Metrics row ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⚡ Fatigue Strength", f"{fat:.1f} MPa")
    m2.metric("🔩 Yield Strength",   f"{ys:.1f} MPa")
    m3.metric("💪 Ultimate Tensile", f"{uts:.1f} MPa")
    m4.metric("🔵 Brinell Hardness", f"{hrd:.1f} BHN")

    # ── Steel classification ──────────────────────────────────────────────
    badge_fn = getattr(st, badge, st.info)
    badge_fn(f"🏭  **Predicted Steel Grade:** {grade}")

    st.divider()

    # ── Stress-Strain Curve ───────────────────────────────────────────────
    chart_col, info_col = st.columns([2, 1], gap="large")

    with chart_col:
        st.markdown("### Stress–Strain Curve")

        E             = 200_000        # MPa, Young's modulus for steel
        strain_yield  = ys / E
        strain_total  = 0.30

        strain = np.linspace(0, strain_total, 800)
        stress = np.zeros_like(strain)

        for i, s in enumerate(strain):
            if s <= strain_yield:
                stress[i] = E * s
            elif s <= 0.15:
                t = (s - strain_yield) / (0.15 - strain_yield)
                stress[i] = ys + (uts - ys) * (t ** 0.45)
            else:
                stress[i] = uts * np.exp(-4.5 * (s - 0.15))

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#0d1117")
        ax.spines[:].set_color("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.title.set_color("#e6edf3")

        ax.plot(strain, stress, color="#1a73e8", linewidth=2.5, label="Stress–Strain")
        ax.axhline(ys,  color="#ff6d00", linestyle="--", linewidth=1.2, label=f"Yield  {ys:.0f} MPa")
        ax.axhline(uts, color="#00c6ff", linestyle="--", linewidth=1.2, label=f"UTS  {uts:.0f} MPa")
        ax.axhline(fat, color="#7c3aed", linestyle=":",  linewidth=1.2, label=f"Fatigue  {fat:.0f} MPa")

        ax.scatter([strain_yield], [ys],  color="#ff6d00", s=60, zorder=5)
        ax.scatter([0.15],        [uts], color="#00c6ff", s=60, zorder=5)

        ax.set_xlabel("Strain (ε)")
        ax.set_ylabel("Stress (σ, MPa)")
        ax.set_title("Predicted Stress–Strain Profile")
        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#e6edf3", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(color="#30363d", linewidth=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with info_col:
        st.markdown("### Engineering Summary")
        props_table = [
            ("Fatigue Limit",       f"{fat:.1f} MPa",  "Direct model output"),
            ("Yield Strength",      f"{ys:.1f} MPa",   "0.6 × Fatigue"),
            ("UTS",                 f"{uts:.1f} MPa",  "1.2 × Yield"),
            ("Brinell Hardness",    f"{hrd:.1f} BHN",  "0.18 × Fatigue"),
            ("Young's Modulus",     "200,000 MPa",     "Assumed (steel)"),
            ("Elastic Strain Limit", f"{strain_yield:.5f}", "YS / E"),
        ]
        for name, val, note in props_table:
            st.markdown(
                f"""<div style='background:#161b22;border:1px solid #30363d;
                    border-radius:10px;padding:10px 14px;margin-bottom:7px;'>
                  <span style='font-weight:600;font-size:.9rem;'>{name}</span><br>
                  <span style='color:#1a73e8;font-size:1.05rem;font-weight:700;'>{val}</span><br>
                  <span style='color:#8b949e;font-size:.78rem;'>{note}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("### Input Snapshot")
        snap_df = pd.DataFrame(
            [(k, round(v, 4)) for k, v in st.session_state.last_inputs.items()],
            columns=["Feature", "Value"],
        )
        st.dataframe(snap_df, use_container_width=True, height=320,
                     hide_index=True)

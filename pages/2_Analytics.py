"""
pages/2_Analytics.py — Model Analytics & Data Insights
"""

import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
from scipy import stats

from utils import load_assets, FEATURE_COLS, TARGET_COL

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MatPredict | Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SHARED STYLES
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
    .section-hdr{font-size:1.05rem;font-weight:700;color:var(--text);
        border-left:4px solid var(--primary);padding-left:10px;margin:14px 0 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ **MatPredict**")
    st.markdown("<span style='color:#8b949e;font-size:.85rem;'>Analytics</span>",
                unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",               label="🏠  Home")
    st.page_link("pages/1_Prediction.py",label="🔮  Predict")
    st.page_link("pages/2_Analytics.py", label="📊  Analytics")
    st.divider()

# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────
assets = load_assets()
if assets["error"]:
    st.error(f"❌ Asset loading error: {assets['error']}")
    st.stop()

model  = assets["model"]
scaler = assets["scaler"]
df_raw = assets["data"]

# Prepare X / y
df = df_raw.copy()
X = df[FEATURE_COLS]
y = df[TARGET_COL]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

residuals = y.values - y_pred

# ─────────────────────────────────────────────
# MATPLOTLIB THEME HELPER
# ─────────────────────────────────────────────
DARK = {
    "fig_bg":   "#0d1117",
    "ax_bg":    "#161b22",
    "spine":    "#30363d",
    "tick":     "#8b949e",
    "grid":     "#21262d",
    "text":     "#e6edf3",
    "blue":     "#1a73e8",
    "orange":   "#ff6d00",
    "cyan":     "#00c6ff",
    "green":    "#3fb950",
    "purple":   "#a371f7",
}

def _style_ax(ax):
    ax.set_facecolor(DARK["ax_bg"])
    for sp in ax.spines.values():
        sp.set_color(DARK["spine"])
    ax.tick_params(colors=DARK["tick"], labelsize=8)
    ax.xaxis.label.set_color(DARK["tick"])
    ax.yaxis.label.set_color(DARK["tick"])
    ax.title.set_color(DARK["text"])
    ax.title.set_fontsize(11)
    ax.grid(color=DARK["grid"], linewidth=0.6, linestyle="--")

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("## 📊 Model Analytics")
st.markdown(
    "<span style='color:#8b949e;'>Deep-dive into model performance, "
    "feature importance, and dataset distributions.</span>",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_fi, tab_res, tab_dist, tab_corr, tab_data = st.tabs([
    "🏆 Feature Importance",
    "📉 Residual Analysis",
    "📈 Distributions",
    "🔗 Correlations",
    "📋 Raw Data",
])

# ══════════════════════════════════════════════
# TAB 1 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════
with tab_fi:
    st.markdown("<div class='section-hdr'>Feature Importance (Mean Decrease in Impurity)</div>",
                unsafe_allow_html=True)

    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    col_chart, col_table = st.columns([2, 1], gap="large")

    with col_chart:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(DARK["fig_bg"])
        _style_ax(ax)

        colors = [DARK["blue"] if i < 3 else DARK["cyan"] if i < 8
                  else "#30363d" for i in range(len(fi_df))]
        bars = ax.barh(fi_df["Feature"], fi_df["Importance"],
                       color=colors, edgecolor="none", height=0.65)

        # Value labels
        for bar, val in zip(bars, fi_df["Importance"]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left",
                    color=DARK["tick"], fontsize=8)

        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title("Random Forest Feature Importances")
        ax.set_xlim(0, fi_df["Importance"].max() * 1.18)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_table:
        st.markdown("**Ranked Importance Table**")
        fi_display = fi_df.copy()
        fi_display.index = range(1, len(fi_df) + 1)
        fi_display["Importance"] = fi_display["Importance"].map("{:.4f}".format)
        fi_display["%"] = (fi_df["Importance"] * 100).map("{:.1f}%".format)
        st.dataframe(fi_display, use_container_width=True, height=520)

    # Cumulative importance
    st.markdown("<div class='section-hdr'>Cumulative Feature Importance</div>",
                unsafe_allow_html=True)
    cum_fig, cum_ax = plt.subplots(figsize=(9, 3))
    cum_fig.patch.set_facecolor(DARK["fig_bg"])
    _style_ax(cum_ax)

    cum_vals = fi_df["Importance"].cumsum().values
    cum_ax.fill_between(range(len(cum_vals)), cum_vals, alpha=0.3, color=DARK["blue"])
    cum_ax.plot(range(len(cum_vals)), cum_vals, color=DARK["blue"], linewidth=2)
    cum_ax.axhline(0.80, color=DARK["orange"], linestyle="--", linewidth=1,
                   label="80% threshold")
    cum_ax.axhline(0.95, color=DARK["cyan"],   linestyle="--", linewidth=1,
                   label="95% threshold")
    cum_ax.set_xticks(range(len(FEATURE_COLS)))
    cum_ax.set_xticklabels(fi_df["Feature"], rotation=45, ha="right", fontsize=8)
    cum_ax.set_ylabel("Cumulative Importance")
    cum_ax.set_title("Cumulative Feature Importance")
    cum_ax.legend(facecolor=DARK["ax_bg"], edgecolor=DARK["spine"],
                  labelcolor=DARK["text"], fontsize=8)
    cum_fig.tight_layout()
    st.pyplot(cum_fig)
    plt.close(cum_fig)

# ══════════════════════════════════════════════
# TAB 2 — RESIDUAL ANALYSIS
# ══════════════════════════════════════════════
with tab_res:
    st.markdown("<div class='section-hdr'>Residual Analysis</div>",
                unsafe_allow_html=True)

    # Compute metrics
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y.values - y.values.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot
    rmse   = np.sqrt(np.mean(residuals ** 2))
    mae    = np.mean(np.abs(residuals))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score",  f"{r2:.4f}")
    m2.metric("RMSE",      f"{rmse:.2f} MPa")
    m3.metric("MAE",       f"{mae:.2f} MPa")
    m4.metric("Max Error", f"{np.max(np.abs(residuals)):.2f} MPa")

    st.markdown("")

    fig_res = plt.figure(figsize=(14, 8))
    fig_res.patch.set_facecolor(DARK["fig_bg"])
    gs = gridspec.GridSpec(2, 2, figure=fig_res, hspace=0.45, wspace=0.35)

    # ── 1. Actual vs Predicted ───────────────────────────────────────────
    ax1 = fig_res.add_subplot(gs[0, 0])
    _style_ax(ax1)
    ax1.scatter(y.values, y_pred, alpha=0.55, s=22,
                color=DARK["blue"], edgecolors="none")
    lims = [min(y.min(), y_pred.min()) - 5, max(y.max(), y_pred.max()) + 5]
    ax1.plot(lims, lims, color=DARK["orange"], linewidth=1.5, linestyle="--",
             label="Perfect fit")
    ax1.set_xlabel("Actual Fatigue (MPa)")
    ax1.set_ylabel("Predicted Fatigue (MPa)")
    ax1.set_title("Actual vs. Predicted")
    ax1.legend(facecolor=DARK["ax_bg"], edgecolor=DARK["spine"],
               labelcolor=DARK["text"], fontsize=8)
    ax1.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax1.transAxes,
             color=DARK["cyan"], fontsize=9, fontweight="bold")

    # ── 2. Residuals vs Predicted ────────────────────────────────────────
    ax2 = fig_res.add_subplot(gs[0, 1])
    _style_ax(ax2)
    ax2.scatter(y_pred, residuals, alpha=0.55, s=22,
                color=DARK["cyan"], edgecolors="none")
    ax2.axhline(0, color=DARK["orange"], linewidth=1.5, linestyle="--")
    ax2.axhline( 2*rmse, color=DARK["purple"], linewidth=1, linestyle=":",
                 label="±2 RMSE")
    ax2.axhline(-2*rmse, color=DARK["purple"], linewidth=1, linestyle=":")
    ax2.set_xlabel("Predicted Fatigue (MPa)")
    ax2.set_ylabel("Residual (MPa)")
    ax2.set_title("Residuals vs. Predicted")
    ax2.legend(facecolor=DARK["ax_bg"], edgecolor=DARK["spine"],
               labelcolor=DARK["text"], fontsize=8)

    # ── 3. Residual Histogram ────────────────────────────────────────────
    ax3 = fig_res.add_subplot(gs[1, 0])
    _style_ax(ax3)
    ax3.hist(residuals, bins=35, color=DARK["blue"], alpha=0.75, edgecolor="none")
    mu, sigma_r = residuals.mean(), residuals.std()
    x_fit = np.linspace(residuals.min(), residuals.max(), 200)
    y_fit = stats.norm.pdf(x_fit, mu, sigma_r) * len(residuals) * (residuals.max()-residuals.min())/35
    ax3.plot(x_fit, y_fit, color=DARK["orange"], linewidth=2, label="Normal fit")
    ax3.axvline(0, color=DARK["cyan"], linewidth=1.5, linestyle="--")
    ax3.set_xlabel("Residual (MPa)")
    ax3.set_ylabel("Count")
    ax3.set_title("Residual Distribution")
    ax3.legend(facecolor=DARK["ax_bg"], edgecolor=DARK["spine"],
               labelcolor=DARK["text"], fontsize=8)

    # ── 4. Q-Q Plot ──────────────────────────────────────────────────────
    ax4 = fig_res.add_subplot(gs[1, 1])
    _style_ax(ax4)
    (osm, osr), (slope, intercept, r_qq) = stats.probplot(residuals, dist="norm")
    ax4.scatter(osm, osr, alpha=0.55, s=22, color=DARK["cyan"], edgecolors="none")
    x_line = np.array([min(osm), max(osm)])
    ax4.plot(x_line, slope * x_line + intercept,
             color=DARK["orange"], linewidth=1.8, linestyle="--", label="Normal line")
    ax4.set_xlabel("Theoretical Quantiles")
    ax4.set_ylabel("Sample Quantiles")
    ax4.set_title("Normal Q-Q Plot of Residuals")
    ax4.legend(facecolor=DARK["ax_bg"], edgecolor=DARK["spine"],
               labelcolor=DARK["text"], fontsize=8)

    st.pyplot(fig_res)
    plt.close(fig_res)

# ══════════════════════════════════════════════
# TAB 3 — DISTRIBUTIONS
# ══════════════════════════════════════════════
with tab_dist:
    st.markdown("<div class='section-hdr'>Feature Distributions</div>",
                unsafe_allow_html=True)

    selected_feats = st.multiselect(
        "Select features to visualise",
        options=FEATURE_COLS + [TARGET_COL],
        default=["C", "Mn", "Cr", "Ni", "TT", "NT", "Fatigue"],
    )

    if selected_feats:
        ncols = 3
        nrows = -(-len(selected_feats) // ncols)   # ceiling division
        fig_d, axes_d = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
        fig_d.patch.set_facecolor(DARK["fig_bg"])
        axes_flat = axes_d.flatten() if hasattr(axes_d, "flatten") else [axes_d]

        for i, feat in enumerate(selected_feats):
            ax = axes_flat[i]
            _style_ax(ax)
            data = df[feat].dropna()
            ax.hist(data, bins=30, color=DARK["blue"], alpha=0.75, edgecolor="none")
            ax.axvline(data.mean(), color=DARK["orange"], linewidth=1.5,
                       linestyle="--", label=f"μ={data.mean():.2f}")
            ax.set_title(feat)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.legend(facecolor=DARK["ax_bg"], edgecolor=DARK["spine"],
                      labelcolor=DARK["text"], fontsize=7)

        # Hide unused axes
        for j in range(len(selected_feats), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig_d.tight_layout()
        st.pyplot(fig_d)
        plt.close(fig_d)
    else:
        st.info("Select at least one feature above.")

    # Target distribution
    st.markdown("<div class='section-hdr'>Target Variable — Fatigue Strength</div>",
                unsafe_allow_html=True)
    fig_t, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(12, 4))
    fig_t.patch.set_facecolor(DARK["fig_bg"])

    for ax_ in (ax_h, ax_b):
        _style_ax(ax_)

    ax_h.hist(y, bins=30, color=DARK["blue"], alpha=0.75, edgecolor="none")
    ax_h.axvline(y.mean(), color=DARK["orange"], linewidth=2,
                 linestyle="--", label=f"Mean = {y.mean():.1f}")
    ax_h.axvline(y.median(), color=DARK["cyan"], linewidth=2,
                 linestyle=":", label=f"Median = {y.median():.1f}")
    ax_h.set_xlabel("Fatigue Strength (MPa)")
    ax_h.set_ylabel("Count")
    ax_h.set_title("Fatigue Strength Distribution")
    ax_h.legend(facecolor=DARK["ax_bg"], edgecolor=DARK["spine"],
                labelcolor=DARK["text"], fontsize=8)

    bp = ax_b.boxplot(y, patch_artist=True, vert=True,
                      boxprops=dict(facecolor=DARK["blue"]+"50", color=DARK["blue"]),
                      whiskerprops=dict(color=DARK["cyan"]),
                      capprops=dict(color=DARK["cyan"]),
                      medianprops=dict(color=DARK["orange"], linewidth=2),
                      flierprops=dict(marker="o", color=DARK["purple"],
                                      markerfacecolor=DARK["purple"], markersize=4))
    ax_b.set_ylabel("Fatigue Strength (MPa)")
    ax_b.set_title("Box Plot")
    ax_b.set_xticklabels(["Fatigue"])

    fig_t.tight_layout()
    st.pyplot(fig_t)
    plt.close(fig_t)

# ══════════════════════════════════════════════
# TAB 4 — CORRELATIONS
# ══════════════════════════════════════════════
with tab_corr:
    st.markdown("<div class='section-hdr'>Correlation Heatmap</div>",
                unsafe_allow_html=True)

    corr_cols = FEATURE_COLS + [TARGET_COL]
    corr_matrix = df[corr_cols].corr()

    fig_c, ax_c = plt.subplots(figsize=(14, 11))
    fig_c.patch.set_facecolor(DARK["fig_bg"])
    ax_c.set_facecolor(DARK["ax_bg"])
    ax_c.tick_params(colors=DARK["tick"], labelsize=7)

    im = ax_c.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_c.set_xticks(range(len(corr_cols)))
    ax_c.set_yticks(range(len(corr_cols)))
    ax_c.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax_c.set_yticklabels(corr_cols)

    # Annotate cells
    for ii in range(len(corr_cols)):
        for jj in range(len(corr_cols)):
            val = corr_matrix.values[ii, jj]
            color = "white" if abs(val) > 0.5 else DARK["tick"]
            ax_c.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                      color=color, fontsize=6)

    cbar = plt.colorbar(im, ax=ax_c, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=DARK["tick"], labelsize=8)
    cbar.set_label("Pearson r", color=DARK["tick"])
    ax_c.set_title("Feature Correlation Matrix (Pearson)", color=DARK["text"], fontsize=12)
    fig_c.tight_layout()
    st.pyplot(fig_c)
    plt.close(fig_c)

    # Top correlations with target
    st.markdown("<div class='section-hdr'>Top Correlations with Fatigue Strength</div>",
                unsafe_allow_html=True)
    target_corr = corr_matrix[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    tc_df = target_corr.reset_index()
    tc_df.columns = ["Feature", "|Correlation|"]
    tc_df["Direction"] = tc_df["Feature"].apply(
        lambda f: "Positive ↑" if corr_matrix.loc[f, TARGET_COL] >= 0 else "Negative ↓"
    )
    tc_df["|Correlation|"] = tc_df["|Correlation|"].map("{:.4f}".format)
    st.dataframe(tc_df, use_container_width=True, height=400, hide_index=True)

# ══════════════════════════════════════════════
# TAB 5 — RAW DATA
# ══════════════════════════════════════════════
with tab_data:
    st.markdown("<div class='section-hdr'>Dataset Preview</div>",
                unsafe_allow_html=True)

    c_filter, c_search = st.columns([2, 1])
    with c_filter:
        fatigue_range = st.slider(
            "Filter by Fatigue Strength range (MPa)",
            int(y.min()), int(y.max()),
            (int(y.min()), int(y.max())),
        )
    with c_search:
        search_col = st.selectbox("Sort by", [TARGET_COL] + FEATURE_COLS)

    df_display = df[
        (df[TARGET_COL] >= fatigue_range[0]) &
        (df[TARGET_COL] <= fatigue_range[1])
    ].sort_values(search_col, ascending=False).reset_index(drop=True)

    st.markdown(f"**Showing {len(df_display)} / {len(df)} rows**")
    st.dataframe(df_display, use_container_width=True, height=450)

    # Summary stats
    st.markdown("<div class='section-hdr'>Descriptive Statistics</div>",
                unsafe_allow_html=True)
    st.dataframe(df[FEATURE_COLS + [TARGET_COL]].describe().round(3),
                 use_container_width=True)

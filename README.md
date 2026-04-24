# ⚙️ MatPredict — Steel Fatigue Strength Predictor

A production-ready SaaS application built with Streamlit that predicts
**fatigue strength, yield strength, UTS, and Brinell hardness** for steel
specimens using a Random Forest model trained on 437 real-world specimens.

---

## 📁 Project Structure

```
matpredict/
├── app.py                  ← Landing page (Streamlit entry-point)
├── utils.py                ← Shared helpers: load_assets, validate, predict
├── pages/
│   ├── 1_Prediction.py     ← Interactive prediction dashboard
│   └── 2_Analytics.py      ← Feature importance, residuals, distributions
├── assets/
│   ├── model.pkl           ← Trained RandomForestRegressor
│   ├── scaler.pkl          ← StandardScaler
│   └── material_data.csv   ← Training dataset (437 specimens)
├── .streamlit/
│   └── config.toml         ← Dark theme & server settings
├── requirements.txt        ← Pinned dependencies
└── runtime.txt             ← Python version for Streamlit Cloud
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone / unzip the project
cd matpredict

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this folder to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set:
   - **Repository**: your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy** — Streamlit Cloud reads `requirements.txt` and
   `runtime.txt` automatically.

---

## 🐳 Deploy on Render (Docker-free)

1. Create a new **Web Service** on [render.com](https://render.com).
2. Connect your GitHub repo.
3. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add Environment Variable: `PYTHON_VERSION = 3.11.0`

---

## 🧠 Model Details

| Property            | Value                          |
|---------------------|--------------------------------|
| Algorithm           | Random Forest Regressor        |
| Estimators          | 100 trees                      |
| Training samples    | 437 steel specimens            |
| Features            | 25 (chemical + processing)     |
| Target              | Fatigue Strength (MPa)         |
| Scaler              | StandardScaler                 |
| Trained with        | scikit-learn 1.6.1             |

---

## 📊 Pages

### 🏠 Home
Overview of the platform, predicted properties, and feature descriptions.

### 🔮 Prediction Dashboard
- Sliders for all 25 input features grouped as:
  - Chemical composition (%C, %Mn, %Si, %Ni, %Cr, %Mo, etc.)
  - Heat treatment stages (NT, THT, CT, TT, DT, etc.)
  - Defect indicators (dA, dB, dC)
- Instant prediction on click
- Metrics: Fatigue, Yield Strength, UTS, Hardness
- Steel grade classification
- Live stress–strain curve
- Input snapshot table (persists via `st.session_state`)

### 📊 Analytics
- **Feature Importance**: bar chart + cumulative curve
- **Residual Analysis**: actual vs predicted, residuals vs predicted,
  histogram with normal fit, Q-Q plot
- **Distributions**: interactive multi-select histogram panel
- **Correlations**: full Pearson heatmap + target correlation ranking
- **Raw Data**: filterable / sortable dataset viewer + descriptive stats

---

## 🔧 Adding New Models

Replace `assets/model.pkl` and `assets/scaler.pkl` with new files.
`load_assets()` in `utils.py` auto-discovers them — no code changes needed
as long as the 25 feature names remain the same.

---

## 📄 License
MIT — free for personal and commercial use.

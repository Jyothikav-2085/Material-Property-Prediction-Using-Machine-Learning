"""
utils.py — shared helpers for MatPredict SaaS
  · load_assets()  — dynamic model / scaler / data discovery
  · classify_steel()
  · validate_inputs()
  · derive_properties()
"""

from __future__ import annotations

import os
import pathlib
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "NT", "THT", "THt", "THQCr",
    "CT", "Ct", "DT", "Dt", "QmT",
    "TT", "Tt", "TCr",
    "C", "Si", "Mn", "P", "S",
    "Ni", "Cr", "Cu", "Mo",
    "RedRatio", "dA", "dB", "dC",
]

TARGET_COL = "Fatigue"

# ──────────────────────────────────────────────
# DYNAMIC ASSET DISCOVERY
# ──────────────────────────────────────────────
def _find_file(root: pathlib.Path, *names: str) -> Optional[pathlib.Path]:
    """
    Walk *root* recursively looking for any file whose name (case-insensitive)
    matches one of *names*.  Returns the first match found, or None.
    """
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.lower() in [n.lower() for n in names]:
                return pathlib.Path(dirpath) / fname
    return None


@st.cache_resource(show_spinner="Loading ML assets …")
def load_assets() -> dict:
    """
    Automatically locate and load:
      • model    (.pkl / .joblib)
      • scaler   (.pkl / .joblib)
      • dataset  (.csv)

    Search order:
      1. assets/ folder next to this file (production bundle)
      2. Two levels up from this file (development / ZIP-extracted tree)

    Returns a dict with keys: model, scaler, data, error (str | None)
    """
    result: dict = {"model": None, "scaler": None, "data": None, "error": None}

    # Candidate roots to search
    this_dir = pathlib.Path(__file__).parent.resolve()
    search_roots = [
        this_dir / "assets",
        this_dir,
        this_dir.parent,
        this_dir.parent.parent,
    ]

    # ── locate model ──────────────────────────────────────────────────────
    model_path: Optional[pathlib.Path] = None
    for root in search_roots:
        if root.exists():
            model_path = _find_file(root, "model.pkl", "model.joblib")
            if model_path:
                break

    # ── locate scaler ─────────────────────────────────────────────────────
    scaler_path: Optional[pathlib.Path] = None
    for root in search_roots:
        if root.exists():
            scaler_path = _find_file(root, "scaler.pkl", "scaler.joblib",
                                     "encoder.pkl", "encoder.joblib")
            if scaler_path and scaler_path != model_path:
                break

    # ── locate CSV ────────────────────────────────────────────────────────
    csv_path: Optional[pathlib.Path] = None
    for root in search_roots:
        if root.exists():
            csv_path = _find_file(root, "material_data.csv",
                                  "data.csv", "dataset.csv")
            if csv_path:
                break

    # ── load model ────────────────────────────────────────────────────────
    try:
        if model_path is None:
            raise FileNotFoundError("model.pkl not found in search paths.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result["model"] = joblib.load(model_path)
    except Exception as exc:
        result["error"] = f"Model load failed: {exc}"
        return result

    # ── load scaler ───────────────────────────────────────────────────────
    try:
        if scaler_path is None:
            raise FileNotFoundError("scaler.pkl not found in search paths.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result["scaler"] = joblib.load(scaler_path)
    except Exception as exc:
        result["error"] = f"Scaler load failed: {exc}"
        return result

    # ── load data ─────────────────────────────────────────────────────────
    try:
        if csv_path is None:
            raise FileNotFoundError("material_data.csv not found in search paths.")
        df = pd.read_csv(csv_path)
        # Drop non-feature columns if present
        drop_cols = [c for c in ["Sl. No.", "Sl.No.", "index"] if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")
        result["data"] = df
    except Exception as exc:
        result["error"] = f"Data load failed: {exc}"

    return result


# ──────────────────────────────────────────────
# CLASSIFICATION
# ──────────────────────────────────────────────
def classify_steel(hardness: float) -> tuple[str, str]:
    """
    Return (steel_grade_label, css_color) based on Brinell hardness.
    """
    if hardness < 120:
        return "Mild / Low-Carbon Steel", "success"
    elif hardness < 200:
        return "Medium Carbon Steel", "info"
    elif hardness < 300:
        return "High Carbon Steel", "warning"
    else:
        return "Alloy / Tool Steel", "error"


# ──────────────────────────────────────────────
# DERIVED PROPERTIES
# ──────────────────────────────────────────────
def derive_properties(fatigue: float) -> dict:
    yield_s   = round(0.6  * fatigue, 2)
    uts       = round(1.2  * yield_s,  2)
    hardness  = round(0.18 * fatigue,  2)
    grade, badge = classify_steel(hardness)
    return {
        "fatigue":       round(fatigue, 2),
        "yield_strength": yield_s,
        "uts":           uts,
        "hardness":      hardness,
        "steel_type":    grade,
        "badge":         badge,
    }


# ──────────────────────────────────────────────
# INPUT VALIDATION
# ──────────────────────────────────────────────
def validate_inputs(input_dict: dict) -> list[str]:
    """
    Compare the keys in input_dict against FEATURE_COLS.
    Return a list of error strings; empty list means OK.
    """
    errors: list[str] = []
    missing = [f for f in FEATURE_COLS if f not in input_dict]
    if missing:
        errors.append(f"Missing features: {', '.join(missing)}")
    for f, v in input_dict.items():
        if f in FEATURE_COLS:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                errors.append(f"Feature '{f}' must be numeric (got {type(v).__name__}).")
                continue
            if np.isnan(fv) or np.isinf(fv):
                errors.append(f"Feature '{f}' has invalid value: {v}.")

    return errors


# ──────────────────────────────────────────────
# PREDICTION HELPER
# ──────────────────────────────────────────────
def run_prediction(model, scaler, input_dict: dict) -> tuple[Optional[float], Optional[str]]:
    """
    Run scaler → model.predict.
    Returns (fatigue_value, None) on success, or (None, error_message).
    """
    errors = validate_inputs(input_dict)
    if errors:
        return None, "; ".join(errors)

    try:
        df_in = pd.DataFrame([{k: input_dict[k] for k in FEATURE_COLS}])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled = scaler.transform(df_in)
            pred = model.predict(X_scaled)[0]
        return float(pred), None
    except Exception as exc:
        return None, str(exc)

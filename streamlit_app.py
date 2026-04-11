# streamlit_app.py — Streamlit Box Office Predictor
# Run: streamlit run streamlit_app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import date

st.set_page_config(
    page_title="🎬 Box Office Predictor",
    page_icon="🎬",
    layout="wide"
)

# ─── Load Artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    pipeline     = joblib.load("box_office_full_pipeline.pkl")
    director_map = joblib.load("director_target_encoding.pkl")
    genre_mlb    = joblib.load("genre_mlb.pkl")
    return pipeline, director_map, genre_mlb

pipeline, director_map, mlb = load_artifacts()

CURRENT_YEAR = date.today().year

# FIX: Use mlb.classes_ directly — these are the exact strings the model was
# trained on (Title Case, e.g. "Science Fiction"). Never hardcode or sort them.
ALL_GENRES = list(mlb.classes_)

# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🎬 Box Office Revenue Predictor")
st.markdown("""
Predict how much a movie will earn based on its key attributes.  
Powered by a Stacking ensemble of XGBoost, LightGBM & RandomForest models.
""")
st.divider()

# ─── Input Form ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Movie Details")
    budget       = st.number_input("Budget ($)", min_value=0, max_value=500_000_000,
                                   value=50_000_000, step=1_000_000)
    runtime      = st.slider("Runtime (minutes)", 60, 240, 120)
    vote_average = st.slider("Expected Audience Rating (1–10)", 1.0, 10.0, 6.5, 0.1)
    vote_count   = st.number_input("Expected Vote Count", 100, 100_000, 1000)
    popularity   = st.number_input("TMDB Popularity Score", 0.0, 2000.0, 100.0)
    release_date = st.date_input("Release Date", value=date(2025, 7, 4))

with col2:
    st.subheader("🎭 Metadata")
    # FIX: default genres must be valid entries from ALL_GENRES (exact match)
    default_genres = [g for g in ["Action", "Adventure"] if g in ALL_GENRES]
    genres        = st.multiselect("Genres", ALL_GENRES, default=default_genres)
    director      = st.text_input("Director", value="Christopher Nolan")
    language      = st.selectbox("Original Language",
                                  ["en", "fr", "ja", "ko", "es", "other"], index=0)
    is_us         = st.checkbox("US Production?", value=True)
    num_companies = st.slider("# Production Companies", 1, 10, 2)
    num_cast      = st.slider("# Main Cast Members", 1, 20, 5)

st.divider()

# ─── Prediction ──────────────────────────────────────────────────────────────
if st.button("🔮 Predict Box Office Revenue", type="primary", use_container_width=True):

    if not genres:
        st.warning("⚠️ Please select at least one genre.")
        st.stop()

    movie_input = {
        "budget":               budget,
        "runtime":              runtime,
        "genres":               genres,           # exact Title Case from mlb.classes_
        "vote_average":         vote_average,
        "vote_count":           int(vote_count),
        "popularity":           popularity,
        "release_date":         str(release_date),
        "director":             director.strip(),
        "original_language":    language,
        "production_countries": ["United States of America"] if is_us else ["United Kingdom"],
        "production_companies": [f"Studio {i}" for i in range(num_companies)],
        "cast":                 [f"Actor {i}" for i in range(num_cast)]
    }

    try:
        from notebook_inference import predict_new_movie
        result = predict_new_movie(movie_input, pipeline)
        rev    = result["predicted_revenue"]

        st.success(f"### 💰 Predicted Box Office Revenue: **${rev / 1e6:.1f}M**")

        col3, col4, col5 = st.columns(3)
        col3.metric("Predicted Revenue", f"${rev / 1e6:.1f}M")

        roi = (rev - budget) / max(budget, 1) * 100
        col4.metric("Estimated ROI", f"{roi:.0f}%", delta=f"{roi:.0f}%")

        col5.metric("Budget", f"${budget / 1e6:.1f}M")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)   # shows full traceback in the app for easier debugging
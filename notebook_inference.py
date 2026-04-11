import pandas as pd
import numpy as np
import joblib
from datetime import date

# ─────────────────────────────────────────────────────────────
# Load artifacts (same as training)
# ─────────────────────────────────────────────────────────────
mlb = joblib.load("genre_mlb.pkl")
director_rev_map = joblib.load("director_target_encoding.pkl")  # pandas Series

CURRENT_YEAR = date.today().year

# Create genre column names — must match training exactly
GENRE_COLS = [f"genre_{g.lower().replace(' ', '_')}" for g in mlb.classes_]


# ─────────────────────────────────────────────────────────────
# Preprocessing function
# ─────────────────────────────────────────────────────────────
def preprocess_new_movie(movie_dict: dict) -> pd.DataFrame:

    m = movie_dict.copy()
    row = {}

    # ── Numeric ───────────────────────────────────────────────
    budget     = float(m.get('budget', 0) or 0)
    popularity = float(m.get('popularity', 50) or 50)
    vote_count = float(m.get('vote_count', 100) or 100)
    vote_avg   = float(m.get('vote_average', 6.0) or 6.0)

    row['log_budget']              = np.log1p(budget)
    row['runtime']                 = float(m.get('runtime', 90) or 90)
    row['vote_average']            = vote_avg
    row['log_vote_count']          = np.log1p(vote_count)
    row['log_popularity']          = np.log1p(popularity)
    row['vote_engagement']         = vote_count * vote_avg
    row['log_budget_x_popularity'] = np.log1p(budget * popularity)
    row['popularity_per_budget']   = popularity / (budget + 1)

    # ── Date ─────────────────────────────────────────────────
    rd = pd.to_datetime(m.get('release_date', '2024-06-15'), errors='coerce')
    if pd.isna(rd):
        rd = pd.Timestamp('2024-06-15')

    row['release_year']          = rd.year
    row['release_month']         = rd.month
    row['release_day']           = rd.day
    row['release_dow']           = rd.dayofweek
    row['release_quarter']       = rd.quarter
    row['movie_age']             = CURRENT_YEAR - rd.year
    row['is_weekend_release']    = int(rd.dayofweek in [4, 5, 6])
    row['is_summer_blockbuster'] = int(rd.month in [5, 6, 7, 8])
    row['is_holiday_release']    = int(rd.month in [11, 12])

    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring',  4: 'spring', 5: 'spring',
        6: 'summer',  7: 'summer', 8: 'summer'
    }
    row['season'] = season_map.get(rd.month, 'fall')

    # ── List features ─────────────────────────────────────────
    genres        = m.get('genres', [])
    cast          = m.get('cast', [])
    prod_countries = m.get('production_countries', [])
    prod_companies = m.get('production_companies', [])

    row['num_genres']               = len(genres)
    row['num_cast']                 = len(cast)
    row['num_production_countries'] = len(prod_countries)
    row['num_production_companies'] = len(prod_companies)
    row['is_us_production']         = int('United States of America' in prod_countries)

    # ── Genre encoding ────────────────────────────────────────
    # Genres must match mlb.classes_ exactly (Title Case e.g. "Science Fiction")
    genre_vec = mlb.transform([genres])[0]
    for col, val in zip(GENRE_COLS, genre_vec):
        row[col] = int(val)

    # ── Language ─────────────────────────────────────────────
    lang      = m.get('original_language', 'en')
    top_langs = ['en', 'fr', 'ja', 'ko', 'es']
    row['language_group'] = lang if lang in top_langs else 'other'

    # ── Director encoding ─────────────────────────────────────
    # FIX: director_rev_map is a pandas Series; use .get() for safe lookup
    # and .mean() for the fallback (not list() which gives index keys on dict)
    director = m.get('director', 'unknown')
    fallback = float(director_rev_map.mean())
    row['director_mean_log_revenue'] = float(
        director_rev_map.get(director, fallback)
    )

    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────────────────────────
def predict_new_movie(movie_dict: dict, pipeline) -> dict:

    X_new = preprocess_new_movie(movie_dict)

    # Align features with what the pipeline was trained on
    expected_features = pipeline.feature_names_in_

    for col in expected_features:
        if col not in X_new.columns:
            X_new[col] = 0

    X_new = X_new[expected_features]

    log_pred = pipeline.predict(X_new)[0]
    revenue  = np.expm1(log_pred)

    return {
        "predicted_revenue":           float(revenue),
        "predicted_revenue_formatted": f"${revenue / 1e6:.1f}M",
        "log_revenue_predicted":       float(log_pred)
    }
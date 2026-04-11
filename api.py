
# api.py — FastAPI Box Office Prediction Endpoint
# Run: uvicorn api:app --reload
# Docs: http://127.0.0.1:8000/docs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import uvicorn

app = FastAPI(
    title="🎬 Box Office Revenue Predictor API",
    description="Predict box office revenue using a production ML pipeline.",
    version="1.0.0"
)

# ─── Load Model at Startup ───────────────────────────────────────────────────
pipeline     = joblib.load("box_office_full_pipeline.pkl")
director_map = joblib.load("director_target_encoding.pkl")
genre_mlb    = joblib.load("genre_mlb.pkl")

# ─── Request / Response Schemas ──────────────────────────────────────────────
class MovieInput(BaseModel):
    budget:               float       = Field(..., ge=0,   example=100_000_000)
    runtime:              float       = Field(90, ge=1,    example=120)
    genres:               List[str]   = Field(...,         example=["Action", "Adventure"])
    vote_average:         float       = Field(6.5, ge=1, le=10, example=7.2)
    vote_count:           int         = Field(500, ge=0,  example=2000)
    popularity:           float       = Field(50.0, ge=0, example=150.0)
    release_date:         str         = Field(...,         example="2025-07-04")
    director:             str         = Field("unknown",  example="Christopher Nolan")
    original_language:    str         = Field("en",       example="en")
    production_countries: List[str]   = Field([], example=["United States of America"])
    production_companies: List[str]   = Field([], example=["Warner Bros. Pictures"])
    cast:                 List[str]   = Field([], example=["Actor A", "Actor B"])

class PredictionResponse(BaseModel):
    predicted_revenue:           float
    predicted_revenue_formatted: str
    log_revenue_predicted:       float
    model_version:               str = "1.0.0"

# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Box Office Predictor API v1.0.0"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(movie: MovieInput):
    try:
        from notebook_inference import predict_new_movie
        result = predict_new_movie(movie.dict(), pipeline)
        result["model_version"] = "1.0.0"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(movies: List[MovieInput]):
    """Predict revenue for multiple movies in one call."""
    from notebook_inference import predict_new_movie
    results = [predict_new_movie(m.dict(), pipeline) for m in movies]
    return {"predictions": results, "count": len(results)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

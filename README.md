# 🎬 Movie Revenue Predictor

An end-to-end Machine Learning project that predicts a movie's box office revenue using metadata such as budget, genres, popularity, director, runtime, ratings, and production details.

The project includes:

* ✅ ML training pipeline
* ✅ FastAPI backend API
* ✅ Streamlit web application
* ✅ Production-ready inference pipeline
* ✅ Serialized trained models using Joblib

---

# 🚀 Features

* Predict estimated box office revenue for upcoming movies
* Interactive Streamlit frontend
* REST API using FastAPI
* Uses ensemble learning models
* Handles categorical encoding and multi-label genres
* Batch prediction support
* Production inference pipeline

---

# 🧠 ML Models Used

The final model uses a **Stacking Ensemble** of:

* XGBoost
* LightGBM
* Random Forest Regressor

These models are combined to improve prediction accuracy and reduce overfitting.

---

# 📂 Project Structure

```bash
Movie_Revenue_Predictor/
│
├── api.py                              # FastAPI backend
├── streamlit_app.py                    # Streamlit frontend
├── notebook_inference.py               # Prediction pipeline
│
├── box_office_full_pipeline.pkl        # Trained ML pipeline
├── director_target_encoding.pkl        # Director encoding map
├── genre_mlb.pkl                       # MultiLabelBinarizer for genres
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# 📊 Input Features

The model predicts revenue using:

* Budget
* Runtime
* Genres
* Vote Average
* Vote Count
* Popularity
* Release Date
* Director
* Original Language
* Production Countries
* Production Companies
* Cast Information

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Movie_Revenue_Predictor.git

cd Movie_Revenue_Predictor
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser automatically.

---

# 🌐 Run FastAPI Server

```bash
uvicorn api:app --reload
```

API Docs available at:

```bash
http://127.0.0.1:8000/docs
```

---

# 📡 API Endpoints

## Health Check

```http
GET /
```

---

## Predict Single Movie

```http
POST /predict
```

### Example Request

```json
{
  "budget": 100000000,
  "runtime": 120,
  "genres": ["Action", "Adventure"],
  "vote_average": 7.5,
  "vote_count": 5000,
  "popularity": 150.0,
  "release_date": "2025-07-04",
  "director": "Christopher Nolan",
  "original_language": "en",
  "production_countries": ["United States of America"],
  "production_companies": ["Warner Bros"],
  "cast": ["Actor A", "Actor B"]
}
```

---

## Batch Prediction

```http
POST /predict/batch
```

Allows prediction for multiple movies in one request.

---

# 🖥️ Streamlit UI

The frontend allows users to:

* Enter movie details
* Select genres
* Choose language
* Estimate ROI
* View predicted revenue instantly

---

# 🔍 Machine Learning Pipeline

The project includes:

* Feature Engineering
* MultiLabel Genre Encoding
* Director Target Encoding
* Numerical Scaling
* Ensemble Regression Models
* Log Revenue Transformation

---

# 📈 Future Improvements

* Add TMDB API integration
* Deploy on AWS/GCP
* Add real-time movie data fetching
* Improve feature engineering
* Add poster/image analysis using Deep Learning

---

# 🛠️ Tech Stack

* Python
* Scikit-learn
* XGBoost
* LightGBM
* FastAPI
* Streamlit
* Pandas
* NumPy
* Joblib

---

# 📌 Use Cases

* Movie revenue forecasting
* Entertainment analytics
* Production budget planning
* Data science portfolio project
* ML deployment demonstration

---

# 👨‍💻 Author

Paras Gupta

If you found this project useful, consider giving it a ⭐ on GitHub.

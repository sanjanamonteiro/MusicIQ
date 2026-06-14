# 🎵 MusicIQ — Full Stack ML Music Intelligence App

<div align="center">

![MusicIQ Banner](https://img.shields.io/badge/MusicIQ-Full%20Stack%20ML%20App-6c63ff?style=for-the-badge&logo=spotify&logoColor=white)

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-GitHub%20Pages-00d4aa?style=for-the-badge)](https://sanjanamonteiro.github.io/MusicIQ/)
[![API](https://img.shields.io/badge/⚙️%20Live%20API-Render.com-6c63ff?style=for-the-badge)](https://musiciq-api.onrender.com)
[![Notebook](https://img.shields.io/badge/📓%20Analysis-Colab%20Notebook-ff6584?style=for-the-badge)](https://github.com/sanjanamonteiro/MusicIQ/blob/main/MusicIQ_Analysis.ipynb)

> **Analyze. Predict. Discover.** — A data-driven music intelligence platform built on 114,000 Spotify tracks using machine learning, FastAPI, and Firebase.

</div>

---

## 📌 Table of Contents

- [Live Demo](#-live-demo)
- [What is MusicIQ?](#-what-is-musiciq)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Machine Learning Journey](#-machine-learning-journey)
- [Data Science Pipeline](#-data-science-pipeline)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [How to Run Locally](#-how-to-run-locally)
- [Deployment](#-deployment)
- [Key Learnings](#-key-learnings)
- [Dataset](#-dataset)

---

## 🌐 Live Demo

| Resource | Link |
|---|---|
| 🌐 Live Web App | https://sanjanamonteiro.github.io/MusicIQ/ |
| ⚙️ REST API | https://musiciq-api.onrender.com |
| 📓 Full Analysis Notebook | View on GitHub |
| 💻 Source Code | https://github.com/sanjanamonteiro/MusicIQ |

---

## 🎯 What is MusicIQ?

MusicIQ is a **full-stack machine learning web application** that answers one of music's most interesting questions:

> *"What actually makes a song popular?"*

Starting from raw Spotify data, I built an end-to-end pipeline covering data cleaning, exploratory analysis, machine learning, a REST API, and a live web app — all from scratch.

The project covers the **complete data science to production lifecycle:**

```
Raw Data → EDA → Feature Engineering → ML Model → REST API → Web App → Deployed Live
```

---

## ✨ Features

### 🎯 Hit Predictor
Enter any combination of audio features and get an instant ML-powered prediction on whether a song has the characteristics of a chart hit — complete with confidence score and hit probability.

### 🎧 Song Recommender
Type any song name and get the most similar songs based on **cosine similarity** across 10 audio dimensions. Live autocomplete suggestions as you type.

### 🔍 Song Search
Search and browse all 114,000 songs in the dataset. Click any result to instantly load it into the recommender.

### 🕘 Activity History
Automatically tracks all your predictions and recommendations in the current session.

### 🤖 AI Music Chatbot
Ask anything about music, audio features, or genres — powered by Claude AI.

### 🔐 Google Authentication
Secure login with Firebase Google Auth. Auth guard protects the dashboard from unauthenticated access.

---

## 🛠️ Tech Stack

### Frontend
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=flat&logo=firebase&logoColor=black)

### Backend
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

### Data Science
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat)

### Deployment
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=black)
![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-222222?style=flat&logo=github&logoColor=white)
![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=flat&logo=googledrive&logoColor=white)

---

## 🧠 Machine Learning Journey

This wasn't a straight line — here's the full honest story of how the ML was built and improved:

### Phase 1 — Baseline Model
Started with **Linear Regression** to predict popularity score directly.
```
R² = 0.028  ← only explains 2.8% of variance
```
**Problem:** Popularity depends on marketing, artist fame, playlist placement — not just audio. Linear regression was too simple.

### Phase 2 — Random Forest
Switched to **Random Forest Regressor** (100 trees, max depth 10).
```
R² = 0.129  ← 4.5x improvement
MAE = 15.34, RMSE = 19.08
```
Better but still limited — the relationship isn't purely linear.

### Phase 3 — Adding Genre Feature
Genre was being completely ignored. Added it via Label Encoding.
```
R² = 0.298  ← doubled again just by adding genre!
```
**Key insight:** Genre is the single most important predictor of popularity.

### Phase 4 — Rethinking the Problem
Switched from regression (predict exact score) to **binary classification** (Hit vs Not a Hit).

```python
threshold = df['popularity'].quantile(0.75)  # top 25% = "popular"
```

### Phase 5 — Solving Class Imbalance
Discovered the model was cheating — predicting "not popular" for everything:
```
Popular song recall = 0.04  ← finding only 4% of actual hits!
```

Fixed with **SMOTE** (Synthetic Minority Oversampling):
```
Before SMOTE: Not Popular 67,220 | Popular 22,520
After SMOTE:  Perfectly balanced 50/50
```

### Final Model Results
```
Model         : Random Forest Classifier
Trees         : 200
Max Depth     : 15
Class Weight  : Balanced
Accuracy      : 70%
Popular Recall: 0.67  ← finds 67% of actual hits (up from 4%!)
Popular F1    : 0.52
```

### Song Recommender
Built using **Cosine Similarity** on 10 audio features:
```python
similarities = cosine_similarity(input_vector, rec_scaled)[0]
```
Cosine similarity measures the angle between audio feature vectors — songs pointing in the same direction sound similar regardless of volume or intensity differences.

---

## 📊 Data Science Pipeline

```
1. DATA LOADING & CLEANING
   ├── Loaded 114,000 tracks from Kaggle Spotify dataset
   ├── Removed duplicates (by track_id)
   ├── Dropped missing values
   └── Final clean dataset: ~89,000 tracks

2. EXPLORATORY DATA ANALYSIS
   ├── Popularity distribution (histogram + boxplot)
   ├── Correlation heatmap across all audio features
   ├── Scatter plots for top correlated features
   ├── Genre song count comparison
   └── Genre audio feature heatmap (normalized)

3. FEATURE ENGINEERING
   ├── Label encoded track_genre → genre_encoded
   ├── Created binary target: is_popular (top 25%)
   └── StandardScaler normalization for ML

4. CLUSTERING (Unsupervised ML)
   ├── KMeans with Elbow Method (k=5)
   ├── PCA dimensionality reduction (7D → 2D)
   ├── Cluster visualization
   └── Cluster profile analysis

5. CLASSIFICATION (Supervised ML)
   ├── Train/Test split (80/20)
   ├── Linear Regression baseline
   ├── Random Forest Regressor
   ├── SMOTE class balancing
   ├── Random Forest Classifier (final model)
   └── Confusion matrix & classification report

6. RECOMMENDATION ENGINE
   ├── StandardScaler on audio features
   ├── Cosine similarity computation
   └── Top-N similar songs ranked by similarity + popularity
```

---

## ⚙️ API Endpoints

Base URL: `https://musiciq-api.onrender.com`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict if a song is a hit |
| `POST` | `/recommend` | Get similar song recommendations |
| `GET` | `/search?query=` | Search songs by name |
| `GET` | `/genres` | Get all available genres |

### Example — Hit Prediction Request
```json
POST /predict
{
  "danceability": 0.8,
  "energy": 0.9,
  "loudness": -4.0,
  "speechiness": 0.05,
  "acousticness": 0.1,
  "instrumentalness": 0.0,
  "liveness": 0.1,
  "valence": 0.7,
  "tempo": 120.0,
  "duration_ms": 200000,
  "genre_encoded": 15
}
```

### Example — Hit Prediction Response
```json
{
  "prediction": "Hit",
  "confidence": 82.3,
  "hit_probability": 82.3
}
```

---

## 📁 Project Structure

```
MusicIQ/
│
├── 📓 MusicIQ_Analysis.ipynb     # Full data science notebook (EDA → ML)
├── 📄 README.md
├── 🔒 .gitignore
│
├── backend/
│   ├── main.py                   # FastAPI server with all endpoints
│   ├── requirements.txt          # Python dependencies
│   ├── scaler.pkl                # Trained StandardScaler
│   └── rec_scaler.pkl            # Recommender scaler
│   # Note: hit_predictor.pkl & rec_data.csv loaded from Google Drive
│
└── frontend/
    ├── index.html                # Landing page + Google Auth
    ├── dashboard.html            # Main app (5 tabs)
    └── style.css                 # Dark theme UI
```

---

## 💻 How to Run Locally

### Prerequisites
- Python 3.9+
- Git
- VS Code with Live Server extension

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/sanjanamonteiro/MusicIQ.git
cd MusicIQ/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your model files to backend/ folder
# (hit_predictor.pkl, scaler.pkl, rec_scaler.pkl, rec_data.csv)

# Run the server
uvicorn main:app --reload
# API runs at http://127.0.0.1:8000
```

### Frontend Setup
```bash
# Open frontend/index.html with Live Server in VS Code
# Right click index.html → Open with Live Server
# App runs at http://127.0.0.1:5500
```

---

## 🚀 Deployment

### Backend → Render.com
- Connected GitHub repo to Render
- Set root directory to `backend/`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port 10000`
- Large model files (89MB) loaded at startup from Google Drive using `gdown`

### Frontend → GitHub Pages
- Pushed HTML/CSS/JS files to root of main branch
- Enabled GitHub Pages in repository settings
- Added GitHub Pages domain to Firebase authorized domains

### Authentication → Firebase
- Enabled Google Sign-In provider
- Added authorized domains: `localhost`, `127.0.0.1`, `sanjanamonteiro.github.io`

---

## 💡 Key Learnings

| Challenge | What I Learned |
|---|---|
| R² of 0.03 on first model | Popularity isn't just about sound — context matters |
| Model predicting "not popular" always | Class imbalance is a real problem in ML |
| SMOTE fixing recall from 4% → 67% | Synthetic oversampling is powerful |
| Genre doubling R² | Feature selection is as important as model choice |
| 89MB model too big for GitHub | Production ML needs smart file management |
| CORS errors on API | Backend must explicitly allow frontend origins |
| Firebase unauthorized-domain error | Every domain must be whitelisted including 127.0.0.1 |

---

## 📦 Dataset

**Spotify Tracks Dataset** by Maharshi Pandya on Kaggle

- 📊 **114,000 tracks** across **114 genres**
- 🎵 **21 features** including audio analysis and metadata
- 📅 Data collected via Spotify Web API

| Feature | Description |
|---|---|
| `danceability` | How suitable for dancing (0–1) |
| `energy` | Intensity and activity (0–1) |
| `loudness` | Overall loudness in dB |
| `speechiness` | Presence of spoken words (0–1) |
| `acousticness` | Confidence of being acoustic (0–1) |
| `instrumentalness` | Predicts no vocals (0–1) |
| `liveness` | Presence of live audience (0–1) |
| `valence` | Musical positiveness (0–1) |
| `tempo` | Estimated tempo in BPM |
| `popularity` | Spotify popularity score (0–100) |

---

## 👨‍💻 Author

Built from scratch as a complete end-to-end data science and full-stack development project — from raw CSV to live deployed web application.

---

<div align="center">

**⭐ If you found this project interesting, please give it a star!**

[![GitHub stars](https://img.shields.io/github/stars/sanjanamonteiro/MusicIQ?style=social)](https://github.com/sanjanamonteiro/MusicIQ)

</div>

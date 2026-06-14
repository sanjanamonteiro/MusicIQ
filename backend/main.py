from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Initialize App ─────────────────────────────────────────────────────
app = FastAPI(title="MusicIQ API", version="1.0")

# Allow frontend to talk to backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models Once at Startup ────────────────────────────────────────
print("Loading models...")

with open("hit_predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("rec_scaler.pkl", "rb") as f:
    rec_scaler = pickle.load(f)

rec_df = pd.read_csv("rec_data.csv")

# Pre-compute scaled features for recommender
REC_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo', 'duration_ms']

rec_scaled = rec_scaler.transform(rec_df[REC_FEATURES])

print("✅ All models loaded!")

# ── Define Input Schemas ───────────────────────────────────────────────
class SongFeatures(BaseModel):
    danceability:     float
    energy:           float
    loudness:         float
    speechiness:      float
    acousticness:     float
    instrumentalness: float
    liveness:         float
    valence:          float
    tempo:            float
    duration_ms:      float
    genre_encoded:    int

class RecommendRequest(BaseModel):
    song_name: str
    n:         int = 5

# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "🎵 MusicIQ API is running!"}


@app.post("/predict")
def predict_hit(features: SongFeatures):
    """
    Takes audio features of a song
    Returns whether it's predicted to be a hit
    """
    input_data = [[
        features.danceability,
        features.energy,
        features.loudness,
        features.speechiness,
        features.acousticness,
        features.instrumentalness,
        features.liveness,
        features.valence,
        features.tempo,
        features.duration_ms,
        features.genre_encoded
    ]]

    scaled = scaler.transform(input_data)
    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]
    confidence  = round(float(max(probability)) * 100, 1)

    return {
        "prediction": "Hit" if prediction == 1 else "Not a Hit",
        "confidence": confidence,
        "hit_probability": round(float(probability[1]) * 100, 1)
    }


@app.post("/recommend")
def recommend(request: RecommendRequest):
    """
    Takes a song name
    Returns N most similar songs
    """
    matches = rec_df[rec_df['track_name'].str.lower() == request.song_name.lower()]

    if matches.empty:
        matches = rec_df[rec_df['track_name'].str.lower().str.contains(
                         request.song_name.lower(), na=False)]

    if matches.empty:
        return {"error": f"Song '{request.song_name}' not found"}

    input_song = matches.sort_values('popularity', ascending=False).iloc[0]
    input_idx  = input_song.name

    input_vector = rec_scaled[input_idx].reshape(1, -1)
    similarities = cosine_similarity(input_vector, rec_scaled)[0]

    rec_copy = rec_df.copy()
    rec_copy['similarity'] = similarities
    rec_copy = rec_copy.drop(index=input_idx)

    results = rec_copy.sort_values(
        ['similarity', 'popularity'],
        ascending=[False, False]
    ).head(request.n)

    return {
        "input_song":    input_song['track_name'],
        "artist":        input_song['artists'],
        "genre":         input_song['track_genre'],
        "popularity":    int(input_song['popularity']),
        "recommendations": results[['track_name', 'artists', 
                                    'track_genre', 'popularity', 
                                    'similarity']].to_dict(orient='records')
    }


@app.get("/search")
def search_songs(query: str):
    """Search for songs by name"""
    results = rec_df[rec_df['track_name'].str.lower().str.contains(
                     query.lower(), na=False)]
    results = results.sort_values('popularity', ascending=False).head(10)

    return {
        "results": results[['track_name', 'artists', 
                            'track_genre', 'popularity']].to_dict(orient='records')
    }


@app.get("/genres")
def get_genres():
    """Returns all available genres"""
    genres = sorted(rec_df['track_genre'].unique().tolist())
    return {"genres": genres}
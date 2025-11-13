from fastapi import FastAPI, Query
import requests
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

app = FastAPI(title="Insta-Analyzer API")

API_KEY = st.secrets.get("RAPIDAPI_KEY")
BASE_URL = "https://instagram-scraper-20251.p.rapidapi.com"


# -------------------------------
# Test Root Endpoint
# -------------------------------
@app.get("/")
def root():
    return {"message": "FastAPI is running!"}


# -------------------------------
# Helper: Fetch User Posts
# -------------------------------
def fetch_posts(username, limit):
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "instagram-scraper-20251.p.rapidapi.com"
    }

    url = f"{BASE_URL}/userposts/?username_or_id={username}&count={limit}"
    response = requests.get(url, headers=headers).json()

    return response.get("data", {}).get("items", [])


# -------------------------------
# Helper: Fetch User Info
# -------------------------------
def fetch_user_info(username):
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "instagram-scraper-20251.p.rapidapi.com"
    }

    url = f"{BASE_URL}/userinfo/?username_or_id={username}"
    response = requests.get(url, headers=headers).json()

    return response.get("data", {})


# -------------------------------
# 🔥 1. ANALYZE ENDPOINT
# -------------------------------
@app.get("/analyze")
def analyze(usernames: str = Query(...), limit: int = 10):
    names = [u.strip() for u in usernames.split(",")]

    all_posts = []
    total_followers = 0

    for name in names:
        user_info = fetch_user_info(name)
        followers = user_info.get("follower_count", 0)
        total_followers += followers

        posts = fetch_posts(name, limit)

        for p in posts:
            likes = p.get("like_count", 0)
            views = (
                p.get("view_count")
                or p.get("play_count")
                or p.get("video_view_count")
                or 1  # avoid divide-by-zero
            )
            eng = likes / views

            all_posts.append({
                "username": name,
                "likes": likes,
                "views": views,
                "engagement": eng
            })

    if not all_posts:
        return {"error": "No posts returned"}

    avg_eng = float(np.mean([p["engagement"] for p in all_posts]))

    return {
        "brands_analyzed": names,
        "total_followers": total_followers,
        "average_engagement": round(avg_eng, 4),
        "posts_analyzed": len(all_posts)
    }


# -------------------------------
# 🔥 2. BEST POSTING TIME
# -------------------------------
@app.get("/best_time")
def best_time(username: str = Query(...), limit: int = 20):
    posts = fetch_posts(username, limit)

    if not posts:
        return {"error": "No posts returned"}

    hours = []
    for p in posts:
        ts = p.get("taken_at")
        if ts:
            hour = int(np.datetime64(ts, "h").astype(int) % 24)
            hours.append(hour)

    if not hours:
        return {"error": "No valid post timestamps"}

    # find the most common posting hour
    best_hour = int(np.bincount(hours).argmax())

    return {"username": username, "best_hour_to_post": best_hour}


# -------------------------------
# 🔥 3. ENGAGEMENT FORECAST
# -------------------------------
@app.get("/forecast")
def forecast(username: str = Query(...), limit: int = 15):
    posts = fetch_posts(username, limit)

    if not posts:
        return {"error": "No posts returned"}

    engagements = []
    for p in posts:
        likes = p.get("like_count", 0)
        views = p.get("view_count") or p.get("play_count") or 1
        engagements.append(likes / views)

    X = np.arange(len(engagements)).reshape(-1, 1)
    y = np.array(engagements)

    model = LinearRegression()
    model.fit(X, y)

    future = model.predict(np.array([[len(engagements) + 1]]))[0]

    return {
        "username": username,
        "predicted_engagement_next_post": round(float(future), 4)
    }


# -------------------------------
# 🔥 4. HASHTAG SUGGESTIONS (simple)
# -------------------------------
@app.get("/hashtags")
def hashtags(keyword: str = Query(...)):
    ideas = [
        f"#{keyword}",
        f"#{keyword}love",
        f"#{keyword}life",
        f"#{keyword}vibes",
        f"#{keyword}daily",
        f"{keyword}community",
        f"{keyword}fans"
    ]
    return {"keyword": keyword, "suggested_hashtags": ideas}

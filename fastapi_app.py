from fastapi import FastAPI, Query
import requests
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI(title="Insta-Analyzer API")

API_KEY = os.environ.get("RAPIDAPI_KEY")

@app.get("/")
def read_root():
    return {"message": "Welcome to Insta-Analyzer FastAPI endpoint!"}

@app.get("/analyze")
def analyze_instagram(usernames: str = Query(..., description="Comma-separated IG usernames"), limit: int = 10):
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "instagram-scraper-20251.p.rapidapi.com"
    }

    names = [u.strip() for u in usernames.split(",")]
    all_posts = []
    followers_list = []

    for u in names:
        f_url = f"https://instagram-scraper-20251.p.rapidapi.com/userinfo/?username_or_id={u}"
        fr = requests.get(f_url, headers=headers).json()
        followers = fr.get("data", {}).get("follower_count", None)
        followers_list.append({"username": u, "followers": followers})

        p_url = f"https://instagram-scraper-20251.p.rapidapi.com/userposts/?username_or_id={u}&count={limit}"
        pr = requests.get(p_url, headers=headers).json()
        posts = pr.get("data", {}).get("items", [])

        for post in posts:
            likes = post.get("like_count", 0)
            views = (
                post.get("view_count")
                or post.get("play_count")
                or post.get("video_view_count")
                or 0
            )
            all_posts.append({
                "username": u,
                "likes": likes,
                "views": views,
                "eng_score": likes / views if views else None
            })

    df = pd.DataFrame(all_posts)
    df_f = pd.DataFrame(followers_list)

    if df.empty:
        return {"error": "No posts returned"}

    avg_eng = round(df["eng_score"].mean(), 3)
    total_followers = df_f["followers"].sum()
    best_hour = None

    result = {
        "average_engagement": avg_eng,
        "total_followers": int(total_followers),
        "brands": list(df["username"].unique())
    }

    return result

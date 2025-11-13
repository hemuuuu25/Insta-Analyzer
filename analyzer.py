import instaloader

L = instaloader.Instaloader()

def get_user_data(username: str, limit: int = 5):
    """Fetch posts + follower count from Instagram public profile."""

    try:
        profile = instaloader.Profile.from_username(L.context, username)
    except Exception as e:
        return {"error": str(e), "posts": [], "followers": 0}

    followers = profile.followers
    posts = []

    for post in profile.get_posts():
        if len(posts) >= limit:
            break

        posts.append({
            "likes": post.likes,
            "comments": post.comments,
            "caption": post.caption if post.caption else "",
            "post_date": post.date_utc.isoformat()
        })

    return {
        "followers": followers,
        "posts": posts
    }


def calculate_engagement(posts):
    """Calculate average engagement rate from posts."""
    if not posts:
        return 0

    total_eng = 0
    total_posts = len(posts)

    for p in posts:
        total_eng += p["likes"] + p["comments"]

    return round(total_eng / total_posts, 3)

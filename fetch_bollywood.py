"""
Fetches Bollywood movies from TMDb and saves to data/bollywood_movies.csv.
Run once before starting the app: python3 fetch_bollywood.py
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import time
from pathlib import Path

TMDB_API_KEY = "327f98e4724270a68ccf7cf9cc2bf050"
BASE_URL = "https://api.themoviedb.org/3"
OUT_PATH = Path("data/bollywood_movies.csv")


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_genre_map(session: requests.Session) -> dict:
    r = session.get(f"{BASE_URL}/genre/movie/list",
                    params={"api_key": TMDB_API_KEY, "language": "en-US"})
    return {g["id"]: g["name"] for g in r.json()["genres"]}


def fetch_movies(pages: int = 20, min_votes: int = 30) -> pd.DataFrame:
    session = make_session()
    genre_map = get_genre_map(session)
    rows = []

    for page in range(1, pages + 1):
        params = {
            "api_key": TMDB_API_KEY,
            "with_original_language": "hi",
            "sort_by": "popularity.desc",
            "page": page,
            "vote_count.gte": min_votes,
            "include_adult": False,
            "primary_release_date.lte": "2026-12-31",
        }
        resp = session.get(f"{BASE_URL}/discover/movie", params=params)
        results = resp.json().get("results", [])

        for m in results:
            genres = [genre_map.get(gid, "") for gid in m.get("genre_ids", [])]
            genres_text = " ".join(g for g in genres if g)
            overview = (m.get("overview") or "")[:200]
            title = m.get("title") or m.get("original_title", "")

            rows.append({
                "item_id": m["id"],
                "title": title,
                "release_date": m.get("release_date", ""),
                "genres_text": genres_text,
                "description": f"{title} | {genres_text} | {overview}".strip(" |"),
                "source": "bollywood",
                "vote_average": m.get("vote_average", 0),
                "popularity": m.get("popularity", 0),
            })

        print(f"  Page {page}/{pages} — {len(results)} movies")
        time.sleep(0.5)

    df = pd.DataFrame(rows).drop_duplicates("item_id").reset_index(drop=True)
    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    print("Fetching Bollywood movies from TMDb...")
    df = fetch_movies(pages=20, min_votes=30)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} movies → {OUT_PATH}")
    print(df[["title", "genres_text", "release_date"]].head(10).to_string(index=False))

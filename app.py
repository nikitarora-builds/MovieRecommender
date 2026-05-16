import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
from io import BytesIO
from pathlib import Path

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = Path("data")
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# ── Data ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading MovieLens 100K...")
def load_data():
    if not (DATA_DIR / "ml-100k").exists():
        r = requests.get(ML100K_URL)
        with zipfile.ZipFile(BytesIO(r.content)) as z:
            z.extractall(DATA_DIR)

    ratings = pd.read_csv(
        DATA_DIR / "ml-100k" / "u.data",
        sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )

    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]
    movies = pd.read_csv(
        DATA_DIR / "ml-100k" / "u.item",
        sep="|", encoding="latin-1",
        names=["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols
    )
    movies["genres_text"] = movies[genre_cols].apply(
        lambda row: " ".join(g for g, v in zip(genre_cols, row) if v == 1), axis=1
    )
    movies["description"] = movies["title"] + " | " + movies["genres_text"]
    return ratings, movies


# ── SVD model ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training SVD model...")
def build_svd(_ratings):
    user_ids = _ratings["user_id"].unique()
    item_ids = _ratings["item_id"].unique()
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {it: i for i, it in enumerate(item_ids)}
    n_users, n_items = len(user_ids), len(item_ids)

    train_df, _ = train_test_split(_ratings, test_size=0.2, random_state=42)

    rows = train_df["user_id"].map(user_idx).values
    cols = train_df["item_id"].map(item_idx).values
    vals = train_df["rating"].astype(float).values
    R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

    counts = np.where(np.diff(R.indptr) == 0, 1, np.diff(R.indptr))
    user_means = np.array(R.sum(axis=1)).flatten() / counts

    R_centered = R.copy().astype(float)
    for i in range(n_users):
        s, e = R_centered.indptr[i], R_centered.indptr[i + 1]
        if s < e:
            R_centered.data[s:e] -= user_means[i]

    U, sigma, Vt = svds(R_centered, k=50)
    R_pred = np.clip((U * sigma) @ Vt + user_means[:, np.newaxis], 1, 5)

    return R_pred, user_idx, item_idx


# ── Semantic index ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building semantic index...")
def build_semantic(_movies):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(
        _movies["description"].tolist(),
        batch_size=128, show_progress_bar=False, normalize_embeddings=True
    )
    idx_to_item_id = {i: row.item_id for i, row in enumerate(_movies.itertuples())}
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings.astype(np.float32))
    return encoder, faiss_index, idx_to_item_id


# ── Recommender logic ────────────────────────────────────────────────────────

def recommend(user_id, query, alpha, n, ratings, movies, R_pred, user_idx, item_idx, encoder, faiss_index, idx_to_item_id):
    user_rated = ratings[ratings.user_id == user_id]["item_id"].tolist()
    all_ids = set(movies["item_id"])
    candidates = list(all_ids - set(user_rated))

    uid = user_idx.get(user_id)
    if uid is not None:
        raw = {mid: float(R_pred[uid, item_idx[mid]]) for mid in candidates if mid in item_idx}
        mn, mx = min(raw.values()), max(raw.values())
        cf = {mid: (r - mn) / (mx - mn or 1) for mid, r in raw.items()}
    else:
        cf = {mid: 0.5 for mid in candidates}

    if query.strip():
        q_vec = encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        sims, idxs = faiss_index.search(q_vec, min(300, faiss_index.ntotal))
        sem = {idx_to_item_id[idx]: float((sim + 1) / 2)
               for idx, sim in zip(idxs[0], sims[0])
               if idx_to_item_id[idx] in set(candidates)}
        scores = {mid: alpha * cf.get(mid, 0) + (1 - alpha) * sem.get(mid, 0) for mid in candidates}
    else:
        scores = cf

    top_ids = sorted(scores, key=lambda m: -scores[m])[:n]
    result = movies[movies["item_id"].isin(top_ids)][["item_id", "title", "genres_text"]].copy()
    result["score"] = result["item_id"].map(scores).round(3)
    return result.sort_values("score", ascending=False).reset_index(drop=True)


# ── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Hybrid Movie Recommender")
st.caption("Collaborative Filtering (SVD) + Semantic Search (sentence-transformers + FAISS)")

ratings, movies = load_data()
R_pred, user_idx, item_idx = build_svd(ratings)
encoder, faiss_index, idx_to_item_id = build_semantic(movies)

user_rated_map = ratings.groupby("user_id")["item_id"].apply(set).to_dict()

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    user_id = st.number_input(
        "User ID", min_value=1, max_value=int(ratings["user_id"].max()),
        value=1, step=1
    )

    query = st.text_input(
        "Mood / genre hint (optional)",
        placeholder="e.g. dark psychological thriller"
    )

    alpha = st.slider(
        "CF ← blend → Semantic",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="1.0 = pure collaborative filtering | 0.0 = pure semantic search"
    )

    n = st.slider("Number of recommendations", 5, 20, 10)

    st.divider()
    st.markdown(f"**Dataset:** MovieLens 100K  \n**Users:** {ratings['user_id'].nunique():,}  \n**Movies:** {len(movies):,}  \n**Ratings:** {len(ratings):,}")

# Main area
col1, col2 = st.columns([1.2, 2])

with col1:
    st.subheader(f"User {user_id}'s top-rated movies")
    watched = (
        ratings[ratings.user_id == user_id]
        .merge(movies[["item_id", "title"]], on="item_id")
        .sort_values("rating", ascending=False)
        [["title", "rating"]]
        .head(8)
        .reset_index(drop=True)
    )
    watched.index += 1
    st.dataframe(watched, use_container_width=True)

with col2:
    mode = "Hybrid" if query.strip() else "Collaborative Filtering"
    st.subheader(f"Recommendations ({mode})")

    recs = recommend(
        user_id, query, alpha, n,
        ratings, movies, R_pred, user_idx, item_idx,
        encoder, faiss_index, idx_to_item_id
    )

    recs.index += 1
    st.dataframe(
        recs[["title", "genres_text", "score"]].rename(columns={"genres_text": "genres"}),
        use_container_width=True
    )

    if query.strip():
        st.info(f'Blending user taste (CF weight: {alpha}) with semantic query: *"{query}"*')

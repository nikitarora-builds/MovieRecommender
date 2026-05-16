# Hybrid Movie Recommender

A two-layer recommender system that combines **collaborative filtering** (SVD matrix factorization) with **semantic search** (sentence-transformers + FAISS) to deliver personalized movie recommendations with natural-language mood queries.

---

## Architecture

```
User ID ──► SVD Matrix Factorization  ──► CF Score  ──┐
                                                        ├──► Weighted Blend ──► Top-N Recs
Query   ──► Sentence Embeddings + FAISS ──► Sem Score ──┘

score(u, i) = α · CF_score(u, i) + (1 - α) · Semantic_score(query, i)
```

| Layer | Tech | Purpose |
|---|---|---|
| Collaborative Filtering | `scipy` SVD (truncated, K=50) | Learns user taste from 100K ratings |
| Semantic Search | `sentence-transformers` + FAISS | Natural-language genre/mood queries |
| Hybrid Fusion | Weighted blend (α tunable) | Handles cold-start, improves diversity |

---

## Results

| Metric | Value |
|---|---|
| SVD RMSE (test set, 80/20 split) | ~1.00 |
| Catalog coverage — CF only | ~18% |
| Catalog coverage — Hybrid | ~34% |

Hybrid fusion nearly **doubles catalog coverage** vs pure collaborative filtering, surfacing a wider range of movies while preserving personalization.

---

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings from 943 users on 1,682 movies. Downloaded automatically on first run.

---

## Setup

```bash
git clone https://github.com/nikitarora-builds/MovieRecommender.git
cd MovieRecommender
pip install -r requirements.txt
```

---

## Run

**Streamlit demo (interactive UI)**
```bash
streamlit run app.py
```

**Jupyter notebook (full walkthrough)**
```bash
jupyter notebook hybrid_recommender.ipynb
```

---

## Demo

- Pick a **User ID** to load their rating history
- Optionally type a **mood/genre hint** (e.g. *"dark psychological thriller"*)
- Adjust the **CF ↔ Semantic blend** slider
- Get instant personalized recommendations

---

## Tech Stack

`Python` · `scipy` · `scikit-learn` · `sentence-transformers` · `FAISS` · `pandas` · `NumPy` · `Streamlit`

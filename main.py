"""FastAPI backend: local content model, local preferences, local Ollama AI."""
import asyncio
import os
import pickle
import sqlite3
import time
import hashlib
import re
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
DB_PATH = ROOT / "user_preferences.db"

app = FastAPI(title="CineSense Local AI", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"])
df: pd.DataFrame | None = None
matrix: Any = None
vectorizer: Any = None
title_index: dict[str, int] = {}
catalog = {}
movie_ids = []

# --- Caches ---
media_cache: dict[str, tuple[float, dict]] = {}
poster_cache: dict[str, tuple[float, str | None]] = {}  # movie_id -> (timestamp, verified_poster_url)
details_cache: dict[str, tuple[float, dict]] = {}       # movie_id -> (timestamp, full TMDB details)

# --- Genre aliases for Hinglish / casual requests ---
GENRE_ALIASES: dict[str, str] = {
    "horror": "Horror", "scary": "Horror", "darawna": "Horror", "darr": "Horror",
    "comedy": "Comedy", "funny": "Comedy", "mazak": "Comedy", "hasna": "Comedy",
    "action": "Action", "dhamaka": "Action", "fight": "Action",
    "romance": "Romance", "love": "Romance", "pyaar": "Romance", "romantic": "Romance",
    "thriller": "Thriller", "suspense": "Thriller", "rahasya": "Thriller",
    "drama": "Drama", "emotional": "Drama",
    "sci-fi": "Science Fiction", "scifi": "Science Fiction", "science fiction": "Science Fiction",
    "fantasy": "Fantasy", "jadoo": "Fantasy",
    "animation": "Animation", "cartoon": "Animation", "animated": "Animation",
    "adventure": "Adventure",
    "crime": "Crime", "mystery": "Mystery",
    "war": "War", "jung": "War",
    "family": "Family", "parivaar": "Family",
    "music": "Music", "gaana": "Music",
    "documentary": "Documentary", "doc": "Documentary",
    "history": "History", "itihas": "History",
    "western": "Western",
    "tv movie": "TV Movie",
}

# Build reverse lookup: collect all multi-word alias keys sorted longest-first
_MULTI_WORD_ALIASES = sorted(
    [(k, v) for k, v in GENRE_ALIASES.items() if " " in k],
    key=lambda pair: len(pair[0]),
    reverse=True,
)


class Preference(BaseModel):
    movie_id: Optional[str] = None
    session_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1)
    interaction: str = Field(pattern="^(like|dislike|watched|rating)$")
    rating: Optional[float] = Field(default=None, ge=1, le=5)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1000)


def normalise(value: str) -> str:
    return " ".join(value.lower().split())


def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS preferences (
            session_id TEXT, title TEXT, interaction TEXT, rating REAL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, title, interaction))""")
        con.commit()


def load_model() -> None:
    global df, matrix, vectorizer, title_index
    names = ("movies.pkl", "tfidf_matrix.pkl", "tfidf_vectorizer.pkl", "indices.pkl")
    absent = [name for name in names if not (ROOT / name).exists()]
    if absent:
        raise RuntimeError(f"Missing artifacts: {', '.join(absent)}. Run python movies.py first.")
    with open(ROOT / "movies.pkl", "rb") as f: df = pickle.load(f)
    with open(ROOT / "tfidf_matrix.pkl", "rb") as f: matrix = pickle.load(f)
    with open(ROOT / "tfidf_vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
    with open(ROOT / "indices.pkl", "rb") as f: saved = pickle.load(f)
    title_index = {normalise(str(title)): int(index) for title, index in saved.items()}
    # Enrich old artifacts in memory without changing their TF-IDF row order.
    raw = pd.read_csv(ROOT / "movies_metadata.csv", dtype=str, usecols=["id","title","overview","poster_path","runtime","release_date"]).fillna("")
    by_story = raw.drop_duplicates(["title","overview"]).set_index(["title","overview"])
    extra = by_story.reindex(pd.MultiIndex.from_frame(df[["title","overview"]].fillna(""))).reset_index(drop=True)
    for column in ["id","poster_path","runtime","release_date"]:
        if column not in df: df[column] = extra[column]
    catalog.clear()
    movie_ids.clear()
    for i, row in enumerate(df.itertuples()):
        key = str(row.id)
        if not key.isdigit() or key in catalog:
            key = "film-" + hashlib.sha256(f"{key}|{row.title}|{row.overview}|{row.release_date}".encode()).hexdigest()[:24]
        while key in catalog: key += "-2"
        catalog[key] = i
        movie_ids.append(key)


@app.on_event("startup")
def start() -> None:
    init_db()
    load_model()


def ready() -> None:
    if df is None or matrix is None or vectorizer is None:
        raise HTTPException(503, "Recommendation model is not loaded.")


def genres(value: Any) -> list[str]:
    return str(value).split() if isinstance(value, str) else []


def numeric_column(name: str) -> pd.Series:
    """Old artifacts contain a duplicate popularity header; safely use its first column."""
    assert df is not None
    value = df.loc[:, name] if name in df.columns else pd.Series(0, index=df.index)
    if isinstance(value, pd.DataFrame): value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce").fillna(0)


def card(i: int, scores: dict[str, float] | None = None) -> dict[str, Any]:
    assert df is not None
    row = df.iloc[int(i)]
    number = numeric_column("vote_average").iloc[int(i)]
    poster = str(row.get("poster_path",""))
    mid = movie_ids[i]
    return {
        "movie_id": mid,
        "year": str(row.release_date)[:4] if hasattr(row, "release_date") else "",
        "runtime": str(row.runtime) if hasattr(row, "runtime") else "",
        "title": str(row["title"]),
        "overview": str(row.get("overview", "") or ""),
        "genres": genres(row.get("genres", "")),
        "vote_average": float(number) if pd.notna(number) else 0.0,
        "popularity": float(numeric_column("popularity").iloc[int(i)]),
        "scores": scores or {},
        "poster_url": "https://image.tmdb.org/t/p/w500"+poster if poster.startswith("/") else None,
        "tmdb_id": str(row.get("id","")) if str(row.get("id","")).isdigit() else None,
        "trailer_url": None,
    }


def prefs(session_id: str) -> pd.DataFrame:
    with closing(sqlite3.connect(DB_PATH)) as con:
        return pd.read_sql_query("SELECT title, interaction, rating FROM preferences WHERE session_id=?", con, params=(session_id,))


def personal_score(session_id: str) -> tuple[np.ndarray, set[int]]:
    """Turn one user's saved feedback into a local re-ranking score."""
    assert matrix is not None
    data = prefs(session_id)
    score, watched, liked, disliked, weights = np.zeros(matrix.shape[0]), set(), [], [], []
    for item in data.itertuples(index=False):
        i = catalog.get(item.title[9:]) if item.title.startswith("@catalog:") else title_index.get(normalise(item.title))
        if i is None: continue
        if item.interaction == "watched": watched.add(i)
        elif item.interaction == "dislike" or (item.interaction == "rating" and (item.rating or 3) <= 2): disliked.append(i)
        elif item.interaction == "like" or (item.interaction == "rating" and (item.rating or 0) >= 4):
            liked.append(i); weights.append(1 if item.interaction == "like" else (item.rating - 3) / 2)
    if liked: score += np.average(cosine_similarity(matrix, matrix[liked]), axis=1, weights=weights)
    if disliked: score -= .6 * cosine_similarity(matrix, matrix[disliked]).max(axis=1)
    return np.clip(score, -1, 1), watched


def rank(content: np.ndarray, session_id: str, limit: int, source: int | None = None, genre_filter: set[str] | None = None) -> list[dict[str, Any]]:
    assert df is not None
    user, watched = personal_score(session_id)
    raw_pop = numeric_column("popularity").to_numpy(float)
    pop = np.log1p(np.maximum(raw_pop, 0)); pop = pop / pop.max() if pop.max() else pop
    final = .55 * content + .30 * user + .15 * pop
    output = []
    for i in np.argsort(final)[::-1]:
        if int(i) in watched or int(i) == source: continue
        # Apply genre filter if specified
        if genre_filter:
            movie_genres = set(genres(df.iloc[int(i)].get("genres", "")))
            if not movie_genres.intersection(genre_filter):
                continue
        output.append(card(int(i), {"content": round(float(content[i]), 4), "preference": round(float(user[i]), 4), "popularity": round(float(pop[i]), 4), "final": round(float(final[i]), 4)}))
        if len(output) == limit: break
    return output


def by_title(title: str, session_id: str, limit: int) -> list[dict[str, Any]]:
    ready(); i = title_index.get(normalise(title))
    if i is None: raise HTTPException(404, f"Movie '{title}' was not found.")
    return rank(cosine_similarity(matrix[i], matrix).flatten(), session_id, limit, i)


def detect_genres(text: str) -> set[str]:
    """Extract genre names from a user query, handling Hinglish and English aliases."""
    text_lower = text.lower()
    found: set[str] = set()

    # First check multi-word aliases (longest first)
    for alias, genre in _MULTI_WORD_ALIASES:
        if alias in text_lower:
            found.add(genre)
            text_lower = text_lower.replace(alias, "")

    # Then check single-word aliases
    words = set(re.split(r"[\s,;.!?]+", text_lower))
    for word in words:
        if word in GENRE_ALIASES:
            found.add(GENRE_ALIASES[word])

    return found


@app.get("/health")
async def health() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client: online = (await client.get(f"{OLLAMA_URL}/api/tags")).is_success
    except httpx.HTTPError: online = False
    return {"status": "ok", "model_loaded": df is not None, "ollama_online": online, "ollama_model": OLLAMA_MODEL}


@app.get("/movies/search")
def search(q: str = Query(min_length=1), limit: int = Query(8, ge=1, le=20)) -> list[dict[str, Any]]:
    ready(); matches = df[df.title.astype(str).str.contains(q, case=False, na=False, regex=False)].head(limit)
    return [card(i) for i in matches.index]


@app.get("/recommendations/{title}")
def recommendations(title: str, session_id: str = "guest", limit: int = Query(8, ge=1, le=20)) -> list[dict[str, Any]]:
    return by_title(title, session_id, limit)


@app.get("/media/{title}")
async def media(title: str) -> dict[str, Optional[str]]:
    return await fetch_media(title, title_index.get(normalise(title)))


def resolve(movie_id: str):
    if movie_id not in catalog: raise HTTPException(404, "Film not found in this catalog.")
    return catalog[movie_id]


@app.get("/catalog/{movie_id}")
def film_lookup(movie_id: str):
    return card(resolve(movie_id))


@app.get("/discovery")
def discovery():
    return [card(title_index[title]) for title in ("avatar","thor: ragnarok","interstellar","toy story","spirited away","inception") if title in title_index]


@app.get("/catalog/{movie_id}/recommendations")
def similar(movie_id: str, session_id: str = "guest"):
    i = resolve(movie_id)
    return rank(cosine_similarity(matrix[i], matrix).flatten(), session_id, 12, i)


@app.get("/catalog/{movie_id}/media")
async def artwork(movie_id: str):
    i = resolve(movie_id)
    return await fetch_media(str(df.iloc[i].title), i)


@app.get("/catalog/{movie_id}/poster")
async def poster_lookup(movie_id: str):
    """Return a verified poster URL for a movie. Tries catalog path first, then live TMDB lookup.
    Cached for 1 hour per movie. Does NOT fetch trailers or credits."""
    i = resolve(movie_id)
    row = df.iloc[i]

    # Check cache
    cached = poster_cache.get(movie_id)
    if cached and time.monotonic() - cached[0] < 3600:
        return {"poster_url": cached[1], "movie_id": movie_id}

    # First try the catalog poster path
    catalog_poster = str(row.get("poster_path", ""))
    if catalog_poster.startswith("/"):
        catalog_url = f"https://image.tmdb.org/t/p/w500{catalog_poster}"
        # Verify it exists with a HEAD request
        try:
            async with httpx.AsyncClient(timeout=4) as client:
                resp = await client.head(catalog_url)
                if resp.status_code == 200:
                    poster_cache[movie_id] = (time.monotonic(), catalog_url)
                    return {"poster_url": catalog_url, "movie_id": movie_id}
        except httpx.HTTPError:
            pass

    # Catalog poster failed — try live TMDB lookup
    if TMDB_API_KEY:
        tmdb_id = str(row.get("id", ""))
        try:
            async with httpx.AsyncClient(timeout=6) as client:
                if tmdb_id.isdigit():
                    r = await client.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}", params={"api_key": TMDB_API_KEY})
                    r.raise_for_status()
                    result = r.json()
                else:
                    params = {"api_key": TMDB_API_KEY, "query": str(row.title)}
                    rd = str(row.get("release_date", ""))[:4]
                    if rd.isdigit():
                        params["year"] = rd
                    r = await client.get("https://api.themoviedb.org/3/search/movie", params=params)
                    r.raise_for_status()
                    result = next(iter(r.json().get("results", [])), None)
                if result and result.get("poster_path"):
                    url = f"https://image.tmdb.org/t/p/w500{result['poster_path']}"
                    poster_cache[movie_id] = (time.monotonic(), url)
                    return {"poster_url": url, "movie_id": movie_id}
        except httpx.HTTPError:
            pass

    # Nothing worked
    poster_cache[movie_id] = (time.monotonic(), None)
    return {"poster_url": None, "movie_id": movie_id}


@app.get("/catalog/{movie_id}/details")
async def movie_details(movie_id: str):
    """Fetch comprehensive movie metadata from TMDB: cast, crew, certifications, budget, etc.
    Cached for 1 hour. Returns what's available; missing fields are null."""
    i = resolve(movie_id)
    row = df.iloc[i]

    # Check cache
    cached = details_cache.get(movie_id)
    if cached and time.monotonic() - cached[0] < 3600:
        return cached[1]

    base = {
        "movie_id": movie_id,
        "title": str(row["title"]),
        "overview": str(row.get("overview", "") or ""),
        "genres": genres(row.get("genres", "")),
        "year": str(row.release_date)[:4] if hasattr(row, "release_date") else "",
        "runtime": int(row.runtime) if hasattr(row, "runtime") and str(row.runtime).isdigit() else None,
        "vote_average": float(numeric_column("vote_average").iloc[int(i)]),
        "tagline": None,
        "original_title": None,
        "original_language": None,
        "status": None,
        "release_date": str(row.release_date) if hasattr(row, "release_date") else None,
        "budget": None,
        "revenue": None,
        "vote_count": None,
        "imdb_id": None,
        "homepage": None,
        "production_companies": [],
        "production_countries": [],
        "directors": [],
        "writers": [],
        "cast": [],
        "certification": None,
        "certification_country": None,
        "poster_url": None,
        "backdrop_url": None,
    }

    if not TMDB_API_KEY:
        details_cache[movie_id] = (time.monotonic(), base)
        return base

    tmdb_id = str(row.get("id", ""))
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Resolve TMDB ID
            resolved_id = None
            if tmdb_id.isdigit():
                resolved_id = int(tmdb_id)
            else:
                params = {"api_key": TMDB_API_KEY, "query": str(row.title)}
                rd = str(row.get("release_date", ""))[:4]
                if rd.isdigit():
                    params["year"] = rd
                r = await client.get("https://api.themoviedb.org/3/search/movie", params=params)
                r.raise_for_status()
                result = next(iter(r.json().get("results", [])), None)
                if result:
                    resolved_id = result["id"]

            if not resolved_id:
                details_cache[movie_id] = (time.monotonic(), base)
                return base

            # Fetch movie details + credits + release dates in one batch
            detail_url = f"https://api.themoviedb.org/3/movie/{resolved_id}"
            credits_url = f"https://api.themoviedb.org/3/movie/{resolved_id}/credits"
            releases_url = f"https://api.themoviedb.org/3/movie/{resolved_id}/release_dates"

            detail_resp, credits_resp, releases_resp = await asyncio.gather(
                client.get(detail_url, params={"api_key": TMDB_API_KEY}),
                client.get(credits_url, params={"api_key": TMDB_API_KEY}),
                client.get(releases_url, params={"api_key": TMDB_API_KEY}),
            )

            if detail_resp.is_success:
                d = detail_resp.json()
                base["tagline"] = d.get("tagline") or None
                base["original_title"] = d.get("original_title") or None
                base["original_language"] = d.get("original_language") or None
                base["status"] = d.get("status") or None
                base["release_date"] = d.get("release_date") or base["release_date"]
                base["budget"] = d.get("budget") if d.get("budget") else None
                base["revenue"] = d.get("revenue") if d.get("revenue") else None
                base["vote_average"] = d.get("vote_average") or base["vote_average"]
                base["vote_count"] = d.get("vote_count") or None
                base["runtime"] = d.get("runtime") or base["runtime"]
                base["imdb_id"] = d.get("imdb_id") or None
                base["homepage"] = d.get("homepage") or None
                base["overview"] = d.get("overview") or base["overview"]
                base["genres"] = [g["name"] for g in d.get("genres", [])] or base["genres"]
                base["year"] = str(d.get("release_date", ""))[:4] or base["year"]
                if d.get("poster_path"):
                    base["poster_url"] = f"https://image.tmdb.org/t/p/w500{d['poster_path']}"
                if d.get("backdrop_path"):
                    base["backdrop_url"] = f"https://image.tmdb.org/t/p/w1280{d['backdrop_path']}"
                base["production_companies"] = [
                    {"name": c["name"], "logo": f"https://image.tmdb.org/t/p/w200{c['logo_path']}" if c.get("logo_path") else None, "country": c.get("origin_country")}
                    for c in d.get("production_companies", [])
                ]
                base["production_countries"] = [c.get("name", c.get("iso_3166_1","")) for c in d.get("production_countries", [])]

            if credits_resp.is_success:
                cr = credits_resp.json()
                base["directors"] = [{"name": c["name"], "profile": f"https://image.tmdb.org/t/p/w185{c['profile_path']}" if c.get("profile_path") else None}
                                     for c in cr.get("crew", []) if c.get("job") == "Director"]
                base["writers"] = [{"name": c["name"], "job": c.get("job","")}
                                   for c in cr.get("crew", []) if c.get("department") == "Writing"][:6]
                base["cast"] = [{"name": c["name"], "character": c.get("character",""), "profile": f"https://image.tmdb.org/t/p/w185{c['profile_path']}" if c.get("profile_path") else None}
                                for c in cr.get("cast", [])[:12]]

            if releases_resp.is_success:
                rel = releases_resp.json()
                # Try US first, then any country
                for country_pref in ["US", "GB", "IN", None]:
                    for entry in rel.get("results", []):
                        if country_pref and entry.get("iso_3166_1") != country_pref:
                            continue
                        for rd_entry in entry.get("release_dates", []):
                            cert = rd_entry.get("certification", "").strip()
                            if cert:
                                base["certification"] = cert
                                base["certification_country"] = entry.get("iso_3166_1", "")
                                break
                        if base["certification"]:
                            break
                    if base["certification"]:
                        break

    except httpx.HTTPError:
        pass
    except Exception:
        pass

    details_cache[movie_id] = (time.monotonic(), base)
    return base


async def fetch_media(title, index):
    """Return optional legal TMDB/YouTube media; no key means a clean no-media result."""
    if not TMDB_API_KEY:
        return {"poster_url": None, "trailer_url": None}
    cache_key = movie_ids[index] if index is not None else title
    cached = media_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < 3600:
        return cached[1]
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            row = df.iloc[index] if index is not None else None
            movie_id = str(row.get("id", "")) if row is not None else ""
            if movie_id.isdigit():
                search_response = await client.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": TMDB_API_KEY})
                search_response.raise_for_status()
                result = search_response.json()
            else:
                params = {"api_key": TMDB_API_KEY, "query": title}
                if row is not None and str(row.get("release_date", ""))[:4].isdigit(): params["year"] = str(row["release_date"])[:4]
                search_response = await client.get("https://api.themoviedb.org/3/search/movie", params=params)
                search_response.raise_for_status()
                result = next(iter(search_response.json().get("results", [])), None)
            if not result:
                return {"poster_url": None, "trailer_url": None}
            videos_response = await client.get(f"https://api.themoviedb.org/3/movie/{result['id']}/videos", params={"api_key": TMDB_API_KEY})
            videos_response.raise_for_status()
            trailer = next((video for video in videos_response.json().get("results", []) if video.get("site") == "YouTube" and video.get("type") == "Trailer"), None)
            poster = result.get("poster_path")
            backdrop = result.get("backdrop_path")
            payload = {"poster_url": f"https://image.tmdb.org/t/p/w500{poster}" if poster else None,
                       "backdrop_url": f"https://image.tmdb.org/t/p/w1280{backdrop}" if backdrop else None,
                       "year": str(result.get("release_date", ""))[:4],
                       "trailer_url": f"https://www.youtube.com/embed/{trailer['key']}?autoplay=1&mute=1&controls=1" if trailer else None}
            media_cache[cache_key] = (time.monotonic(), payload)
            return payload
    except httpx.HTTPError:
        return {"poster_url": None, "trailer_url": None}


@app.post("/preferences")
def save_preference(value: Preference) -> dict[str, str]:
    ready()
    if normalise(value.title) not in title_index: raise HTTPException(404, "Movie not found.")
    storage_title = value.title
    if value.movie_id:
        i = resolve(value.movie_id)
        if df.iloc[i].title != value.title: raise HTTPException(422, "Movie ID and title differ.")
        if title_index[normalise(value.title)] != i: storage_title = "@catalog:" + value.movie_id
    if value.interaction == "rating" and value.rating is None: raise HTTPException(422, "Rating is required.")
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute("""INSERT INTO preferences(session_id,title,interaction,rating) VALUES(?,?,?,?)
          ON CONFLICT(session_id,title,interaction) DO UPDATE SET rating=excluded.rating,updated_at=CURRENT_TIMESTAMP""", (value.session_id, storage_title, value.interaction, value.rating)); con.commit()
    return {"status": "saved"}


@app.get("/preferences/{session_id}")
def read_preferences(session_id: str) -> list[dict[str, Any]]:
    records = prefs(session_id).astype(object).where(lambda data: data.notna(), None).to_dict("records")
    for item in records:
        title = item["title"]
        i = catalog.get(title[9:]) if title.startswith("@catalog:") else title_index.get(normalise(title))
        if i is not None: item.update(movie_id=movie_ids[i], title=str(df.iloc[i].title))
    return records


async def gemma(message: str, movies: list[dict[str, Any]]) -> str | None:
    titles = ", ".join(f'"{m["title"]}"' for m in movies[:6])
    context = "\n".join(f"- {m['title']} ({', '.join(m['genres'])}): {m['overview'][:180]}" for m in movies[:6])
    prompt = f"""You are CineSense, a movie recommendation assistant. Reply in Hinglish if the user writes Hinglish. 
IMPORTANT: Only discuss these exact movies: {titles}. Never invent or mention any title not in this list.
Explain why the top 2-4 of these match the user's request. Be concise (3-5 sentences). Do not use markdown formatting.

User: {message}
Movies:
{context}"""
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json={"model": OLLAMA_MODEL, "messages": [{"role":"user","content":prompt}], "stream":False, "options":{"temperature":.3}})
            r.raise_for_status(); return r.json().get("message", {}).get("content")
    except httpx.HTTPError: return None


def _validate_gemma_response(response: str | None, movies: list[dict[str, Any]]) -> str:
    """Ensure Gemma's response only references movies from the actual result list.
    If it mentions films not in the list or is unusable, generate a reliable explanation."""
    if not response or len(response.strip()) < 10:
        return _fallback_explanation(movies)

    # Check if Gemma invented titles (mentioned titles not in our list)
    movie_titles_lower = {m["title"].lower() for m in movies}
    # Simple heuristic: look for quoted titles or capitalized sequences that look like movie names
    # If the response is reasonable, use it; otherwise fall back
    clean = response.replace("**", "").replace("*", "").strip()
    if not clean:
        return _fallback_explanation(movies)
    return clean


def _fallback_explanation(movies: list[dict[str, Any]]) -> str:
    """Generate a short reliable explanation based on the actual ranked results."""
    if not movies:
        return "Koi matching movie nahi mili is request ke liye."
    top = movies[:3]
    parts = []
    for m in top:
        g = ", ".join(m["genres"][:2]) if m["genres"] else "Film"
        parts.append(f'"{m["title"]}" ({g})')
    return f"Top picks: {', '.join(parts)}. Yeh movies aapki request se match karti hain based on content similarity aur genre."


@app.post("/assistant/chat")
async def chat(value: ChatRequest) -> dict[str, Any]:
    ready(); text = normalise(value.message)

    # Detect genres from the query
    detected_genres = detect_genres(text)

    # Try to match a movie title first
    matched = next((title for title in title_index if title in text), None)

    if matched:
        movies = by_title(matched, value.session_id, 6)
    else:
        # Use TF-IDF similarity for content matching
        content_scores = cosine_similarity(vectorizer.transform([text]), matrix).flatten()
        movies = rank(content_scores, value.session_id, 6, genre_filter=detected_genres if detected_genres else None)

    # If genre filter produced too few results and we have genres, try broader search
    if detected_genres and len(movies) < 3:
        movies = rank(cosine_similarity(vectorizer.transform([text]), matrix).flatten(), value.session_id, 6, genre_filter=detected_genres)

    # Get Gemma explanation, validated against actual results
    raw_response = await gemma(value.message, movies)
    ai_available = raw_response is not None
    response = _validate_gemma_response(raw_response, movies)

    if not ai_available:
        response = "Ollama/Gemma available nahi hai. Local recommendations shown hain based on genre aur content matching. `ollama run gemma2:2b` chalao for AI explanations."

    return {"reply": response, "recommendations": movies, "ai_available": ai_available, "detected_genres": list(detected_genres)}

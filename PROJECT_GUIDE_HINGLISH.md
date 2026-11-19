# CineSense — A to Z Hinglish Guide

## Project kya karta hai?

CineSense movie recommendation app hai. Yeh **content-based model**, local preference system aur optional local Gemma AI ko combine karta hai. Cloud AI key ki zarurat nahi hai.

```text
movies_metadata.csv → movies.py → TF-IDF artifacts
                                     ↓
SQLite likes/ratings → FastAPI hybrid ranker → Streamlit UI
                                     ↓
                           Ollama Gemma 2B explanation
```

## Content model

`movies.py` movie ka title, genres, overview aur tagline ko `tags` mein join karta hai. **TF-IDF** text ko numeric vector banata hai: common words ki value kam aur meaningful unique words ki value zyada hoti hai. **Cosine similarity** do movie vectors compare karta hai; score 1 ke near ho to content zyada similar hota hai.

Saved files:

| File | Kaam |
|---|---|
| `movies.pkl` | Clean movie dataframe |
| `tfidf_matrix.pkl` | Har movie ke text vectors |
| `tfidf_vectorizer.pkl` | Query ko vector banata hai |
| `indices.pkl` | Title se safe row lookup |
| `main.py` | API, ranking, SQLite, Ollama |
| `frontend/` | Main React frontend: smooth browser-side UI without Streamlit reruns |
| `ui.py` | Old Streamlit prototype; main UI nahi hai |

## Hybrid logic

```text
final = 55% content match + 30% local preference + 15% popularity
```

- Like/4–5 rating similar movies ko boost karta hai.
- Dislike/1–2 rating similar movies ko reduce karta hai.
- Seen movie recommendation mein hide hoti hai.
- New user ke liye content + popularity use hota hai.

Yeh personalized hybrid model hai. Yeh multi-user collaborative filtering nahi hai; uske liye MovieLens `ratings.csv` aur many users ka data chahiye.

## Gemma AI ka role

Gemma 2B local Ollama model hai. Yeh Hinglish/English query samajhkar real recommendation results ko explain karta hai. Yeh fake movie invent nahi karta: backend pehle real movies retrieve karta hai, phir Gemma ko wahi context deta hai. Ollama off ho to normal recommendation still work karega.

Run:

```powershell
ollama run gemma2:2b
```

## Installation aur run

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python movies.py
uvicorn main:app --reload
```

Naye terminal mein React app run karo:

```powershell
cd frontend
npm install
npm run dev
```

Browser: `http://localhost:5173`

Homepage par featured film, search result, poster ya Gemma suggestion select karo. Har film ka apna `/movie/<catalog-id>` URL hai. Link share ya refresh karne par wahi film khulti hai. Back se search aur scroll position wapas milti hai. Film ke page par uska artwork, synopsis, trailer, Like/Seen aur phir similar films dikhte hain. Backdrop unavailable ho to usi film ka poster ya neutral background dikhta hai.

Explore par click karne se selected movie ki details aur recommendations section khulta hai. Like aur Seen save hone ke baad highlighted rehte hain, browser reload ke baad bhi. Seen movies future recommendations mein hide hoti hain. Session browser ki local storage se linked hai.

`.env` mein `TMDB_API_KEY=your_key` add karo aur backend restart karo. Backend posters/trailers TMDB se lata hai; key browser ko nahi bhejta. Metadata ek ghante ke liye memory mein cache hota hai. TMDB unavailable ho to placeholder dikhega aur recommendations kaam karti rahengi. Ollama unavailable ho to local recommendations ke saath fallback message dikhega.

## Trailer aur posters

`.env` mein optional `TMDB_API_KEY` ho to selected movie ka poster aur muted YouTube trailer preview show ho sakta hai. Valid trailer na mile to app blank player ya fake video nahi dikhata. Browser autoplay sound block karta hai, isliye preview hamesha muted start hota hai; player controls se user unmute kar sakta hai.

## UI use

1. Movie search karo ya quick suggestion choose karo.
2. **Explore** click karke similar movies dekho.
3. Heart/Seen actions se local taste improve karo.
4. Chat box mein bolo: `Mujhe dark sci-fi chahiye, romance kam ho`.

## Common errors

| Error | Fix |
|---|---|
| Missing artifacts | `python movies.py` run karo |
| Backend unavailable | `uvicorn main:app --reload` run karo |
| React page nahi khul raha | `cd frontend`, then `npm run dev` run karo aur `localhost:5173` kholo |
| Gemma unavailable | `ollama run gemma2:2b` chalao |
| Movie not found | Search result se exact title select karo |

## Security aur future

`.env` ko GitHub par upload mat karo. Local preferences `user_preferences.db` mein store hoti hain. Future upgrades: TMDB posters/trailers, account login, MovieLens ratings se true collaborative filtering, aur semantic embeddings.

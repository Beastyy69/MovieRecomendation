# CineSense

**Render deployment with Gemma included:** see [RENDER_DEPLOY.md](RENDER_DEPLOY.md).
The repo contains a two-service blueprint: React Static Site and a paid Docker
Web Service running FastAPI + private Ollama. Gemma weights are downloaded into
the Docker image during build. No cloud AI key is needed.

Local TF-IDF movie recommendations, SQLite preferences, optional local Gemma through Ollama, and a React frontend.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama run gemma2:2b
python movies.py
uvicorn main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The older Streamlit UI (`ui.py`) is no longer the main frontend. See `PROJECT_GUIDE_HINGLISH.md` for the full guide.

Explore opens the film's shareable catalog URL. Use Back to return to discovery. Likes and watched states persist for the browser's local session ID, and watched titles are excluded from future recommendations.

Add `TMDB_API_KEY=your_key` to the existing `.env` for posters and trailers. The key stays on the backend; metadata is cached in memory for one hour. Without TMDB the recommender still works with poster placeholders. Ollama is optional for explanations.

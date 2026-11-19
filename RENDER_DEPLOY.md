# CineSense on Render: exactly two services

1. React/Vite **Static Site**.
2. Docker **Web Service** containing FastAPI **and Ollama Gemma 2B**.

No separate AI service, external AI API, or Postgres service is required. The
backend includes a small persistent disk for SQLite; this is storage attached to
the same service, not a third service. Existing laptop preferences are not
uploaded. A new hosted database is created on the disk.

## Compute and model

`gemma2:2b` has approximately 1.6 GB of weights, plus runtime/context memory,
Python, the catalog and sparse model. Do not use the free 512 MB instance for
this configuration. The blueprint selects a **paid Pro** backend. Review the
current price and RAM in the Render dashboard before applying; increase memory
if your observed workload needs it. CPU inference can be slow, especially on
the first request. This is not a promise of GPU performance or production load
capacity.

The Docker build downloads the model directly to `/opt/ollama/models` in the
image. Git contains the download configuration, not a 1.6 GB binary. Rebuilds can
reuse Docker's cached model layer. Your laptop can be switched off after deploy.
Do not mount a blank disk on `/opt/ollama/models`, as that would hide the bundled
weights. The model uses Google's Gemma terms: https://ai.google.dev/gemma/terms.

## Deploy manually (easiest way to enter the actual generated URLs)

### A. Backend

Render → New → Web Service → connect `Beastyy69/MovieRecomendation`, branch `main`.

| Setting | Value |
| --- | --- |
| Language/runtime | Docker |
| Root directory | Leave blank |
| Dockerfile | `./Dockerfile` |
| Docker command | Leave blank; use image CMD |
| Health check | `/healthz` |
| Instance | Paid, enough RAM for Gemma + Python; blueprint defaults to Pro |

Set backend environment values:

```dotenv
TMDB_API_KEY=your_real_key
OLLAMA_MODEL=gemma2:2b
OLLAMA_TIMEOUT_SECONDS=120
DB_PATH=/var/data/user_preferences.db
```

Attach a 1 GB persistent disk at `/var/data`. Do not enter a pip build command or
uvicorn start command: the Dockerfile and supervisor run both processes.
The initial build downloads the Ollama image and Gemma weights, so allow time.
Copy the backend's actual HTTPS URL after creation.

### B. Frontend

Render → New → Static Site → same repository and branch.

| Setting | Value |
| --- | --- |
| Root directory | `frontend` |
| Build command | `npm ci && npm run build` |
| Publish directory | `dist` |

Set `VITE_API_URL` to the backend's actual HTTPS origin, for example
`https://your-cinesense-api.onrender.com` (no `/api` suffix). Rebuild the frontend
whenever this value changes, because Vite embeds it during build.

Add a **Rewrite**, not a redirect: `/*` → `/index.html`. This preserves direct
movie-page links and refreshes. Copy the frontend's actual HTTPS URL.

### C. Connect CORS

On the backend set `CORS_ORIGINS` to the frontend's exact HTTPS origin, with no
trailing slash, then redeploy the backend. Multiple origins can be comma-separated.
Never put `TMDB_API_KEY` in any `VITE_` variable.

## Blueprint alternative

`render.yaml` defines both services, the SQLite disk and the SPA rewrite.
Use Render → New → Blueprint and select this repo. Review the paid plan first.
Supply `TMDB_API_KEY`, `CORS_ORIGINS` and `VITE_API_URL` when prompted. Use the
actual assigned domains, not assumed service names; correct them in the service
settings and rebuild/redeploy if necessary. No Render resources have been
provisioned merely by committing this file.

## Verify after deploying

- Backend `/healthz`: HTTP 200, `status: ok`.
- Backend `/health`: `model_loaded`, `ollama_online`, and
  `ollama_model_available` should all be true.
- Open frontend; check posters, search, movie links and a direct detail-page refresh.
- Ask a short question; inspect `/assistant/chat` response for `ai_available: true`.
- Like/Seen, refresh, then redeploy and confirm the disk retained preferences.
- Use browser Network tab: requests must reach the HTTPS backend, not localhost.

AI requests have a 120-second backend timeout and 150-second browser timeout.
One inference runs at a time. If busy/unavailable, content recommendations still
return with an honest fallback message. The public port serves FastAPI only;
Ollama listens on `127.0.0.1:11434` inside the container.

## Local container smoke test (requires Docker)

```powershell
docker build -t cinesense-api .
docker run --rm -p 8000:10000 --env-file .env -e CORS_ORIGINS=http://localhost:5173 -e DB_PATH=/var/data/user_preferences.db -v cinesense-data:/var/data cinesense-api
```

Then run the normal frontend with `npm run dev`. The Vite proxy already targets
port 8000. Docker needs internet during build, but inference uses the bundled
model after deployment.

## Validation recorded during preparation

- Frontend production build passed.
- Five API/supervisor tests passed, covering CORS, catalog lookup, preferences,
  AI failure/busy fallback, readiness and process shutdown.
- Real local Ollama `gemma2:2b` generated a response through `main.gemma`.
- Official pinned Ollama image tag was available in the registry.
- Docker was not installed on the preparation machine, so the full Linux image
  build and a live Render deployment have **not** been verified.

Run API tests with bounded numerical-library threads:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

References: https://render.com/docs/docker,
https://render.com/docs/blueprint-spec, https://docs.ollama.com/docker.

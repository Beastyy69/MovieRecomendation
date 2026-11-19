# Gemma model directory

Gemma 2B is downloaded into `/opt/ollama/models` **inside the Docker image** at
build time. Both its content-addressed blobs and manifest are required, not a
single Python file. The roughly 1.6 GB weights are intentionally not Git blobs.

Run `docker build -t cinesense-api .` from the repository root to create the
self-contained backend image. The same build runs on Render. The host laptop
does not need to remain on after deployment.

Model: https://ollama.com/library/gemma2:2b
Terms: https://ai.google.dev/gemma/terms

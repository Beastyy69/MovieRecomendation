"""Supervise private Ollama and public FastAPI in one Linux container."""
import json
import os
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def wait_for_ollama(process, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Ollama exited before becoming ready")
        try:
            with urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
                return json.load(response)
        except (URLError, TimeoutError, OSError):
            time.sleep(0.5)
    raise RuntimeError("Ollama did not become ready within 60 seconds")


def stop(processes):
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    processes = []
    def shutdown(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    # Never publish the Ollama API or model-management endpoints.
    os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
    os.environ["OLLAMA_URL"] = "http://127.0.0.1:11434"
    model = os.environ.get("OLLAMA_MODEL", "gemma2:2b")
    try:
        ollama = subprocess.Popen(["ollama", "serve"])
        processes.append(ollama)
        tags = wait_for_ollama(ollama)
        if "--prepare-model" in sys.argv:
            subprocess.run(["ollama", "pull", model], check=True, timeout=1800)
            return 0
        if not any(item.get("name") == model for item in tags.get("models", [])):
            raise RuntimeError(f"Model {model} is not bundled. Rebuild the Docker image with this model.")
        api = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0", "--port", os.environ.get("PORT", "10000"), "--workers", "1"])
        processes.append(api)
        print("CineSense: API + private Ollama running; Gemma loads on first request.", flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        # A failed child must fail the container so Render can restart it.
        print("A service process exited; stopping container.", file=sys.stderr)
        return 1
    finally:
        stop(processes)


if __name__ == "__main__":
    sys.exit(main())

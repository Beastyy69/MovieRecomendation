"""Run: python -m unittest discover -s tests -v"""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["CORS_ORIGINS"] = "https://cinesense.example"
from fastapi.testclient import TestClient
import httpx
import main
from deploy import runtime


class DeploymentAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.old_db = main.DB_PATH
        main.DB_PATH = Path(cls.directory.name) / "nested" / "preferences.db"
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)
        main.DB_PATH = cls.old_db
        cls.directory.cleanup()

    def test_readiness_and_cors(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        response = self.client.options("/preferences", headers={
            "Origin": "https://cinesense.example", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://cinesense.example")
        self.assertEqual(self.client.options("/healthz", headers={
            "Origin": "https://unexpected.example", "Access-Control-Request-Method": "GET"}).status_code, 400)

    def test_catalog_and_preferences(self):
        movies = self.client.get("/movies/search", params={"q":"Avatar"}).json()
        movie = next(movie for movie in movies if movie["title"] == "Avatar")
        self.assertEqual(self.client.get(f'/catalog/{movie["movie_id"]}').status_code, 200)
        for interaction in ("like", "watched"):
            response = self.client.post("/preferences", json={"session_id":"deployment-test",
                "title":movie["title"], "movie_id":movie["movie_id"], "interaction":interaction})
            self.assertEqual(response.status_code, 200, response.text)
        rows = self.client.get("/preferences/deployment-test").json()
        self.assertEqual({row["interaction"] for row in rows}, {"like", "watched"})
        self.assertTrue(main.DB_PATH.exists())

    def test_ai_failure_releases_capacity(self):
        async def check():
            main.AI_GATE = asyncio.Semaphore(1)
            with patch.object(main.httpx, "AsyncClient", side_effect=httpx.ConnectError("offline")):
                self.assertIsNone(await main.gemma("horror", []))
            self.assertFalse(main.AI_GATE.locked())
            await main.AI_GATE.acquire()
            try:
                self.assertIsNone(await main.gemma("horror", []))
            finally:
                main.AI_GATE.release()
        asyncio.run(check())


class Supervisor(unittest.TestCase):
    def test_early_ollama_exit(self):
        process = MagicMock()
        process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "exited"):
            runtime.wait_for_ollama(process)

    def test_shutdown_reaps_children(self):
        process = MagicMock()
        process.poll.return_value = None
        runtime.stop([process])
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=15)


if __name__ == "__main__":
    unittest.main()

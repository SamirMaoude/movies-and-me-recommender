"""Exercise the HTTP contract and startup with a local ALS bundle."""

import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.api.main import DEFAULT_MODEL_URI, create_app
from src.api.service import RecommendationService
from src.models.collaborative import ImplicitRecommender
from src.models.registry import LoadedRecommender, load_recommender
from test_recommendation_service import make_model


class RecommendationAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="api-tests-")
        assert (
            Path(cls.directory.name).resolve().parent
            == Path(tempfile.gettempdir()).resolve()
        )
        cls.addClassCleanup(cls.directory.cleanup)
        cls.bundle = Path(cls.directory.name) / "model"
        cls.model = make_model()
        cls.model.save(cls.bundle, metadata={"run_id": "a" * 32})
        cls.service = RecommendationService.from_source(cls.bundle)

    def app(self, **kwargs):
        return create_app(str(self.bundle), **kwargs)

    def test_health_and_local_model_identity(self):
        with TestClient(self.app()) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            response = client.get("/model/info")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), self.service.model_info.model_dump())
            self.assertIsNone(response.json()["version"])

    def test_profiles_match_the_service_and_exclude_favorites(self):
        with TestClient(self.app()) as client:
            for favorites in (
                [],
                [999],
                [100],
                [100, 100, 999],
                [100, 200, 300, 400, 500, 600],
            ):
                with self.subTest(favorites=favorites):
                    response = client.post(
                        "/recommendations",
                        json={"favorite_movie_ids": favorites, "n": 3},
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(
                        response.json(),
                        self.service.recommend(favorites, n=3).model_dump(),
                    )
                    ids = [row["tmdb_id"] for row in response.json()["recommendations"]]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertFalse(set(ids).intersection(favorites))

    def test_invalid_bodies_return_422(self):
        bodies = [
            {},
            [],
            {"favorite_movie_ids": None},
            {"favorite_movie_ids": "100"},
            {"favorite_movie_ids": [True]},
            {"favorite_movie_ids": [0]},
            {"favorite_movie_ids": [-1]},
            {"favorite_movie_ids": [100.0]},
            {"favorite_movie_ids": ["100"]},
            {"favorite_movie_ids": [100] * 501},
            {"favorite_movie_ids": [], "unexpected": 1},
        ]
        bodies += [
            {"favorite_movie_ids": [], "n": value}
            for value in (0, 51, True, 3.0, "3", None)
        ]
        with TestClient(self.app()) as client:
            for body in bodies:
                with self.subTest(body=body):
                    response = client.post("/recommendations", json=body)
                    self.assertEqual(response.status_code, 422)
                    self.assertIn("detail", response.json())

    def test_default_limit_and_valid_boundaries(self):
        with TestClient(self.app()) as client:
            response = client.post("/recommendations", json={"favorite_movie_ids": []})
            self.assertEqual(response.json(), self.service.recommend([]).model_dump())
            for n in (1, 50):
                response = client.post(
                    "/recommendations", json={"favorite_movie_ids": [100] * 500, "n": n}
                )
                self.assertEqual(response.status_code, 200)
                self.assertLessEqual(len(response.json()["recommendations"]), n)

    def test_routes_are_unavailable_outside_the_lifespan(self):
        factory = Mock(return_value=self.service)
        app = self.app(service_factory=factory)
        client = TestClient(app)
        self.addCleanup(client.close)
        self.assertEqual(client.get("/health").status_code, 503)
        self.assertEqual(client.get("/model/info").status_code, 503)
        self.assertEqual(client.get("/version").status_code, 503)
        self.assertEqual(
            client.post(
                "/recommendations", json={"favorite_movie_ids": []}
            ).status_code,
            503,
        )
        factory.assert_not_called()

    def test_environment_is_read_at_startup_and_service_is_loaded_once(self):
        factory = Mock(return_value=self.service)
        app = create_app(service_factory=factory)
        factory.assert_not_called()
        values = {
            "MODEL_URI": "environment-model",
            "MLFLOW_TRACKING_URI": "tracking",
            "MLFLOW_REGISTRY_URI": "registry",
        }
        with patch.dict(os.environ, values):
            with TestClient(app) as client:
                for path in ("/health", "/model/info", "/health"):
                    self.assertEqual(client.get(path).status_code, 200)
                client.post("/recommendations", json={"favorite_movie_ids": []})
                self.assertIs(app.state.service, self.service)
        factory.assert_called_once_with(
            "environment-model", tracking_uri="tracking", registry_uri="registry"
        )
        self.assertIsNone(app.state.service)

    def test_explicit_configuration_takes_precedence(self):
        factory = Mock(return_value=self.service)
        values = {
            "MODEL_URI": "ignored",
            "MLFLOW_TRACKING_URI": "ignored",
            "MLFLOW_REGISTRY_URI": "ignored",
        }
        with patch.dict(os.environ, values):
            app = create_app(
                "explicit-model",
                tracking_uri="tracking",
                registry_uri="registry",
                service_factory=factory,
            )
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
        factory.assert_called_once_with(
            "explicit-model", tracking_uri="tracking", registry_uri="registry"
        )

    def test_default_source_is_an_exact_registered_version(self):
        factory = Mock(return_value=self.service)
        with patch.dict(os.environ, {}, clear=True):
            with TestClient(create_app(service_factory=factory)) as client:
                self.assertEqual(client.get("/health").status_code, 200)
        self.assertEqual(DEFAULT_MODEL_URI, "models:/movies-and-me-als/1")
        factory.assert_called_once_with(
            DEFAULT_MODEL_URI, tracking_uri=None, registry_uri=None
        )

    def test_missing_model_prevents_startup(self):
        app = create_app(str(Path(self.directory.name) / "missing"))
        with self.assertRaisesRegex(ValueError, "Invalid ALS artifact bundle"):
            with TestClient(app):
                self.fail("A missing model must not become ready")
        self.assertIsNone(app.state.service)

    def test_corrupt_model_prevents_startup(self):
        with tempfile.TemporaryDirectory(prefix="api-corrupt-") as directory:
            self.assertEqual(
                Path(directory).resolve().parent, Path(tempfile.gettempdir()).resolve()
            )
            bundle = Path(directory) / "model"
            shutil.copytree(self.bundle, bundle)
            (bundle / "model.npz").write_bytes(b"corrupt")
            app = create_app(str(bundle))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                with TestClient(app):
                    self.fail("A corrupt model must not become ready")
            self.assertIsNone(app.state.service)

    def test_computation_failure_returns_a_generic_500(self):
        broken = Mock(spec=RecommendationService)
        broken.recommend.side_effect = RuntimeError("private failure detail")
        with TestClient(
            self.app(service_factory=lambda *args, **kwargs: broken)
        ) as client:
            with self.assertLogs("src.api.main", level="ERROR"):
                response = client.post(
                    "/recommendations", json={"favorite_movie_ids": [100]}
                )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(), {"detail": "Recommendation computation failed."}
        )
        self.assertNotIn("private failure detail", response.text)

    def test_nonfinite_model_output_returns_500(self):
        service = RecommendationService(LoadedRecommender(self.model, "local-test"))
        result = {
            "method": "als",
            "recommendations": [{"tmdb_id": 200, "score": float("nan")}],
        }
        with patch.object(self.model, "recommend_with_method", return_value=result):
            with TestClient(
                self.app(service_factory=lambda *args, **kwargs: service)
            ) as client:
                with self.assertLogs("src.api.main", level="ERROR"):
                    response = client.post(
                        "/recommendations", json={"favorite_movie_ids": [100]}
                    )
        self.assertEqual(response.status_code, 500)

    def test_concurrent_requests_match_sequential_results(self):
        profiles = [[], [100], [200], [999], [100, 300], [400], [500, 600], [100, 100]]
        expected = [
            self.service.recommend(profile, n=3).model_dump() for profile in profiles
        ]
        with TestClient(self.app()) as client:

            def request(profile):
                response = client.post(
                    "/recommendations", json={"favorite_movie_ids": profile, "n": 3}
                )
                self.assertEqual(response.status_code, 200)
                return response.json()

            with ThreadPoolExecutor(max_workers=8) as pool:
                actual = list(pool.map(request, profiles))
        self.assertEqual(actual, expected)

    def test_requests_neither_reload_nor_train_the_model(self):
        with patch(
            "src.api.service.load_recommender", wraps=load_recommender
        ) as loader:
            with patch.object(
                ImplicitRecommender,
                "fit",
                side_effect=AssertionError("Unexpected training"),
            ):
                with TestClient(self.app()) as client:
                    for favorites in ([], [100], [999]):
                        self.assertEqual(
                            client.post(
                                "/recommendations",
                                json={"favorite_movie_ids": favorites},
                            ).status_code,
                            200,
                        )
            self.assertEqual(loader.call_count, 1)

    def test_registered_identity_is_returned_in_info_and_recommendations(self):
        service = RecommendationService(
            LoadedRecommender(
                self.model,
                "models:/movies-and-me-als/7",
                "movies-and-me-als",
                "7",
                "a" * 32,
            )
        )
        with TestClient(
            self.app(service_factory=lambda *args, **kwargs: service)
        ) as client:
            info = client.get("/model/info").json()
            response = client.post(
                "/recommendations", json={"favorite_movie_ids": [100]}
            )
        self.assertEqual(info["version"], "7")
        self.assertEqual(info["model_uri"], "models:/movies-and-me-als/7")
        self.assertEqual(response.json()["model"], info)

    def test_version_reports_the_revision_and_the_model(self):
        with patch.dict(os.environ, {"APP_REVISION": "b" * 40}):
            with TestClient(self.app()) as client:
                info = client.get("/model/info").json()
                self.assertEqual(
                    client.get("/version").json(), {"revision": "b" * 40, "model": info}
                )
            with TestClient(self.app(revision="c" * 40)) as client:
                self.assertEqual(client.get("/version").json()["revision"], "c" * 40)
            os.environ["APP_REVISION"] = ""
            with TestClient(self.app()) as client:
                self.assertIsNone(client.get("/version").json()["revision"])
            del os.environ["APP_REVISION"]
            with TestClient(self.app()) as client:
                self.assertIsNone(client.get("/version").json()["revision"])

    def test_interactive_documentation_describes_the_routes(self):
        with TestClient(self.app()) as client:
            self.assertEqual(client.get("/docs").status_code, 200)
            schema = client.get("/openapi.json").json()
        self.assertEqual(
            set(schema["paths"]),
            {"/health", "/model/info", "/recommendations", "/version"},
        )


if __name__ == "__main__":
    unittest.main()

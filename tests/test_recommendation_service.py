"""Check serving behavior with a small ALS model and no external services."""

import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.api.service import RecommendationService
from src.models.collaborative import ALSConfig, ImplicitRecommender
from src.models.registry import LoadedRecommender, load_recommender


def make_model():
    links = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40, 50, 60],
            "tmdbId": [100, 200, 300, 400, 500, 600],
        }
    )
    pairs = [
        (1, 10),
        (1, 20),
        (1, 30),
        (2, 10),
        (2, 30),
        (2, 40),
        (3, 20),
        (3, 50),
        (3, 60),
        (4, 40),
        (4, 50),
    ]
    train = pd.DataFrame(
        [(user, movie, movie * 10, True) for user, movie in pairs],
        columns=["userId", "movieId", "tmdbId", "is_positive"],
    )
    return ImplicitRecommender.fit(
        train,
        links,
        ALSConfig(factors=3, iterations=3, random_state=42),
    )


class RecommendationServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_model()
        cls.loaded = LoadedRecommender(
            cls.model,
            "models:/movies-and-me-als/7",
            "movies-and-me-als",
            "7",
            "a" * 32,
        )

    def setUp(self):
        self.service = RecommendationService(self.loaded)

    def test_profiles_match_native_recommendations_and_exclude_favorites(self):
        for favorites in (
            [],
            [999],
            [100],
            [100, 100, 999],
            [100, 200, 300, 400, 500, 600],
        ):
            with self.subTest(favorites=favorites):
                actual = self.service.recommend(favorites, n=3).model_dump()
                expected = self.model.recommend_with_method(favorites, n=3)
                self.assertEqual(actual["method"], expected["method"])
                self.assertEqual(actual["recommendations"], expected["recommendations"])
                ids = [row["tmdb_id"] for row in actual["recommendations"]]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertFalse(set(ids).intersection(favorites))
                self.assertEqual(actual["model"], self.service.model_info.model_dump())

    def test_known_favorites_are_counted_once_from_the_catalog(self):
        for favorites, known in (
            ([], 0),
            ([999], 0),
            ([100, 100, 999], 1),
            ([100, 200, 300, 400, 500, 600], 6),
        ):
            with self.subTest(favorites=favorites):
                self.assertEqual(self.service.count_known(favorites), known)

    def test_accepts_boundaries_and_returns_only_available_movies(self):
        self.assertEqual(len(self.service.recommend([], n=1).recommendations), 1)
        result = self.service.recommend([100] * 500, n=50)
        self.assertEqual(len(result.recommendations), 5)
        self.assertNotIn(100, [row.tmdb_id for row in result.recommendations])

    def test_local_bundle_keeps_run_id_without_a_registry_version(self):
        with tempfile.TemporaryDirectory(prefix="service-bundle-") as directory:
            self.assertEqual(
                Path(directory).resolve().parent, Path(tempfile.gettempdir()).resolve()
            )
            bundle = Path(directory) / "model"
            self.model.save(bundle, metadata={"run_id": "a" * 32})
            with patch(
                "src.api.service.load_recommender", wraps=load_recommender
            ) as loader:
                service = RecommendationService.from_source(bundle)
                for favorites in ([], [100], [999]):
                    service.recommend(favorites, n=3)
            self.assertEqual(loader.call_count, 1)
            self.assertIsNone(service.model_info.name)
            self.assertIsNone(service.model_info.version)
            self.assertEqual(service.model_info.run_id, "a" * 32)
            self.assertEqual(service.model_info.model_uri, str(bundle.resolve()))

    def test_source_passes_tracking_and_registry_settings_to_loader(self):
        with patch(
            "src.api.service.load_recommender", return_value=self.loaded
        ) as loader:
            service = RecommendationService.from_source(
                self.loaded.model_uri,
                tracking_uri="tracking",
                registry_uri="registry",
            )
        loader.assert_called_once_with(
            self.loaded.model_uri,
            tracking_uri="tracking",
            registry_uri="registry",
        )
        self.assertEqual(service.model_info.version, "7")

    def test_model_identity_cannot_be_changed_by_a_caller(self):
        with self.assertRaises(ValidationError):
            self.service.model_info.version = "8"
        self.assertEqual(self.service.recommend([], n=1).model.version, "7")

    def test_concurrent_services_match_sequential_results_without_mutating_factors(
        self,
    ):
        profiles = [[], [100], [200], [999], [100, 300], [400], [500, 600], [100, 100]]
        expected = [
            self.service.recommend(profile, n=3).model_dump() for profile in profiles
        ]
        users = self.model.model.user_factors.copy()
        items = self.model.model.item_factors.copy()
        other = RecommendationService(self.loaded)
        barrier, guard = Barrier(len(profiles)), Lock()
        active = peak = 0
        original = self.model.recommend_with_method

        def measured(favorites, n=10):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.002)
                return original(favorites, n=n)
            finally:
                with guard:
                    active -= 1

        def request(index):
            barrier.wait(timeout=10)
            service = self.service if index % 2 else other
            return service.recommend(profiles[index], n=3).model_dump()

        with patch.object(self.model, "recommend_with_method", side_effect=measured):
            with ThreadPoolExecutor(max_workers=len(profiles)) as pool:
                actual = list(pool.map(request, range(len(profiles))))
        self.assertEqual(actual, expected)
        self.assertEqual(peak, 1)
        np.testing.assert_array_equal(self.model.model.user_factors, users)
        np.testing.assert_array_equal(self.model.model.item_factors, items)

    def test_invalid_model_scores_are_rejected(self):
        for score in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(score=score):
                result = {
                    "method": "als",
                    "recommendations": [{"tmdb_id": 200, "score": score}],
                }
                with patch.object(
                    self.model, "recommend_with_method", return_value=result
                ):
                    with self.assertRaises(ValidationError):
                        self.service.recommend([100], n=1)

    def test_requests_do_not_train_the_model(self):
        with patch.object(
            ImplicitRecommender,
            "fit",
            side_effect=AssertionError("Unexpected training"),
        ):
            self.assertTrue(self.service.recommend([100], n=3).recommendations)


if __name__ == "__main__":
    unittest.main()

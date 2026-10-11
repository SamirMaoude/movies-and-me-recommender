"""Small CPU ALS tests for favorites, matrix orientation and stateless requests."""

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from implicit.cpu.als import AlternatingLeastSquares
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, isspmatrix_csr

from src.models.collaborative import (
    ALSConfig,
    ImplicitRecommender,
    build_interaction_matrix,
)


def sample_links():
    return pd.DataFrame({
        "movieId": [60, 30, 20, 10, 40, 50, 70],
        "tmdbId": [600, 100, 200, 100, 300, 400, 500],
    })


def sample_train():
    """Three positive users, five films, one alias and a negative-only user."""
    return pd.DataFrame([
        (19, 50, 400, True),
        (2, 30, 100, True),
        (7, 40, 300, True),
        (2, 20, 200, True),
        (7, 10, 100, True),
        (19, 70, 500, True),
        (2, 10, 100, True),
        (99, 60, 600, False),
        (7, 20, 200, False),
    ], columns=["userId", "movieId", "tmdbId", "is_positive"])


def small_config():
    return ALSConfig(factors=4, regularization=0.1, alpha=20.0,
                     iterations=3, random_state=42, num_threads=1)


EXPECTED_MATRIX = np.array([
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1],
], dtype=np.float32)


class InteractionMatrixTests(unittest.TestCase):
    def test_users_are_rows_films_are_columns_and_aliases_count_once(self):
        data = build_interaction_matrix(sample_train(), sample_links())

        self.assertEqual(data.user_ids, (2, 7, 19))
        self.assertEqual(data.tmdb_ids, (100, 200, 300, 400, 500))
        self.assertEqual(data.movie_ids, (10, 20, 40, 50, 70))
        self.assertTrue(isspmatrix_csr(data.matrix))
        self.assertEqual(data.matrix.dtype, np.float32)
        self.assertEqual(data.matrix.nnz, 6)
        np.testing.assert_array_equal(data.matrix.toarray(), EXPECTED_MATRIX)

    def test_ratings_timestamps_duplicates_and_row_order_cannot_change_feedback(self):
        train = sample_train()
        train = pd.concat([train, train.iloc[[1, 1, 6]]], ignore_index=True)
        train = train.sample(frac=1, random_state=91)
        train["rating"] = "unused"
        train["timestamp"] = np.nan
        data = build_interaction_matrix(train, sample_links().iloc[::-1])

        np.testing.assert_array_equal(data.matrix.toarray(), EXPECTED_MATRIX)
        self.assertEqual(data.tmdb_ids, (100, 200, 300, 400, 500))

    def test_canonical_movie_can_be_absent_from_training_rows(self):
        train = sample_train()
        train = train.loc[train["movieId"].ne(10)]
        data = build_interaction_matrix(train, sample_links())

        self.assertEqual(data.movie_ids[0], 10)
        self.assertEqual(data.matrix[0, 0], 1)

    def test_inputs_are_snapshotted_without_mutation(self):
        train, links = sample_train(), sample_links()
        original_train, original_links = train.copy(deep=True), links.copy(deep=True)
        data = build_interaction_matrix(train, links)

        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(links, original_links)
        train["is_positive"] = False
        links["tmdbId"] = 999
        np.testing.assert_array_equal(data.matrix.toarray(), EXPECTED_MATRIX)
        self.assertEqual(data.tmdb_ids, (100, 200, 300, 400, 500))

    def test_empty_or_negative_only_training_is_rejected(self):
        negative = sample_train().copy()
        negative["is_positive"] = False
        for train in (sample_train().iloc[:0], negative):
            with self.subTest(rows=len(train)), self.assertRaises(ValueError):
                build_interaction_matrix(train, sample_links())
            with self.subTest(fit_rows=len(train)), self.assertRaises(ValueError):
                ImplicitRecommender.fit(train, sample_links(), small_config())

    def test_schema_ids_and_feedback_are_validated_before_filtering(self):
        for column in ("userId", "movieId", "tmdbId", "is_positive"):
            with self.subTest(missing=column), self.assertRaises(ValueError):
                build_interaction_matrix(sample_train().drop(columns=column), sample_links())
            duplicate = pd.concat([sample_train(), sample_train()[[column]]], axis=1)
            with self.subTest(duplicate=column), self.assertRaises(ValueError):
                build_interaction_matrix(duplicate, sample_links())
        invalid = {
            "userId": [0, -1, True, 2.0, "2", None, pd.NA],
            "movieId": [0, -1, True, 60.0, "60", None, pd.NA],
            "tmdbId": [0, -1, True, 600.0, "600", None, pd.NA],
            "is_positive": [0, 1, "False", None, pd.NA],
        }
        for column, values in invalid.items():
            for value in values:
                train = sample_train().astype(object)
                train.loc[7, column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    build_interaction_matrix(train, sample_links())

    def test_inconsistent_or_unknown_movie_associations_are_rejected(self):
        for movie_id, tmdb_id in ((60, 100), (999, 600), (60, 999)):
            train = sample_train().copy()
            train.loc[7, ["movieId", "tmdbId"]] = [movie_id, tmdb_id]
            with self.subTest(movie_id=movie_id, tmdb_id=tmdb_id), self.assertRaises(ValueError):
                build_interaction_matrix(train, sample_links())


class ALSConfigurationTests(unittest.TestCase):
    def test_defaults_are_frozen_and_fit_can_receive_an_explicit_configuration(self):
        config = ALSConfig()
        self.assertEqual(
            (config.factors, config.regularization, config.alpha,
             config.iterations, config.random_state, config.num_threads),
            (32, 0.1, 20.0, 30, 42, 1),
        )
        with self.assertRaises(FrozenInstanceError):
            config.alpha = 1.0

    def test_invalid_training_parameters_fail_before_calling_the_library(self):
        invalid = {
            "factors": [0, -1, True, 4.0, "4"],
            "iterations": [0, -1, True, 3.0],
            "num_threads": [0, -1, True, 1.0],
            "random_state": [-1, True, 42.0, None, 2 ** 32],
            "regularization": [0, -0.1, True, "0.1", np.nan, np.inf],
            "alpha": [0, -1, True, "20", np.nan, np.inf],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ALSConfig(**{name: value})


class ImplicitRecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = ImplicitRecommender.fit(sample_train(), sample_links(), small_config())

    def test_real_cpu_als_receives_users_by_items_and_confidence_is_applied_once(self):
        config = small_config()
        reference = AlternatingLeastSquares(
            factors=config.factors, regularization=config.regularization,
            alpha=config.alpha, iterations=config.iterations,
            random_state=config.random_state, num_threads=config.num_threads,
            dtype=np.float32,
        )
        reference.fit(csr_matrix(EXPECTED_MATRIX), show_progress=False)

        self.assertEqual(self.recommender.model.user_factors.shape, (3, 4))
        self.assertEqual(self.recommender.model.item_factors.shape, (5, 4))
        np.testing.assert_allclose(self.recommender.model.user_factors,
                                   reference.user_factors, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(self.recommender.model.item_factors,
                                   reference.item_factors, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(self.recommender.training_data.matrix.toarray(), EXPECTED_MATRIX)

    def test_request_scores_match_an_independent_favorites_only_least_squares_solution(self):
        recommender = self.recommender
        factors = recommender.model.item_factors.astype(np.float64)
        favorites = [100, 300]
        preference = np.array([1, 0, 1, 0, 0], dtype=np.float64)
        confidence = np.where(preference, recommender.config.alpha, 1.0)
        system = factors.T @ (confidence[:, None] * factors)
        system += recommender.config.regularization * np.eye(recommender.config.factors)
        user_factor = np.linalg.solve(system, factors.T @ (confidence * preference))
        scores = factors @ user_factor
        expected = sorted(
            ((tmdb_id, scores[index]) for index, tmdb_id in
             enumerate(recommender.training_data.tmdb_ids) if tmdb_id not in favorites),
            key=lambda pair: (-pair[1], pair[0]),
        )

        actual = recommender.recommend(favorites, n=10)

        self.assertEqual([row["tmdb_id"] for row in actual], [pair[0] for pair in expected])
        np.testing.assert_allclose([row["score"] for row in actual],
                                   [pair[1] for pair in expected], rtol=1e-4, atol=1e-5)

    def test_known_favorites_use_als_unknown_or_empty_profiles_use_popularity(self):
        expected = [
            {"tmdb_id": 100, "score": 2.0},
            {"tmdb_id": 200, "score": 1.0},
            {"tmdb_id": 300, "score": 1.0},
            {"tmdb_id": 400, "score": 1.0},
            {"tmdb_id": 500, "score": 1.0},
        ]
        for favorites in ([], [999], [600], (value for value in ())):
            result = self.recommender.recommend_with_method(favorites, n=10)
            self.assertEqual(result, {"method": "popularity", "recommendations": expected})
        actual = self.recommender.recommend_with_method([100], n=10)
        self.assertEqual(actual["method"], "als")
        self.assertEqual(actual["recommendations"], self.recommender.recommend([100], n=10))

    def test_exclusions_duplicates_unknown_ids_and_limits(self):
        expected = self.recommender.recommend([100, 300], n=10)
        self.assertEqual(len(expected), 3)
        self.assertEqual({row["tmdb_id"] for row in expected}, {200, 400, 500})
        for favorites in ([300, 100, 999, 100, 300],
                          (value for value in [np.int64(300), np.uint32(100), 999])):
            self.assertEqual(self.recommender.recommend(favorites), expected)
        self.assertEqual(self.recommender.recommend([100, 300], n=np.int64(2)), expected[:2])
        self.assertEqual(self.recommender.recommend([100, 200, 300, 400, 500]), [])

    def test_two_successive_profiles_cannot_modify_training_factors_or_each_other(self):
        user_factors = self.recommender.model.user_factors.copy()
        item_factors = self.recommender.model.item_factors.copy()
        first_favorites = [100, 200]
        first = self.recommender.recommend(first_favorites)

        second = self.recommender.recommend([400, 500])
        repeated = self.recommender.recommend(first_favorites)

        self.assertTrue(second)
        self.assertEqual(repeated, first)
        self.assertEqual(first_favorites, [100, 200])
        np.testing.assert_array_equal(self.recommender.model.user_factors, user_factors)
        np.testing.assert_array_equal(self.recommender.model.item_factors, item_factors)

    def test_result_mutation_does_not_affect_future_requests_and_values_are_json_native(self):
        expected = self.recommender.recommend([100])
        result = self.recommender.recommend([100])
        for row in result:
            self.assertIs(type(row["tmdb_id"]), int)
            self.assertIs(type(row["score"]), float)
            self.assertTrue(np.isfinite(row["score"]))
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        result[0]["tmdb_id"] = 999
        result[0]["score"] = np.nan
        self.assertEqual(self.recommender.recommend([100]), expected)

    def test_invalid_favorite_ids_and_limits_are_rejected(self):
        invalid_favorites = (None, 100, "100", b"100", [True], [0], [-1],
                             [100.0], ["100"], [None], [pd.NA], [np.nan], [[100]])
        for favorites in invalid_favorites:
            with self.subTest(favorites=favorites), self.assertRaises(ValueError):
                self.recommender.recommend(favorites)
        for n in (0, -1, True, 1.0, "1", None, np.inf):
            with self.subTest(n=n), self.assertRaises(ValueError):
                self.recommender.recommend([], n=n)


class ALSArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = ImplicitRecommender.fit(sample_train(), sample_links(), small_config())

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.bundle = Path(temporary.name) / "als"
        self.recommender.save(self.bundle, metadata={"purpose": "unit test"})

    def read_json(self, name):
        return json.loads((self.bundle / name).read_text(encoding="utf-8"))

    def write_json(self, name, payload):
        (self.bundle / name).write_text(json.dumps(payload), encoding="utf-8")

    def update_checksum(self, name):
        manifest = self.read_json("manifest.json")
        manifest["sha256"][name] = hashlib.sha256((self.bundle / name).read_bytes()).hexdigest()
        self.write_json("manifest.json", manifest)

    def test_round_trip_preserves_mappings_matrix_scores_and_fallback_without_csv_files(self):
        self.assertEqual(
            {path.name for path in self.bundle.iterdir()},
            {"model.npz", "interactions.npz", "catalog.json", "manifest.json"},
        )
        restored = ImplicitRecommender.load(self.bundle)

        self.assertEqual(restored.config, self.recommender.config)
        self.assertEqual(restored.metadata, {"purpose": "unit test"})
        self.assertEqual(restored.training_data.user_ids, (2, 7, 19))
        self.assertEqual(restored.training_data.tmdb_ids, (100, 200, 300, 400, 500))
        self.assertEqual(restored.training_data.movie_ids, (10, 20, 40, 50, 70))
        np.testing.assert_array_equal(restored.training_data.matrix.toarray(), EXPECTED_MATRIX)
        for favorites in ([], [999], [100], [100, 300], [100, 200, 300, 400, 500]):
            with self.subTest(favorites=favorites):
                expected = self.recommender.recommend_with_method(favorites)
                actual = restored.recommend_with_method(favorites)
                self.assertEqual(actual["method"], expected["method"])
                self.assertEqual([row["tmdb_id"] for row in actual["recommendations"]],
                                 [row["tmdb_id"] for row in expected["recommendations"]])
                np.testing.assert_allclose(
                    [row["score"] for row in actual["recommendations"]],
                    [row["score"] for row in expected["recommendations"]],
                    rtol=1e-6, atol=1e-7,
                )

    def test_save_refuses_existing_directories_without_changing_their_contents(self):
        before = {path.name: path.read_bytes() for path in self.bundle.iterdir()}
        with self.assertRaises(FileExistsError):
            self.recommender.save(self.bundle)
        self.assertEqual({path.name: path.read_bytes() for path in self.bundle.iterdir()}, before)
        empty_directory = self.bundle.parent / "empty"
        empty_directory.mkdir()
        with self.assertRaises(FileExistsError):
            self.recommender.save(empty_directory)
        self.assertEqual(list(empty_directory.iterdir()), [])

    def test_metadata_must_be_finite_json_and_invalid_metadata_leaves_no_bundle(self):
        for index, metadata in enumerate(([], {"bad": np.nan}, {"bad": object()})):
            target = self.bundle.parent / f"bad-metadata-{index}"
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.recommender.save(target, metadata=metadata)
            self.assertFalse(target.exists())

    def test_missing_manifest_or_data_file_is_rejected(self):
        for name in ("manifest.json", "model.npz", "interactions.npz", "catalog.json"):
            path = self.bundle / name
            original = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaises(ValueError):
                ImplicitRecommender.load(self.bundle)
            path.write_bytes(original)

    def test_changed_data_files_are_rejected_by_their_checksums(self):
        for name in ("model.npz", "interactions.npz", "catalog.json"):
            path = self.bundle / name
            original = path.read_bytes()
            path.write_bytes(original + b"corruption")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "checksum"):
                ImplicitRecommender.load(self.bundle)
            path.write_bytes(original)

    def test_missing_metadata_and_unsupported_format_are_rejected(self):
        original = self.read_json("manifest.json")
        for key in ("config", "dimensions", "metadata", "implicit_version"):
            manifest = dict(original)
            del manifest[key]
            self.write_json("manifest.json", manifest)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                ImplicitRecommender.load(self.bundle)
        manifest = dict(original)
        manifest["format_version"] = 999
        self.write_json("manifest.json", manifest)
        with self.assertRaises(ValueError):
            ImplicitRecommender.load(self.bundle)

    def test_dimension_mismatch_is_rejected_after_checksum_verification(self):
        manifest = self.read_json("manifest.json")
        manifest["dimensions"]["items"] += 1
        self.write_json("manifest.json", manifest)

        with self.assertRaisesRegex(ValueError, "dimensions"):
            ImplicitRecommender.load(self.bundle)

    def test_invalid_factor_shape_or_nonfinite_values_are_rejected_even_with_updated_checksum(self):
        path = self.bundle / "model.npz"
        original = path.read_bytes()
        for corruption in ("shape", "nan"):
            path.write_bytes(original)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            if corruption == "shape":
                arrays["item_factors"] = arrays["item_factors"][:-1]
            else:
                arrays["item_factors"][0, 0] = np.nan
            np.savez(path, **arrays)
            self.update_checksum("model.npz")
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, "item_factors"):
                ImplicitRecommender.load(self.bundle)

    def test_invalid_external_ids_are_rejected_even_with_updated_checksum(self):
        original = self.read_json("catalog.json")
        for invalid_ids in ([100, 100, 300, 400, 500],
                            [100, 200, 300, 400, 0],
                            [100, 200, 300, 400, 500.0],
                            [100, 200, 300, 400, True],
                            [100, 300, 200, 400, 500]):
            catalog = dict(original, tmdb_ids=invalid_ids)
            self.write_json("catalog.json", catalog)
            self.update_checksum("catalog.json")
            with self.subTest(ids=invalid_ids), self.assertRaises(ValueError):
                ImplicitRecommender.load(self.bundle)

    def test_fallback_counts_must_match_the_saved_training_matrix(self):
        catalog = self.read_json("catalog.json")
        catalog["popularity"][0][1] += 1
        self.write_json("catalog.json", catalog)
        self.update_checksum("catalog.json")

        with self.assertRaisesRegex(ValueError, "popularity"):
            ImplicitRecommender.load(self.bundle)


if __name__ == "__main__":
    unittest.main()

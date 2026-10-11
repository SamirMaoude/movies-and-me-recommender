"""Hand-calculated ranking checks and leakage guards for model comparison."""

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.evaluation.compare import (
    ValidationCase,
    build_report,
    build_validation_cases,
    compare_models,
    evaluate_recommender,
)


def interactions(rows):
    return pd.DataFrame(
        rows,
        columns=["userId", "movieId", "tmdbId", "rating", "timestamp", "is_positive"],
    )


def small_splits():
    train = interactions(
        [
            (2, 20, 200, 4.0, 10, True),
            (2, 30, 300, 5.0, 11, True),
            (1, 10, 100, 4.5, 12, True),
            (1, 30, 300, 1.0, 13, False),
            (3, 40, 400, 2.0, 14, False),
            (4, 40, 400, 5.0, 15, True),
            (5, 50, 500, 4.0, 16, True),
        ]
    )
    validation = interactions(
        [
            (2, 10, 100, 5.0, 100, True),
            (2, 99, 999, 4.0, 101, True),
            (1, 20, 200, 4.5, 102, True),
            (1, 60, 600, 1.0, 103, False),
            (3, 10, 100, 4.0, 104, True),
            (4, 99, 999, 5.0, 105, True),
            (6, 10, 100, 4.0, 106, True),
        ]
    )
    return train, validation


def metadata():
    movies = pd.DataFrame(
        [
            (10, "First", "Drama"),
            (20, "Second", "Comedy"),
            (30, "Third", "Drama|Comedy"),
            (40, "Fourth", "Action"),
            (50, "Fifth", "Drama"),
            (60, "Not positive", "FutureGenre"),
            (99, "Future only", "FutureGenre"),
        ],
        columns=["movieId", "title", "genres"],
    )
    links = pd.DataFrame(
        [(10, 100), (20, 200), (30, 300), (40, 400), (50, 500), (60, 600), (99, 999)],
        columns=["movieId", "tmdbId"],
    )
    return movies, links


class FixedRanking:
    def __init__(self, rankings):
        self.rankings = rankings
        self.calls = []

    def recommend(self, favorites, n=10):
        favorites = tuple(favorites)
        self.calls.append((favorites, n))
        ranking = self.rankings[favorites]
        return [
            {"tmdb_id": movie_id, "score": float(len(ranking) - position)}
            for position, movie_id in enumerate(ranking)
        ]


class ValidationCaseTests(unittest.TestCase):
    def test_common_cohort_uses_train_favorites_and_keeps_unavailable_targets(self):
        train, validation = small_splits()

        candidate_ids, cases = build_validation_cases(train, validation)

        self.assertEqual(candidate_ids, (100, 200, 300, 400, 500))
        self.assertEqual(
            cases,
            [
                ValidationCase(1, (100,), (200,)),
                ValidationCase(2, (200, 300), (100, 999)),
            ],
        )
        # User 3 has only a low training rating, 4 only unavailable targets,
        # 5 no validation target, and 6 no training history.
        self.assertEqual([case.user_id for case in cases], [1, 2])

    def test_cases_do_not_depend_on_row_order_and_leave_inputs_unchanged(self):
        train, validation = small_splits()
        original_train, original_validation = (
            train.copy(deep=True),
            validation.copy(deep=True),
        )
        expected = build_validation_cases(train, validation)

        self.assertEqual(
            build_validation_cases(
                train.sample(frac=1, random_state=4),
                validation.sample(frac=1, random_state=8),
            ),
            expected,
        )
        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(validation, original_validation)

    def test_no_positive_train_catalog_or_no_eligible_users_produces_no_cases(self):
        train, validation = small_splits()
        no_positives = train.copy()
        no_positives["is_positive"] = False
        candidate_ids, cases = build_validation_cases(no_positives, validation)
        self.assertEqual(candidate_ids, ())
        self.assertEqual(cases, [])
        self.assertEqual(build_validation_cases(train, validation.iloc[:0])[1], [])


class RankingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = (100, 200, 300, 400, 500)
        self.cases = [
            ValidationCase(1, (100,), (200, 300, 999)),
            ValidationCase(2, (400,), (100,)),
        ]

    def test_macro_metrics_have_separate_denominators_for_available_and_all_targets(
        self,
    ):
        recommender = FixedRanking({(100,): [400, 200, 500], (400,): [100]})

        result = evaluate_recommender(recommender, self.cases, self.catalog)

        rank_two_discount = 1 / math.log2(3)
        user_one_ndcg = rank_two_discount / (1 + rank_two_discount)
        user_one_ndcg_all = rank_two_discount / (1 + rank_two_discount + 0.5)
        self.assertAlmostEqual(result["metrics"]["recall_at_10"], 0.75)
        self.assertAlmostEqual(result["metrics"]["ndcg_at_10"], (user_one_ndcg + 1) / 2)
        self.assertAlmostEqual(result["metrics_all_targets"]["recall_at_10"], 2 / 3)
        self.assertAlmostEqual(
            result["metrics_all_targets"]["ndcg_at_10"], (user_one_ndcg_all + 1) / 2
        )
        self.assertEqual(result["users_evaluated"], 2)
        self.assertEqual(result["empty_recommendation_users"], 0)
        self.assertEqual(result["mean_recommendations"], 2.0)
        self.assertEqual(recommender.calls, [((100,), 10), ((400,), 10)])
        first, second = result["per_user"]
        self.assertEqual(first["user_id"], 1)
        self.assertEqual(first["favorite_count"], 1)
        self.assertEqual(first["target_count"], 3)
        self.assertEqual(first["target_count_in_catalog"], 2)
        self.assertEqual(first["target_count_outside_catalog"], 1)
        self.assertEqual(first["recommended_tmdb_ids"], [400, 200, 500])
        self.assertEqual(first["hits_at_10"], 1)
        self.assertAlmostEqual(first["recall_at_10"], 0.5)
        self.assertAlmostEqual(first["ndcg_at_10"], user_one_ndcg)
        self.assertAlmostEqual(first["recall_at_10_all_targets"], 1 / 3)
        self.assertAlmostEqual(first["ndcg_at_10_all_targets"], user_one_ndcg_all)
        self.assertEqual(second["hits_at_10"], 1)
        self.assertEqual(second["recall_at_10"], 1.0)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_empty_rankings_are_scored_zero_for_every_eligible_user(self):
        for recommender in (None, FixedRanking({(100,): [], (400,): []})):
            with self.subTest(recommender=recommender):
                result = evaluate_recommender(recommender, self.cases, self.catalog)
                self.assertEqual(
                    result["metrics"], {"recall_at_10": 0.0, "ndcg_at_10": 0.0}
                )
                self.assertEqual(result["metrics_all_targets"], result["metrics"])
                self.assertEqual(result["empty_recommendation_users"], 2)
                self.assertEqual(result["mean_recommendations"], 0.0)
                self.assertEqual(
                    [row["hits_at_10"] for row in result["per_user"]], [0, 0]
                )

    def test_no_eligible_users_is_null_rather_than_perfect_or_zero_quality(self):
        result = evaluate_recommender(None, [], self.catalog)
        self.assertEqual(result["metrics"], {"recall_at_10": None, "ndcg_at_10": None})
        self.assertEqual(result["metrics_all_targets"], result["metrics"])
        self.assertEqual(result["users_evaluated"], 0)
        self.assertEqual(result["per_user"], [])
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_k_changes_the_request_and_limits_the_ideal_ranking(self):
        recommender = FixedRanking({(100,): [200]})

        result = evaluate_recommender(recommender, self.cases[:1], self.catalog, k=1)

        self.assertEqual(recommender.calls, [((100,), 1)])
        self.assertEqual(result["metrics"], {"recall_at_1": 0.5, "ndcg_at_1": 1.0})
        self.assertAlmostEqual(result["metrics_all_targets"]["recall_at_1"], 1 / 3)
        self.assertEqual(result["metrics_all_targets"]["ndcg_at_1"], 1.0)


class ModelComparisonTests(unittest.TestCase):
    def test_both_models_use_the_same_users_favorites_and_candidate_catalog(self):
        train, validation = small_splits()
        movies, links = metadata()
        favorites_seen = {"popularity": [], "genres": []}

        class RecordingPopularity:
            def __init__(self, supplied_train):
                pd.testing.assert_frame_equal(supplied_train, train)

            def recommend(self, favorites, n=10):
                favorites_seen["popularity"].append(tuple(favorites))
                return []

        class RecordingGenres:
            def __init__(self, recommender, mapper):
                pass

            def recommend(self, favorites, n=10):
                favorites_seen["genres"].append(tuple(favorites))
                return []

        with (
            patch("src.evaluation.compare.PopularityRecommender", RecordingPopularity),
            patch(
                "src.evaluation.compare.TMDBRecommender",
                RecordingGenres,
            ),
        ):
            result = compare_models(train, validation, movies, links)

        self.assertEqual(favorites_seen["popularity"], [(100,), (200, 300)])
        self.assertEqual(favorites_seen["genres"], favorites_seen["popularity"])
        self.assertEqual(result["coverage"]["candidate_movies"], 5)
        self.assertEqual(result["coverage"]["positive_targets"], 6)
        self.assertEqual(result["coverage"]["eligible_users"], 2)
        self.assertEqual(result["coverage"]["eligible_positive_targets"], 2)
        self.assertAlmostEqual(result["coverage"]["eligible_target_coverage"], 2 / 6)
        self.assertEqual(
            result["coverage"]["eligible_positive_targets_outside_catalog"], 1
        )
        for model in result["models"].values():
            self.assertEqual([row["user_id"] for row in model["per_user"]], [1, 2])

    def test_low_ratings_do_not_exclude_candidates_and_validation_cannot_train_popularity(
        self,
    ):
        train = interactions(
            [
                (1, 10, 100, 5.0, 10, True),
                (1, 20, 200, 1.0, 11, False),
                (2, 20, 200, 4.0, 12, True),
                (3, 30, 300, 4.0, 13, True),
            ]
        )
        validation = interactions(
            [(1, 30, 300, 4.0, 100, True)]
            + [(user, 30, 300, 5.0, 100 + user, True) for user in range(10, 20)]
        )
        movies, links = metadata()

        result = compare_models(train, validation, movies, links, k=1)

        row = result["models"]["popularity"]["per_user"][0]
        # 200 and 300 tie in train. 200 stays eligible despite user's low note;
        # the many validation positives for 300 cannot move it ahead of 200.
        self.assertEqual(row["recommended_tmdb_ids"], [200])
        self.assertEqual(row["favorite_count"], 1)
        self.assertEqual(row["hits_at_1"], 0)

    def test_genre_training_ignores_metadata_outside_positive_train_catalog(self):
        train, validation = small_splits()
        movies, links = metadata()
        expected = compare_models(train, validation, movies, links)
        altered = movies.copy()
        altered.loc[altered["movieId"].isin([60, 99]), "genres"] = "Drama|Comedy|Action"
        many_future_movies = pd.DataFrame(
            [(movie_id, "Future", "Drama") for movie_id in range(1000, 1030)],
            columns=movies.columns,
        )
        altered = pd.concat([altered, many_future_movies], ignore_index=True)

        actual = compare_models(train, validation, altered, links)

        self.assertEqual(actual["models"], expected["models"])

    def test_genre_profile_without_features_has_zero_metrics_and_no_popularity_fallback(
        self,
    ):
        train = interactions([(1, 10, 100, 4.0, 10, True), (2, 20, 200, 4.0, 11, True)])
        validation = interactions([(1, 20, 200, 4.0, 100, True)])
        movies, links = metadata()
        movies.loc[movies["movieId"].eq(10), "genres"] = "(no genres listed)"

        result = compare_models(train, validation, movies, links)

        genres = result["models"]["genres"]
        self.assertEqual(genres["users_evaluated"], 1)
        self.assertEqual(genres["empty_recommendation_users"], 1)
        self.assertEqual(genres["metrics"], {"recall_at_10": 0.0, "ndcg_at_10": 0.0})
        self.assertEqual(genres["per_user"][0]["recommended_tmdb_ids"], [])
        self.assertEqual(result["models"]["popularity"]["metrics"]["recall_at_10"], 1.0)

    def test_all_genres_missing_remains_an_empty_baseline(self):
        train, validation = small_splits()
        movies, links = metadata()
        movies["genres"] = "(no genres listed)"

        result = compare_models(train, validation, movies, links)

        self.assertEqual(result["models"]["genres"]["empty_recommendation_users"], 2)
        self.assertEqual(result["models"]["genres"]["metrics"]["recall_at_10"], 0.0)

    def test_alias_uses_canonical_metadata(self):
        train = interactions([(1, 10, 100, 4.0, 10, True), (2, 20, 200, 4.0, 11, True)])
        validation = interactions([(1, 20, 200, 4.0, 100, True)])
        movies = pd.DataFrame(
            [
                (10, "Canonical", "Drama"),
                (11, "Alias", "Comedy"),
                (20, "Target", "Drama"),
            ],
            columns=["movieId", "title", "genres"],
        )
        links = pd.DataFrame(
            [(10, 100), (11, 100), (20, 200)], columns=["movieId", "tmdbId"]
        )

        result = compare_models(train, validation, movies, links)

        self.assertEqual(result["models"]["genres"]["metrics"]["recall_at_10"], 1.0)

    def test_comparison_without_eligible_users_has_null_metrics_for_both_models(self):
        train, validation = small_splits()
        movies, links = metadata()

        result = compare_models(
            train, validation.loc[validation["userId"].eq(6)], movies, links
        )

        self.assertEqual(result["coverage"]["eligible_users"], 0)
        for model in result["models"].values():
            self.assertEqual(model["users_evaluated"], 0)
            self.assertIsNone(model["metrics"]["recall_at_10"])
            self.assertIsNone(model["metrics"]["ndcg_at_10"])


class ComparisonReportTests(unittest.TestCase):
    def setUp(self):
        temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temp_directory.cleanup)
        self.directory = Path(temp_directory.name)
        self.config_path = self.directory / "evaluation.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "positive_threshold": 4.0,
                    "train_end": "1970-01-01T00:01:40Z",
                    "validation_end": "1970-01-01T00:03:20Z",
                }
            ),
            encoding="utf-8",
        )
        files = {
            "movies.csv": "movieId,title,genres\n10,First,Drama\n20,Second,Drama\n30,Future,Comedy\n",
            "links.csv": "movieId,imdbId,tmdbId\n10,0000010,100\n20,0000020,200\n30,0000030,300\n",
            "ratings.csv": (
                "userId,movieId,rating,timestamp\n"
                "1,10,4.0,10\n2,20,4.0,20\n1,20,5.0,100\n3,30,5.0,200\n"
            ),
            "tags.csv": "userId,movieId,tag,timestamp\n",
        }
        for filename, contents in files.items():
            (self.directory / filename).write_text(contents, encoding="utf-8")

    def test_report_is_reproducible_json_with_three_source_hashes_and_no_file_changes(
        self,
    ):
        before = {path: path.read_bytes() for path in self.directory.iterdir()}

        report = build_report(self.directory, self.config_path)

        self.assertEqual(build_report(self.directory, self.config_path), report)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        for filename in ("ratings.csv", "links.csv", "movies.csv"):
            self.assertEqual(
                report["sources"][filename]["sha256"],
                hashlib.sha256(before[self.directory / filename]).hexdigest(),
            )
        self.assertEqual(
            {path: path.read_bytes() for path in self.directory.iterdir()}, before
        )
        for model in report["models"].values():
            self.assertEqual(model["users_evaluated"], 1)
            self.assertEqual(model["metrics"], {"recall_at_10": 1.0, "ndcg_at_10": 1.0})

    def test_changing_test_ratings_cannot_change_fitted_models_validation_cohort_or_scores(
        self,
    ):
        before = build_report(self.directory, self.config_path)
        ratings_path = self.directory / "ratings.csv"
        ratings_path.write_text(
            ratings_path.read_text(encoding="utf-8")
            + "4,30,5.0,201\n5,20,4.0,202\n6,10,1.0,203\n",
            encoding="utf-8",
        )

        after = build_report(self.directory, self.config_path)

        self.assertEqual(after["models"], before["models"])
        self.assertEqual(after["coverage"], before["coverage"])
        self.assertNotEqual(
            after["sources"]["ratings.csv"], before["sources"]["ratings.csv"]
        )


if __name__ == "__main__":
    unittest.main()

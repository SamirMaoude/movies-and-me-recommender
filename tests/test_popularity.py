"""Synthetic tests for distinct-user popularity recommendations."""

import json
import unittest

import numpy as np
import pandas as pd

from src.models.popularity import PopularityRecommender


EXPECTED_GLOBAL = [
    {"tmdb_id": 100, "score": 3.0},
    {"tmdb_id": 200, "score": 2.0},
    {"tmdb_id": 300, "score": 2.0},
    {"tmdb_id": 400, "score": 1.0},
]


def sample_train():
    """Repeated movie aliases share TMDB IDs; ratings are irrelevant here."""
    return pd.DataFrame(
        [
            (1, 100, True, 10, 5.0, 100),
            (1, 100, True, 30, 5.0, 101),
            (2, 100, True, 10, 4.0, 102),
            (3, 100, True, 30, 4.0, 103),
            (1, 200, True, 20, 4.0, 104),
            (2, 200, True, 20, 4.0, 105),
            (1, 300, True, 50, 1.0, 106),
            (3, 300, True, 50, 1.0, 107),
            (4, 400, True, 60, 4.5, 108),
            (9, 100, False, 10, 5.0, 109),
            (8, 200, False, 20, 5.0, 110),
            (5, 500, False, 70, 5.0, 111),
        ],
        columns=["userId", "tmdbId", "is_positive", "movieId", "rating", "timestamp"],
        index=[8, 4, 2, 9, 1, 7, 6, 3, 5, 11, 10, 0],
    )


def empty_train():
    return sample_train().iloc[:0].copy()


class PopularityScoringTests(unittest.TestCase):
    def test_counts_distinct_positive_users_and_breaks_ties_by_tmdb_id(self):
        model = PopularityRecommender(sample_train())

        self.assertEqual(model.recommend([]), EXPECTED_GLOBAL)

    def test_repeated_events_and_movie_aliases_cannot_increase_popularity(self):
        train = sample_train()
        duplicated = pd.concat([train, train.iloc[[0, 1, 3, 3, 3]]], ignore_index=True)

        self.assertEqual(PopularityRecommender(duplicated).recommend([]), EXPECTED_GLOBAL)

    def test_other_feedback_and_extra_rating_or_timestamp_values_do_not_change_scores(self):
        train = sample_train()
        other_interactions = pd.DataFrame(
            {
                "userId": range(1000, 1100),
                "tmdbId": [500] * 50 + [100] * 50,
                "is_positive": [False] * 100,
            }
        )
        train = pd.concat([train, other_interactions], ignore_index=True)
        train["rating"] = "not a rating"
        train["timestamp"] = np.nan

        self.assertEqual(PopularityRecommender(train).recommend([]), EXPECTED_GLOBAL)

    def test_excludes_favorites_before_applying_the_limit(self):
        model = PopularityRecommender(sample_train())

        self.assertEqual(model.recommend([100, 200], n=2), EXPECTED_GLOBAL[2:])
        self.assertEqual(model.recommend([100], n=1), EXPECTED_GLOBAL[1:2])
        self.assertEqual(model.recommend([100, 200, 300, 400]), [])

    def test_limit_accepts_positive_python_and_numpy_integers(self):
        model = PopularityRecommender(sample_train())
        for limit in (1, 3, 4, 20, np.int64(2), np.uint32(3)):
            with self.subTest(limit=limit):
                self.assertEqual(model.recommend([], n=limit), EXPECTED_GLOBAL[:int(limit)])

    def test_empty_or_unknown_favorites_return_the_global_ranking(self):
        model = PopularityRecommender(sample_train())
        for favorites in ([], (), iter(()), [999], [999, 888], [500]):
            with self.subTest(favorites=favorites):
                self.assertEqual(model.recommend(favorites), EXPECTED_GLOBAL)

    def test_favorite_duplicates_order_and_unknown_values_have_no_weight(self):
        model = PopularityRecommender(sample_train())
        self.assertEqual(
            model.recommend([300, 999, 100, 300, 100, 888]),
            model.recommend([100, 300]),
        )

    def test_generators_and_numpy_integer_favorites_are_supported(self):
        model = PopularityRecommender(sample_train())
        favorites = (value for value in (np.int64(100), np.uint32(300), np.int64(999)))

        self.assertEqual(model.recommend(favorites), [EXPECTED_GLOBAL[1], EXPECTED_GLOBAL[3]])

    def test_permuting_training_rows_cannot_change_the_ranking(self):
        shuffled = sample_train().sample(frac=1, random_state=29)

        self.assertEqual(PopularityRecommender(shuffled).recommend([]), EXPECTED_GLOBAL)

    def test_empty_training_or_no_positive_feedback_returns_no_candidates(self):
        other_feedback_train = sample_train().copy()
        other_feedback_train["is_positive"] = False
        for train in (empty_train(), other_feedback_train):
            with self.subTest(rows=len(train)):
                model = PopularityRecommender(train)
                self.assertEqual(model.recommend([]), [])
                self.assertEqual(model.recommend([100, 999]), [])

    def test_input_and_result_mutations_cannot_change_the_model_snapshot(self):
        train = sample_train()
        original = train.copy(deep=True)
        favorites = [100]
        model = PopularityRecommender(train)

        self.assertEqual(model.recommend(favorites), EXPECTED_GLOBAL[1:])
        self.assertEqual(favorites, [100])
        pd.testing.assert_frame_equal(train, original)
        train["is_positive"] = False
        train["tmdbId"] = 999
        result = model.recommend([])
        result[0]["score"] = -10.0
        result[0]["tmdb_id"] = 999
        result.append({"tmdb_id": 777, "score": 99.0})

        self.assertEqual(model.recommend([]), EXPECTED_GLOBAL)
        self.assertEqual(model.recommend([100]), EXPECTED_GLOBAL[1:])
        self.assertEqual(model.recommend([]), EXPECTED_GLOBAL)

    def test_results_use_native_json_types_and_preserve_counts_above_one(self):
        result = PopularityRecommender(sample_train()).recommend([])

        self.assertIs(type(result), list)
        for recommendation in result:
            self.assertIs(type(recommendation), dict)
            self.assertEqual(set(recommendation), {"tmdb_id", "score"})
            self.assertIs(type(recommendation["tmdb_id"]), int)
            self.assertIs(type(recommendation["score"]), float)
        self.assertEqual(result[0]["score"], 3.0)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)


class PopularityValidationTests(unittest.TestCase):
    def test_object_columns_with_integer_ids_and_real_booleans_are_accepted(self):
        train = pd.DataFrame(
            {
                "userId": pd.Series([np.int64(1), 2, 3], dtype=object),
                "tmdbId": pd.Series([np.uint32(100), 100, 200], dtype=object),
                "is_positive": pd.Series([True, np.bool_(True), np.bool_(False)], dtype=object),
            }
        )

        self.assertEqual(
            PopularityRecommender(train).recommend([]), [{"tmdb_id": 100, "score": 2.0}]
        )

    def test_invalid_favorites_are_rejected_even_without_candidates(self):
        invalid_favorites = (
            None, 100, 100.0, True, "", "100", b"", b"100", np.array(100),
            [0], [-1], [100.0], ["100"], [True], [np.bool_(False)],
            [None], [pd.NA], [np.nan], [[100]],
        )
        for train in (sample_train(), empty_train()):
            model = PopularityRecommender(train)
            for favorites in invalid_favorites:
                with self.subTest(rows=len(train), favorites=favorites):
                    with self.assertRaises(ValueError):
                        model.recommend(favorites)

    def test_invalid_limits_are_rejected_even_without_candidates(self):
        invalid_limits = (0, -1, True, np.bool_(False), 1.0, "1", None, np.nan, np.inf)
        for train in (sample_train(), empty_train()):
            model = PopularityRecommender(train)
            for limit in invalid_limits:
                with self.subTest(rows=len(train), limit=limit):
                    with self.assertRaises(ValueError):
                        model.recommend([], n=limit)

    def test_training_requires_the_schema_positive_integer_ids_and_real_booleans(self):
        for train in (None, [], {"userId": [1]}):
            with self.subTest(input_type=type(train)):
                with self.assertRaises(ValueError):
                    PopularityRecommender(train)
        for column in ("userId", "tmdbId", "is_positive"):
            with self.subTest(missing_column=column):
                with self.assertRaises(ValueError):
                    PopularityRecommender(sample_train().drop(columns=column))
            with self.subTest(duplicate_column=column):
                duplicate_column = pd.concat([sample_train(), sample_train()[[column]]], axis=1)
                with self.assertRaises(ValueError):
                    PopularityRecommender(duplicate_column)

        invalid_values = {
            "userId": [0, -1, 1.0, "1", True, np.bool_(False), None, pd.NA, np.nan],
            "tmdbId": [0, -1, 100.0, "100", True, None, pd.NA, np.inf],
            "is_positive": [1, 0, 1.0, "True", "False", None, pd.NA, np.nan],
        }
        for column, values in invalid_values.items():
            for value in values:
                with self.subTest(column=column, value=value):
                    row = {"userId": 1, "tmdbId": 100, "is_positive": True}
                    row[column] = value
                    with self.assertRaises(ValueError):
                        PopularityRecommender(pd.DataFrame([row], dtype=object))


if __name__ == "__main__":
    unittest.main()

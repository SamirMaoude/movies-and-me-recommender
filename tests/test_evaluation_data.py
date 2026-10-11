"""Synthetic regression tests for evaluation data preparation and time splits."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation.data import prepare_interactions, temporal_split

INTERACTION_COLUMNS = [
    "userId",
    "movieId",
    "tmdbId",
    "rating",
    "timestamp",
    "is_positive",
]


def sample_links():
    return pd.DataFrame(
        {
            "movieId": [30, 20, 10, 40, 50],
            "tmdbId": pd.array([100, 200, 100, None, 300], dtype="Int64"),
            "imdbId": ["0030", "0020", "0010", "0040", "0050"],
        }
    )


def sample_ratings():
    """An old negative alias, a future positive alias and an exact duplicate."""
    return pd.DataFrame(
        [
            (1, 30, 1.0, 100),
            (1, 10, 5.0, 300),
            (1, 20, 4.0, 200),
            (2, 30, 4.5, 200),
            (2, 20, 3.5, 200),
            (3, 50, 5.0, 400),
            (3, 40, 4.0, 100),
            (4, 999, 5.0, 100),
            (2, 30, 4.5, 200),
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
        index=[8, 3, 7, 1, 9, 4, 2, 6, 5],
    )


def split_fixture():
    """Unsorted interactions with multiple rows at both cutoff timestamps."""
    timestamps = [200, 100, 99, 200, 199, 100, 0]
    return pd.DataFrame(
        {
            "userId": range(1, 8),
            "movieId": range(10, 80, 10),
            "tmdbId": range(100, 800, 100),
            "rating": [4.0] * 7,
            "timestamp": timestamps,
            "is_positive": [True] * 7,
        },
        index=[8, 2, 7, 1, 6, 3, 5],
    )


def utc_timestamp(seconds):
    return pd.Timestamp(seconds, unit="s", tz="UTC")


class PrepareInteractionsTests(unittest.TestCase):
    def test_output_schema_counts_and_chronological_order(self):
        interactions, counts = prepare_interactions(sample_ratings(), sample_links())

        self.assertEqual(list(interactions.columns), INTERACTION_COLUMNS)
        expected = pd.DataFrame(
            [
                (1, 10, 100, 1.0, 100, False),
                (1, 20, 200, 4.0, 200, True),
                (2, 10, 100, 4.5, 200, True),
                (2, 20, 200, 3.5, 200, False),
                (3, 50, 300, 5.0, 400, True),
            ],
            columns=INTERACTION_COLUMNS,
        )
        pd.testing.assert_frame_equal(interactions, expected, check_dtype=False)
        self.assertEqual(
            counts,
            {
                "input_ratings": 9,
                "ratings_without_tmdb": 2,
                "ratings_with_tmdb": 7,
                "remapped_ratings": 3,
                "duplicate_user_movies_removed": 2,
                "prepared_interactions": 5,
                "positive_interactions": 3,
                "other_interactions": 2,
                "users": 3,
                "movies": 3,
                "ambiguous_tmdb_ids": 1,
            },
        )

    def test_canonical_alias_comes_from_links_even_if_it_has_no_rating(self):
        ratings = sample_ratings().iloc[[0]].copy()
        interactions, counts = prepare_interactions(ratings, sample_links())

        self.assertEqual(interactions["movieId"].tolist(), [10])
        self.assertEqual(interactions["tmdbId"].tolist(), [100])
        self.assertEqual(counts["remapped_ratings"], 1)

    def test_future_positive_alias_cannot_replace_an_older_negative_interaction(self):
        interactions, _ = prepare_interactions(sample_ratings(), sample_links())
        parts = temporal_split(interactions, utc_timestamp(150), utc_timestamp(250))

        old_interaction = parts["train"].loc[
            (parts["train"]["userId"] == 1) & (parts["train"]["tmdbId"] == 100)
        ]
        self.assertEqual(old_interaction["rating"].tolist(), [1.0])
        self.assertEqual(old_interaction["is_positive"].tolist(), [False])
        pair_sets = {
            name: set(zip(frame["userId"], frame["tmdbId"]))
            for name, frame in parts.items()
        }
        self.assertFalse(pair_sets["train"] & pair_sets["validation"])
        self.assertFalse(pair_sets["train"] & pair_sets["test"])
        self.assertFalse(pair_sets["validation"] & pair_sets["test"])
        self.assertNotIn((1, 100), pair_sets["test"])

    def test_same_timestamp_tie_uses_original_movie_id_instead_of_rating(self):
        ratings = pd.DataFrame(
            [(1, 30, 5.0, 100), (1, 10, 1.0, 100)],
            columns=["userId", "movieId", "rating", "timestamp"],
        )
        interactions, _ = prepare_interactions(ratings, sample_links())

        self.assertEqual(interactions["rating"].tolist(), [1.0])
        self.assertEqual(interactions["movieId"].tolist(), [10])
        self.assertEqual(interactions["is_positive"].tolist(), [False])

    def test_reordering_ratings_and_links_cannot_change_the_result(self):
        expected, expected_counts = prepare_interactions(
            sample_ratings(), sample_links()
        )
        actual, counts = prepare_interactions(
            sample_ratings().sample(frac=1, random_state=19),
            sample_links().sample(frac=1, random_state=23),
        )

        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(counts, expected_counts)

    def test_keeps_every_half_step_rating_and_applies_the_configured_threshold(self):
        ratings = pd.DataFrame(
            {
                "userId": range(1, 11),
                "movieId": [20] * 10,
                "rating": np.arange(0.5, 5.5, 0.5),
                "timestamp": range(10),
            }
        )
        interactions, counts = prepare_interactions(
            ratings, sample_links(), positive_threshold=4.5
        )

        self.assertEqual(interactions["rating"].tolist(), ratings["rating"].tolist())
        self.assertEqual(interactions["is_positive"].tolist(), [False] * 8 + [True] * 2)
        self.assertEqual(counts["positive_interactions"], 2)
        self.assertEqual(counts["other_interactions"], 8)

    def test_does_not_mutate_ratings_or_links(self):
        ratings, links = sample_ratings(), sample_links()
        original_ratings, original_links = (
            ratings.copy(deep=True),
            links.copy(deep=True),
        )

        prepare_interactions(ratings, links)

        pd.testing.assert_frame_equal(ratings, original_ratings)
        pd.testing.assert_frame_equal(links, original_links)

    def test_empty_or_entirely_unmapped_ratings_have_zero_prepared_interactions(self):
        for ratings in (sample_ratings().iloc[:0], sample_ratings().iloc[[6, 7]]):
            with self.subTest(input_rows=len(ratings)):
                interactions, counts = prepare_interactions(ratings, sample_links())
                self.assertTrue(interactions.empty)
                self.assertEqual(list(interactions.columns), INTERACTION_COLUMNS)
                self.assertEqual(counts["input_ratings"], len(ratings))
                self.assertEqual(counts["ratings_without_tmdb"], len(ratings))
                for key in (
                    "ratings_with_tmdb",
                    "remapped_ratings",
                    "duplicate_user_movies_removed",
                    "prepared_interactions",
                    "positive_interactions",
                    "other_interactions",
                    "users",
                    "movies",
                ):
                    self.assertEqual(counts[key], 0, key)


class TemporalSplitTests(unittest.TestCase):
    def test_exact_boundaries_and_timestamp_blocks_stay_together(self):
        parts = temporal_split(split_fixture(), utc_timestamp(100), utc_timestamp(200))

        self.assertEqual(set(parts), {"train", "validation", "test"})
        self.assertEqual(set(parts["train"]["userId"]), {3, 7})
        self.assertEqual(set(parts["validation"]["userId"]), {2, 5, 6})
        self.assertEqual(set(parts["test"]["userId"]), {1, 4})
        for name, expected_timestamps in (
            ("train", {0, 99}),
            ("validation", {100, 199}),
            ("test", {200}),
        ):
            self.assertEqual(set(parts[name]["timestamp"]), expected_timestamps)
        self.assertEqual(sum(map(len, parts.values())), len(split_fixture()))

    def test_iso_offsets_and_aware_timestamp_bounds_are_normalized_to_utc(self):
        expected = temporal_split(
            split_fixture(), "1970-01-01T00:01:40Z", "1970-01-01T00:03:20Z"
        )
        actual = temporal_split(
            split_fixture(),
            "1970-01-01T01:01:40+01:00",
            pd.Timestamp("1969-12-31T19:03:20-05:00"),
        )

        for name in expected:
            pd.testing.assert_frame_equal(actual[name], expected[name])

    def test_reversed_or_equal_cutoffs_are_rejected(self):
        for train_end, validation_end in ((200, 100), (100, 100)):
            with self.subTest(train_end=train_end, validation_end=validation_end):
                with self.assertRaises(ValueError):
                    temporal_split(
                        split_fixture(),
                        utc_timestamp(train_end),
                        utc_timestamp(validation_end),
                    )

    def test_splitting_does_not_mutate_the_input(self):
        interactions = split_fixture()
        original = interactions.copy(deep=True)

        temporal_split(interactions, utc_timestamp(100), utc_timestamp(200))

        pd.testing.assert_frame_equal(interactions, original)

    def test_empty_groups_and_empty_inputs_are_allowed(self):
        for interactions in (split_fixture().iloc[[6]], split_fixture().iloc[:0]):
            with self.subTest(input_rows=len(interactions)):
                parts = temporal_split(
                    interactions, utc_timestamp(100), utc_timestamp(200)
                )
                self.assertEqual(len(parts["train"]), len(interactions))
                self.assertTrue(parts["validation"].empty)
                self.assertTrue(parts["test"].empty)
                for frame in parts.values():
                    self.assertEqual(list(frame.columns), INTERACTION_COLUMNS)


if __name__ == "__main__":
    unittest.main()

"""Behavioral and analytic tests for the genre content-based baseline."""

import math
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr

from src.models.content_based import ContentBasedRecommender, tokenize_genres


RESULT_COLUMNS = ["movieId", "title", "genres", "score"]


def balanced_catalogue():
    """Equal document frequencies make the common IDF cancel in each row.

    Action, Comedy, and Drama each occur in four films. Consequently a
    single-genre row has weight 1, a two-genre row has weights 1/sqrt(2),
    and the three-genre row has weights 1/sqrt(3), independently of the
    common IDF value. The final movie exercises a zero-feature row.
    """
    return pd.DataFrame(
        {
            "movieId": [70, 30, 50, 10, 80, 40, 20, 60],
            "title": [
                "All three",
                "Comedy only",
                "Action and comedy",
                "Action only",
                "Unclassified",
                "Action and drama",
                "Drama only",
                "Comedy and drama",
            ],
            "genres": [
                "Action|Comedy|Drama",
                "Comedy",
                "Action|Comedy",
                "Action",
                "(no genres listed)",
                "Action|Drama",
                "Drama",
                "Comedy|Drama",
            ],
        },
        index=[12, 7, 21, 4, 19, 2, 13, 8],
    )


class GenreTokenizerTests(unittest.TestCase):
    def test_preserves_hyphenated_genres_and_removes_duplicate_blank_tokens(self):
        self.assertEqual(
            tokenize_genres(" Sci-Fi |Film-Noir|Action|Sci-Fi|| "),
            ["Action", "Film-Noir", "Sci-Fi"],
        )

    def test_missing_and_placeholder_genres_have_no_tokens(self):
        for value in (None, np.nan, pd.NA, "", "   ", "(no genres listed)"):
            with self.subTest(value=value):
                self.assertEqual(tokenize_genres(value), [])
        self.assertEqual(
            tokenize_genres("Drama| (no genres listed) | |Drama"), ["Drama"]
        )

    def test_imax_is_optional_and_not_a_default_genre(self):
        self.assertEqual(tokenize_genres("IMAX|Action|IMAX"), ["Action"])
        self.assertEqual(
            tokenize_genres("IMAX|Action|IMAX", include_imax=True),
            ["Action", "IMAX"],
        )
        self.assertEqual(tokenize_genres("IMAX"), [])


class FeatureMatrixTests(unittest.TestCase):
    def test_sparse_rows_align_with_the_sorted_catalogue_and_known_unit_vectors(self):
        model = ContentBasedRecommender(balanced_catalogue())

        self.assertTrue(isspmatrix_csr(model.feature_matrix))
        self.assertIsInstance(model.feature_names, np.ndarray)
        self.assertEqual(model.feature_names.tolist(), ["Action", "Comedy", "Drama"])
        self.assertEqual(model.movies["movieId"].tolist(), list(range(10, 90, 10)))
        self.assertEqual(model.feature_matrix.shape, (8, 3))
        self.assertEqual(model.feature_matrix.nnz, 12)
        two = 1 / math.sqrt(2)
        three = 1 / math.sqrt(3)
        expected = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [two, 0, two],
                [two, two, 0],
                [0, two, two],
                [three, three, three],
                [0, 0, 0],
            ]
        )
        np.testing.assert_allclose(model.feature_matrix.toarray(), expected)

    def test_hyphenated_genres_remain_atomic_features(self):
        movies = pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "title": ["Future", "Noir", "Both"],
                "genres": ["Sci-Fi", "Film-Noir", "Sci-Fi|Film-Noir"],
            }
        )
        model = ContentBasedRecommender(movies)
        self.assertEqual(set(model.feature_names), {"Sci-Fi", "Film-Noir"})
        self.assertEqual(model.feature_matrix.shape, (3, 2))

    def test_missing_genres_become_zero_rows(self):
        for missing in (None, np.nan, pd.NA, "", "IMAX"):
            with self.subTest(missing=missing):
                movies = balanced_catalogue()
                movies.loc[movies["movieId"] == 80, "genres"] = missing
                model = ContentBasedRecommender(movies)
                row = model.movies["movieId"].tolist().index(80)
                self.assertEqual(model.feature_matrix.getrow(row).nnz, 0)

    def test_imax_option_changes_the_vocabulary_and_similarities(self):
        movies = pd.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["Big screen one", "Big screen two"],
                "genres": ["IMAX", "IMAX"],
            }
        )
        with self.assertRaises(ValueError):
            ContentBasedRecommender(movies)
        model = ContentBasedRecommender(movies, include_imax=True)
        self.assertEqual(model.feature_names.tolist(), ["IMAX"])
        result = model.recommend([1])
        self.assertEqual(result["movieId"].tolist(), [2])
        self.assertAlmostEqual(result.iloc[0]["score"], 1.0)

    def test_constructor_copies_the_catalogue(self):
        movies = balanced_catalogue()
        original = movies.copy(deep=True)
        model = ContentBasedRecommender(movies)
        pd.testing.assert_frame_equal(movies, original)
        movies.loc[movies["movieId"] == 10, "title"] = "Changed externally"
        movies.loc[movies["movieId"] == 10, "genres"] = "Western"
        self.assertEqual(
            model.movies.loc[model.movies["movieId"] == 10, "title"].iloc[0],
            "Action only",
        )
        self.assertEqual(
            model.movies.loc[model.movies["movieId"] == 10, "genres"].iloc[0],
            "Action",
        )


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.model = ContentBasedRecommender(balanced_catalogue())

    def test_single_favorite_excludes_itself_and_zero_similarity_movies(self):
        result = self.model.recommend([10])

        self.assertEqual(result.columns.tolist(), RESULT_COLUMNS)
        self.assertEqual(result["movieId"].tolist(), [40, 50, 70])
        np.testing.assert_allclose(
            result["score"].to_numpy(),
            [1 / math.sqrt(2), 1 / math.sqrt(2), 1 / math.sqrt(3)],
        )
        self.assertEqual(result["title"].tolist(), [
            "Action and drama", "Action and comedy", "All three"
        ])
        self.assertEqual(result["genres"].tolist(), [
            "Action|Drama", "Action|Comedy", "Action|Comedy|Drama"
        ])

    def test_cosine_renormalizes_the_mean_profile_before_scoring(self):
        result = self.model.recommend([10, 20])

        self.assertEqual(result["movieId"].tolist(), [40, 70, 50, 60])
        np.testing.assert_allclose(
            result["score"].to_numpy(), [1, math.sqrt(2 / 3), 0.5, 0.5]
        )

    def test_profile_averages_normalized_tfidf_rows_not_raw_genre_counts(self):
        result = self.model.recommend([10, 40])

        # [1, 0, 0] and [1/sqrt(2), 0, 1/sqrt(2)] form the profile.
        action = 1 + 1 / math.sqrt(2)
        drama = 1 / math.sqrt(2)
        length = math.sqrt(action ** 2 + drama ** 2)
        expected_scores = {
            20: drama / length,
            50: action / (math.sqrt(2) * length),
            60: drama / (math.sqrt(2) * length),
            70: (action + drama) / (math.sqrt(3) * length),
        }
        self.assertEqual(result["movieId"].tolist(), [70, 50, 20, 60])
        for row in result.itertuples(index=False):
            self.assertAlmostEqual(row.score, expected_scores[row.movieId])

    def test_duplicate_favorites_do_not_weight_the_user_profile(self):
        expected = self.model.recommend([10, 40])
        actual = self.model.recommend([10, 10, 40, 10, 40])
        pd.testing.assert_frame_equal(actual, expected)

    def test_favorite_and_catalogue_order_do_not_change_recommendations(self):
        expected = self.model.recommend([10, 20])
        reordered = ContentBasedRecommender(
            balanced_catalogue().sample(frac=1, random_state=17)
        )
        pd.testing.assert_frame_equal(reordered.recommend([20, 10]), expected)

    def test_favorite_iterable_is_not_modified_and_generators_are_accepted(self):
        favorite_ids = [20, 10, 20]
        expected = self.model.recommend([10, 20])
        actual = self.model.recommend(favorite_ids)
        self.assertEqual(favorite_ids, [20, 10, 20])
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(
            self.model.recommend(movie_id for movie_id in [20, 10]), expected
        )

    def test_numpy_integer_favorites_and_limits_are_accepted(self):
        result = self.model.recommend([np.int64(10)], n=np.int64(2))
        self.assertEqual(result["movieId"].tolist(), [40, 50])

    def test_requested_limit_applies_after_filtering_and_ranking(self):
        self.assertEqual(self.model.recommend([10], n=1)["movieId"].tolist(), [40])
        self.assertEqual(
            self.model.recommend([10], n=100)["movieId"].tolist(), [40, 50, 70]
        )

    def test_zero_feature_favorite_does_not_dilute_cosine_scores(self):
        pd.testing.assert_frame_equal(
            self.model.recommend([10, 80]), self.model.recommend([10])
        )

    def test_all_available_signal_movies_as_favorites_returns_empty_schema(self):
        result = self.model.recommend(range(10, 80, 10))
        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), RESULT_COLUMNS)

    def test_disjoint_candidates_return_empty_schema(self):
        movies = balanced_catalogue()
        movies = movies.loc[movies["movieId"].isin([10, 20, 30, 80])]
        result = ContentBasedRecommender(movies).recommend([10])
        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), RESULT_COLUMNS)

    def test_scores_are_finite_positive_and_bounded(self):
        for favorites in ([10], [10, 20], [10, 40], [70], [10, 20, 30]):
            with self.subTest(favorites=favorites):
                scores = self.model.recommend(favorites)["score"].to_numpy()
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue((scores > 0).all())
                self.assertTrue((scores <= 1).all())

    def test_recommend_does_not_mutate_features_or_catalogue(self):
        before_movies = self.model.movies.copy(deep=True)
        before_features = self.model.feature_matrix.copy()
        expected = self.model.recommend([10, 20])
        changed_result = self.model.recommend([10, 20])
        changed_result.loc[:, "title"] = "Edited returned results"
        changed_result.loc[:, "score"] = 0.0

        pd.testing.assert_frame_equal(self.model.movies, before_movies)
        self.assertEqual((self.model.feature_matrix != before_features).nnz, 0)
        pd.testing.assert_frame_equal(self.model.recommend([10, 20]), expected)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.model = ContentBasedRecommender(balanced_catalogue())

    def test_empty_favorites_are_rejected(self):
        for favorites in ([], (), iter(())):
            with self.subTest(favorites=favorites):
                with self.assertRaises(ValueError):
                    self.model.recommend(favorites)

    def test_unknown_favorite_is_rejected_even_among_known_ids(self):
        for favorites in ([999], [10, 999]):
            with self.subTest(favorites=favorites):
                with self.assertRaises(ValueError):
                    self.model.recommend(favorites)

    def test_favorite_ids_must_be_integers_and_bool_is_not_an_id(self):
        for favorites in (["10"], [10.0], [True], [None], [np.nan], 10, "10", None):
            with self.subTest(favorites=favorites):
                with self.assertRaises(ValueError):
                    self.model.recommend(favorites)

    def test_profile_without_usable_genres_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.recommend([80])

    def test_n_must_be_a_positive_integer_and_bool_is_not_a_limit(self):
        for n in (0, -1, 1.5, "3", True, False, None, np.nan):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    self.model.recommend([10], n=n)

    def test_required_catalogue_columns_are_checked(self):
        for column in ("movieId", "title", "genres"):
            with self.subTest(column=column):
                with self.assertRaises(ValueError):
                    ContentBasedRecommender(balanced_catalogue().drop(columns=column))

    def test_empty_catalogue_is_rejected(self):
        with self.assertRaises(ValueError):
            ContentBasedRecommender(balanced_catalogue().iloc[0:0])

    def test_duplicate_movie_ids_are_rejected(self):
        movies = balanced_catalogue()
        duplicated = pd.concat([movies, movies.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            ContentBasedRecommender(duplicated)

    def test_catalogue_ids_must_be_integers_and_cannot_be_missing_or_boolean(self):
        for bad_id in ("10", 10.0, None, np.nan, True):
            with self.subTest(bad_id=bad_id):
                movies = balanced_catalogue()
                movies["movieId"] = movies["movieId"].astype(object)
                movies.loc[movies["movieId"] == 10, "movieId"] = bad_id
                with self.assertRaises(ValueError):
                    ContentBasedRecommender(movies)

    def test_catalogue_without_usable_vocabulary_is_rejected(self):
        movies = pd.DataFrame(
            {
                "movieId": [1, 2, 3, 4],
                "title": ["A", "B", "C", "D"],
                "genres": [None, "", "(no genres listed)", "IMAX"],
            }
        )
        with self.assertRaises(ValueError):
            ContentBasedRecommender(movies)


if __name__ == "__main__":
    unittest.main()

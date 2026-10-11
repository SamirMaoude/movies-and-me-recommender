"""Synthetic tests for the API-facing TMDB recommendation adapter."""

import json
import math
import unittest

import numpy as np
import pandas as pd

from src.data.id_mapper import IdMapper
from src.models.content_based import ContentBasedRecommender
from src.models.tmdb_recommender import TMDBRecommender


def movie_catalogue(rows):
    return pd.DataFrame(
        [(movie_id, f"Movie {movie_id}", genres) for movie_id, genres in rows],
        columns=["movieId", "title", "genres"],
    )


def balanced_catalogue():
    """Each of the three genres has document frequency four."""
    return movie_catalogue(
        [
            (70, "Action|Comedy|Drama"),
            (30, "Comedy"),
            (50, "Action|Comedy"),
            (10, "Action"),
            (80, "(no genres listed)"),
            (40, "Action|Drama"),
            (20, "Drama"),
            (60, "Comedy|Drama"),
        ]
    )


def adapter_for(movies, mapping_rows):
    model = ContentBasedRecommender(movies)
    mapper = IdMapper(pd.DataFrame(mapping_rows, columns=["movieId", "tmdbId"]))
    return TMDBRecommender(model, mapper)


class TMDBRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.movies = balanced_catalogue()
        self.links = pd.DataFrame(
            {
                "movieId": list(range(10, 90, 10)) + [900],
                "tmdbId": list(range(100, 900, 100)) + [9000],
            }
        )
        self.model = ContentBasedRecommender(self.movies)
        self.mapper = IdMapper(self.links)
        self.adapter = TMDBRecommender(self.model, self.mapper)

    def test_results_contain_only_tmdb_ids_and_preserve_analytic_scores(self):
        result = self.adapter.recommend([100])
        self.assertIs(type(result), list)
        self.assertEqual([item["tmdb_id"] for item in result], [400, 500, 700])
        np.testing.assert_allclose(
            [item["score"] for item in result],
            [1 / math.sqrt(2), 1 / math.sqrt(2), 1 / math.sqrt(3)],
        )
        for item in result:
            self.assertEqual(set(item), {"tmdb_id", "score"})
            self.assertIs(type(item["tmdb_id"]), int)
            self.assertIs(type(item["score"]), float)

    def test_multiple_favorites_preserve_cosine_of_the_average_profile(self):
        result = self.adapter.recommend([100, 200])
        self.assertEqual([item["tmdb_id"] for item in result], [400, 700, 500, 600])
        np.testing.assert_allclose(
            [item["score"] for item in result], [1, math.sqrt(2 / 3), 0.5, 0.5]
        )

    def test_unknown_tmdb_ids_and_links_outside_catalogue_are_ignored(self):
        self.assertEqual(
            self.adapter.recommend([99999, 100, 9000]), self.adapter.recommend([100])
        )

    def test_empty_or_all_unknown_favorites_return_empty_list(self):
        for favorites in ([], (), iter(()), [99999], [9000], [99999, 9000]):
            with self.subTest(favorites=favorites):
                self.assertEqual(self.adapter.recommend(favorites), [])

    def test_all_zero_genre_favorites_return_empty_list(self):
        self.assertEqual(self.adapter.recommend([800]), [])
        self.assertEqual(self.adapter.recommend([800, 99999]), [])

    def test_mixed_valid_and_zero_genre_favorites_keep_the_same_cosine_scores(self):
        self.assertEqual(
            self.adapter.recommend([100, 800]), self.adapter.recommend([100])
        )

    def test_deduplicating_tmdb_favorites_does_not_weight_the_profile(self):
        self.assertEqual(
            self.adapter.recommend([100, 100, 400, 100, 400]),
            self.adapter.recommend([100, 400]),
        )

    def test_generators_and_favorite_permutations_are_supported(self):
        expected = self.adapter.recommend([100, 200])
        self.assertEqual(self.adapter.recommend([200, 100]), expected)
        self.assertEqual(
            self.adapter.recommend(tmdb_id for tmdb_id in [200, 100, 200]), expected
        )

    def test_catalogue_and_link_row_order_do_not_change_results(self):
        reordered = TMDBRecommender(
            ContentBasedRecommender(self.movies.sample(frac=1, random_state=5)),
            IdMapper(self.links.sample(frac=1, random_state=7)),
        )
        self.assertEqual(
            reordered.recommend([100, 200]), self.adapter.recommend([100, 200])
        )

    def test_numpy_integer_favorites_and_limit_produce_json_compatible_python_types(
        self,
    ):
        result = self.adapter.recommend([np.int64(100)], n=np.int64(2))
        self.assertEqual([item["tmdb_id"] for item in result], [400, 500])
        for item in result:
            self.assertIs(type(item["tmdb_id"]), int)
            self.assertIs(type(item["score"]), float)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_returns_available_positive_similarity_candidates_without_filling(self):
        result = self.adapter.recommend([100], n=100)
        self.assertEqual([item["tmdb_id"] for item in result], [400, 500, 700])
        self.assertTrue(all(0 < item["score"] <= 1 for item in result))

    def test_all_signal_movies_as_favorites_leave_no_candidates(self):
        self.assertEqual(self.adapter.recommend(range(100, 800, 100)), [])

    def test_recommendations_do_not_mutate_inputs_or_the_underlying_model(self):
        favorites = [100, 200, 100, 99999]
        source_movies = self.movies.copy(deep=True)
        source_links = self.links.copy(deep=True)
        fitted_movies = self.model.movies.copy(deep=True)
        features = self.model.feature_matrix.copy()
        expected = self.adapter.recommend(favorites)
        changed_result = self.adapter.recommend(favorites)
        changed_result[0]["tmdb_id"] = 99999
        changed_result[0]["score"] = 0.0

        self.assertEqual(favorites, [100, 200, 100, 99999])
        pd.testing.assert_frame_equal(self.movies, source_movies)
        pd.testing.assert_frame_equal(self.links, source_links)
        pd.testing.assert_frame_equal(self.model.movies, fitted_movies)
        self.assertEqual((self.model.feature_matrix != features).nnz, 0)
        self.assertEqual(self.mapper.to_tmdb_id(10), 100)
        self.assertEqual(self.adapter.recommend(favorites), expected)


class TMDBCollisionAndFilteringTests(unittest.TestCase):
    def test_limit_is_applied_after_missing_mappings_duplicates_and_favorite_aliases(
        self,
    ):
        movies = movie_catalogue(
            [(movie_id, "Action") for movie_id in range(10, 90, 10)]
        )
        adapter = adapter_for(
            movies,
            [
                (10, 100),
                (20, None),
                (30, 300),
                (40, 300),
                (50, 100),
                (60, np.nan),
                (70, 700),
                (80, 800),
            ],
        )
        # The raw ranking is 20, 30, 40, 50, 60, 70, 80; only 30, 70, 80 survive.
        self.assertEqual(
            adapter.recommend([100], n=1), [{"tmdb_id": 300, "score": 1.0}]
        )
        self.assertEqual(
            adapter.recommend([100], n=2),
            [{"tmdb_id": 300, "score": 1.0}, {"tmdb_id": 700, "score": 1.0}],
        )
        self.assertEqual(
            adapter.recommend([100], n=10),
            [
                {"tmdb_id": 300, "score": 1.0},
                {"tmdb_id": 700, "score": 1.0},
                {"tmdb_id": 800, "score": 1.0},
            ],
        )

    def test_favorite_collision_uses_smallest_catalogue_alias_and_excludes_all_aliases(
        self,
    ):
        movies = movie_catalogue(
            [
                (10, "Action"),
                (20, "Action|Comedy"),
                (30, "Action"),
                (40, "Comedy"),
                (50, "Action|Comedy"),
                (60, "Comedy"),
            ]
        )
        adapter = adapter_for(
            movies, [(20, 100), (10, 100), (30, 300), (40, 400), (50, 500), (60, 600)]
        )
        result = adapter.recommend([100])
        self.assertEqual([item["tmdb_id"] for item in result], [300, 500])
        np.testing.assert_allclose(
            [item["score"] for item in result], [1, 1 / math.sqrt(2)]
        )
        self.assertNotIn(100, [item["tmdb_id"] for item in result])

    def test_missing_global_canonical_alias_falls_back_to_smallest_present_alias(self):
        movies = movie_catalogue(
            [
                (10, "Action"),
                (20, "Action|Comedy"),
                (30, "Action"),
                (40, "Comedy"),
                (50, "Action|Comedy"),
                (60, "Comedy"),
            ]
        )
        mappings = [
            (5, 100),
            (20, 100),
            (10, 100),
            (30, 300),
            (40, 400),
            (50, 500),
            (60, 600),
        ]
        mapper = IdMapper(pd.DataFrame(mappings, columns=["movieId", "tmdbId"]))
        self.assertEqual(mapper.to_movie_id(100), 5)
        result = TMDBRecommender(ContentBasedRecommender(movies), mapper).recommend(
            [100]
        )
        self.assertEqual([item["tmdb_id"] for item in result], [300, 500])
        np.testing.assert_allclose(
            [item["score"] for item in result], [1, 1 / math.sqrt(2)]
        )

    def test_canonical_alias_choice_does_not_skip_a_present_zero_genre_movie(self):
        adapter = adapter_for(
            movie_catalogue([(10, ""), (20, "Action"), (30, "Action")]),
            [(10, 100), (20, 100), (30, 300)],
        )
        self.assertEqual(adapter.recommend([100]), [])

    def test_candidate_collision_keeps_the_best_score_rather_than_smallest_alias(self):
        adapter = adapter_for(
            movie_catalogue(
                [(10, "Action"), (20, "Action|Comedy"), (30, "Action"), (40, "Comedy")]
            ),
            [(10, 100), (20, 300), (30, 300), (40, 400)],
        )
        self.assertEqual(adapter.recommend([100]), [{"tmdb_id": 300, "score": 1.0}])

    def test_equal_score_candidates_keep_movielens_order_after_tmdb_translation(self):
        adapter = adapter_for(
            movie_catalogue([(10, "Action"), (20, "Action"), (30, "Action")]),
            [(10, 900), (20, 800), (30, 100)],
        )
        self.assertEqual(
            adapter.recommend([900]),
            [{"tmdb_id": 800, "score": 1.0}, {"tmdb_id": 100, "score": 1.0}],
        )

    def test_all_positive_candidates_without_tmdb_mappings_return_empty(self):
        adapter = adapter_for(
            movie_catalogue([(10, "Action"), (20, "Action"), (30, "Action")]),
            [(10, 100), (20, None)],
        )
        self.assertEqual(adapter.recommend([100]), [])

    def test_empty_links_produce_empty_recommendations(self):
        adapter = adapter_for(movie_catalogue([(10, "Action"), (20, "Action")]), [])
        self.assertEqual(adapter.recommend([100]), [])


class TMDBValidationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = adapter_for(
            movie_catalogue([(10, "Action"), (20, "Action")]), [(10, 100), (20, 200)]
        )

    def test_limit_is_validated_even_for_empty_or_unknown_favorites(self):
        for n in (0, -1, 1.5, "2", True, np.bool_(True), None, np.nan):
            for favorites in ([100], [], [99999]):
                with self.subTest(n=n, favorites=favorites):
                    with self.assertRaises(ValueError):
                        self.adapter.recommend(favorites, n=n)


if __name__ == "__main__":
    unittest.main()

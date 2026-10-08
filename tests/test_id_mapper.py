"""Synthetic tests for MovieLens/TMDB mapping and identifier validation."""

import unittest

import numpy as np
import pandas as pd

from src.data.id_mapper import IdMapper


def sample_links():
    return pd.DataFrame(
        {
            "movieId": [30, 50, 20, 10, 60, 40],
            "tmdbId": [100, pd.NA, 200, 100, np.nan, None],
            "imdbId": ["0030", "0050", "0020", "0010", "0060", "0040"],
        },
        index=[8, 3, 12, 5, 20, 1],
    )


class IdMapperLookupTests(unittest.TestCase):
    def setUp(self):
        self.mapper = IdMapper(sample_links())

    def test_forward_mapping_keeps_each_movielens_alias(self):
        self.assertEqual(self.mapper.to_tmdb_id(10), 100)
        self.assertEqual(self.mapper.to_tmdb_id(30), 100)
        self.assertEqual(self.mapper.to_tmdb_id(20), 200)
        self.assertIs(type(self.mapper.to_tmdb_id(10)), int)

    def test_reverse_mapping_chooses_the_smallest_alias_independent_of_row_order(self):
        self.assertEqual(self.mapper.to_movie_id(100), 10)
        self.assertEqual(self.mapper.to_movie_id(200), 20)
        reordered = IdMapper(sample_links().sample(frac=1, random_state=31))
        self.assertEqual(reordered.to_movie_id(100), 10)
        self.assertEqual(reordered.movie_ids_for_tmdb(100), (10, 30))

    def test_all_aliases_are_sorted_immutable_python_integer_tuples(self):
        aliases = self.mapper.movie_ids_for_tmdb(100)
        self.assertEqual(aliases, (10, 30))
        self.assertIs(type(aliases), tuple)
        self.assertTrue(all(type(movie_id) is int for movie_id in aliases))
        self.assertEqual(self.mapper.movie_ids_for_tmdb(200), (20,))

    def test_missing_tmdb_ids_and_unknown_ids_have_no_mapping(self):
        for movie_id in (40, 50, 60, 999):
            with self.subTest(movie_id=movie_id):
                self.assertIsNone(self.mapper.to_tmdb_id(movie_id))
        self.assertIsNone(self.mapper.to_movie_id(999))
        self.assertEqual(self.mapper.movie_ids_for_tmdb(999), ())

    def test_only_ambiguous_tmdb_groups_are_reported_and_the_dict_is_a_copy(self):
        ambiguity = self.mapper.ambiguous_mappings
        self.assertEqual(ambiguity, {100: (10, 30)})
        self.assertIs(type(ambiguity), dict)
        self.assertIs(type(next(iter(ambiguity))), int)
        ambiguity[100] = (999,)
        ambiguity[200] = (888, 999)
        self.assertEqual(self.mapper.ambiguous_mappings, {100: (10, 30)})
        self.assertEqual(self.mapper.to_movie_id(100), 10)
        self.assertEqual(self.mapper.movie_ids_for_tmdb(100), (10, 30))

    def test_numpy_integer_lookup_inputs_are_accepted(self):
        self.assertEqual(self.mapper.to_tmdb_id(np.int64(10)), 100)
        self.assertEqual(self.mapper.to_movie_id(np.int32(100)), 10)
        self.assertEqual(self.mapper.movie_ids_for_tmdb(np.uint64(100)), (10, 30))

    def test_lookup_inputs_are_strict_positive_integers(self):
        invalid_ids = (0, -1, 10.0, "10", True, np.bool_(False), None, np.nan)
        for method in (
            self.mapper.to_tmdb_id,
            self.mapper.to_movie_id,
            self.mapper.movie_ids_for_tmdb,
        ):
            for invalid_id in invalid_ids:
                with self.subTest(method=method.__name__, invalid_id=invalid_id):
                    with self.assertRaises(ValueError):
                        method(invalid_id)


class IdMapperConstructionTests(unittest.TestCase):
    def test_csv_style_integral_floats_are_accepted_in_both_columns(self):
        mapper = IdMapper(
            pd.DataFrame(
                {"movieId": [20.0, 10.0, 30.0], "tmdbId": [100.0, 100.0, np.nan]}
            )
        )
        self.assertEqual(mapper.to_tmdb_id(20), 100)
        self.assertEqual(mapper.to_movie_id(100), 10)
        self.assertEqual(mapper.movie_ids_for_tmdb(100), (10, 20))
        self.assertIsNone(mapper.to_tmdb_id(30))
        self.assertIs(type(mapper.to_movie_id(100)), int)

    def test_nullable_integer_columns_and_numpy_integers_are_accepted(self):
        mapper = IdMapper(
            pd.DataFrame(
                {
                    "movieId": pd.array([np.int64(1), np.int64(2)], dtype="Int64"),
                    "tmdbId": pd.array([np.int64(101), None], dtype="Int64"),
                }
            )
        )
        self.assertEqual(mapper.to_tmdb_id(1), 101)
        self.assertIsNone(mapper.to_tmdb_id(2))

    def test_empty_links_with_required_schema_are_valid(self):
        mapper = IdMapper(pd.DataFrame(columns=["movieId", "tmdbId"]))
        self.assertIsNone(mapper.to_tmdb_id(1))
        self.assertIsNone(mapper.to_movie_id(1))
        self.assertEqual(mapper.movie_ids_for_tmdb(1), ())
        self.assertEqual(mapper.ambiguous_mappings, {})

    def test_all_missing_tmdb_values_are_valid(self):
        mapper = IdMapper(
            pd.DataFrame({"movieId": [1, 2, 3], "tmdbId": [None, np.nan, pd.NA]})
        )
        for movie_id in (1, 2, 3):
            self.assertIsNone(mapper.to_tmdb_id(movie_id))
        self.assertEqual(mapper.ambiguous_mappings, {})

    def test_missing_schema_or_non_dataframe_inputs_are_rejected(self):
        for links in (
            pd.DataFrame({"movieId": [1]}),
            pd.DataFrame({"tmdbId": [100]}),
            pd.DataFrame(),
            {"movieId": [1], "tmdbId": [100]},
            None,
        ):
            with self.subTest(links=links):
                with self.assertRaises(ValueError):
                    IdMapper(links)

    def test_duplicate_movielens_ids_are_rejected_even_when_the_mapping_agrees(self):
        for tmdb_ids in ([100, 100], [100, 200], [None, 100]):
            with self.subTest(tmdb_ids=tmdb_ids):
                with self.assertRaises(ValueError):
                    IdMapper(pd.DataFrame({"movieId": [1, 1], "tmdbId": tmdb_ids}))

    def test_movie_ids_reject_missing_nonpositive_and_nonintegral_values(self):
        invalid_ids = (None, pd.NA, np.nan, 0, -1, 1.5, "1", True, np.bool_(True), np.inf)
        for invalid_id in invalid_ids:
            with self.subTest(invalid_id=invalid_id):
                links = pd.DataFrame(
                    {"movieId": pd.Series([invalid_id], dtype=object), "tmdbId": [100]}
                )
                with self.assertRaises(ValueError):
                    IdMapper(links)

    def test_tmdb_ids_reject_nonpositive_and_nonintegral_values(self):
        invalid_ids = (0, -1, 100.5, "100", True, np.bool_(True), np.inf, -np.inf)
        for invalid_id in invalid_ids:
            with self.subTest(invalid_id=invalid_id):
                links = pd.DataFrame(
                    {"movieId": [1], "tmdbId": pd.Series([invalid_id], dtype=object)}
                )
                with self.assertRaises(ValueError):
                    IdMapper(links)

    def test_mapper_does_not_mutate_or_retain_mutable_source_links(self):
        links = sample_links()
        original = links.copy(deep=True)
        mapper = IdMapper(links)
        pd.testing.assert_frame_equal(links, original)
        links.loc[links["movieId"] == 10, "tmdbId"] = 999
        links.loc[links["movieId"] == 30, "movieId"] = 333
        self.assertEqual(mapper.to_tmdb_id(10), 100)
        self.assertEqual(mapper.movie_ids_for_tmdb(100), (10, 30))


if __name__ == "__main__":
    unittest.main()

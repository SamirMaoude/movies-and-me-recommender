"""Regression tests for the phase-one MovieLens exploration helpers."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data.exploration import (
    data_quality_report,
    genre_counts,
    interaction_tables,
    load_movielens,
    summarize_dataset,
)


def sample_tables():
    """Keep an unrated movie and links outside the movie catalogue."""
    return {
        "movies": pd.DataFrame(
            {
                "movieId": [10, 20, 30],
                "title": ["First (2000)", "Second (2001)", "Third (2002)"],
                "genres": ["Drama|Comedy", "Drama", "(no genres listed)"],
            }
        ),
        "ratings": pd.DataFrame(
            {
                "userId": [1, 1, 2, 2],
                "movieId": [10, 20, 10, 10],
                "rating": [4.0, 5.0, 2.0, 3.0],
                "timestamp": [100, 101, 102, 103],
            }
        ),
        "links": pd.DataFrame(
            {
                "movieId": [10, 20, 40],
                "imdbId": ["0000010", "0000020", "0000040"],
                "tmdbId": pd.array([100, None, 400], dtype="Int64"),
            }
        ),
        "tags": pd.DataFrame(
            {
                "userId": [1, 1],
                "movieId": [10, 10],
                "tag": ["classic", "funny"],
                "timestamp": [100, 101],
            }
        ),
    }


class LoadMovieLensTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.data_dir = Path(self.temp_directory.name)
        csv_files = {
            "movies.csv": (
                "movieId,title,genres\n"
                '10,"First, the (2000)",Drama|Comedy\n'
                "20,Second (2001),Drama\n"
            ),
            "ratings.csv": (
                "userId,movieId,rating,timestamp\n"
                "1,10,4.5,100\n"
                "2,20,3.0,101\n"
            ),
            "links.csv": (
                "movieId,imdbId,tmdbId\n"
                "10,0000010,100\n"
                "20,0000020,\n"
            ),
            "tags.csv": "userId,movieId,tag,timestamp\n1,10,classic,100\n",
        }
        for filename, content in csv_files.items():
            (self.data_dir / filename).write_text(content, encoding="utf-8")

    def test_loads_all_four_tables_and_preserves_identifier_types(self):
        tables = load_movielens(self.data_dir)

        self.assertEqual(set(tables), {"movies", "ratings", "links", "tags"})
        self.assertEqual(tables["movies"].loc[0, "title"], "First, the (2000)")
        self.assertEqual(tables["links"]["imdbId"].tolist(), ["0000010", "0000020"])
        self.assertEqual(str(tables["links"]["tmdbId"].dtype), "Int64")
        self.assertEqual(tables["links"].loc[0, "tmdbId"], 100)
        self.assertTrue(pd.isna(tables["links"].loc[1, "tmdbId"]))
        self.assertEqual(tables["ratings"].loc[0, "rating"], 4.5)

    def test_every_required_csv_must_exist(self):
        for filename in ("movies.csv", "ratings.csv", "links.csv", "tags.csv"):
            with self.subTest(filename=filename):
                file_path = self.data_dir / filename
                content = file_path.read_text(encoding="utf-8")
                file_path.unlink()
                try:
                    with self.assertRaises(FileNotFoundError):
                        load_movielens(self.data_dir)
                finally:
                    file_path.write_text(content, encoding="utf-8")

    def test_missing_data_directory_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_movielens(self.data_dir / "missing")


class DataQualityTests(unittest.TestCase):
    def test_reports_sizes_missing_cells_and_repeated_rating_pairs_separately(self):
        tables = sample_tables()
        tables["ratings"].loc[1, "rating"] = float("nan")
        tables["ratings"] = pd.concat(
            [tables["ratings"], tables["ratings"].iloc[[0]]], ignore_index=True
        )

        quality = data_quality_report(tables)

        self.assertEqual(set(quality.index), set(tables))
        self.assertEqual(
            set(quality.columns),
            {"num_rows", "num_columns", "missing_cells", "duplicate_rows", "duplicate_keys"},
        )
        self.assertEqual(quality.loc["ratings", "num_rows"], 5)
        self.assertEqual(quality.loc["ratings", "num_columns"], 4)
        self.assertEqual(quality.loc["ratings", "missing_cells"], 1)
        self.assertEqual(quality.loc["ratings", "duplicate_rows"], 1)
        self.assertEqual(quality.loc["ratings", "duplicate_keys"], 2)
        self.assertEqual(quality.loc["links", "missing_cells"], 1)

    def test_movie_and_link_keys_detect_conflicting_rows(self):
        tables = sample_tables()
        conflicting_movie = tables["movies"].iloc[[0]].copy()
        conflicting_movie["title"] = "Conflicting title"
        tables["movies"] = pd.concat(
            [tables["movies"], conflicting_movie], ignore_index=True
        )
        conflicting_link = tables["links"].iloc[[0]].copy()
        conflicting_link["tmdbId"] = pd.array([999], dtype="Int64")
        tables["links"] = pd.concat(
            [tables["links"], conflicting_link], ignore_index=True
        )

        quality = data_quality_report(tables)

        for dataset in ("movies", "links"):
            with self.subTest(dataset=dataset):
                self.assertEqual(quality.loc[dataset, "duplicate_rows"], 0)
                self.assertEqual(quality.loc[dataset, "duplicate_keys"], 1)

    def test_different_tags_or_timestamps_are_legitimate_for_the_same_pair(self):
        tables = sample_tables()
        tables["tags"] = pd.DataFrame(
            {
                "userId": [1, 1, 1],
                "movieId": [10, 10, 10],
                "tag": ["classic", "funny", "classic"],
                "timestamp": [100, 100, 101],
            }
        )

        quality = data_quality_report(tables)
        self.assertEqual(quality.loc["tags", "duplicate_keys"], 0)

        tables["tags"] = pd.concat(
            [tables["tags"], tables["tags"].iloc[[0]]], ignore_index=True
        )
        quality = data_quality_report(tables)
        self.assertEqual(quality.loc["tags", "duplicate_rows"], 1)
        self.assertEqual(quality.loc["tags", "duplicate_keys"], 1)


class InteractionTests(unittest.TestCase):
    def test_counts_rows_and_averages_available_ratings_for_each_user_and_movie(self):
        tables = sample_tables()
        tables["ratings"].loc[3, "rating"] = float("nan")

        per_user, per_movie = interaction_tables(tables["movies"], tables["ratings"])

        self.assertEqual(per_user.index.name, "userId")
        self.assertEqual(per_movie.index.name, "movieId")
        self.assertEqual(set(per_user.columns), {"num_ratings", "avg_rating"})
        self.assertTrue({"num_ratings", "avg_rating"}.issubset(per_movie.columns))
        self.assertEqual(per_user.loc[1, "num_ratings"], 2)
        self.assertEqual(per_user.loc[2, "num_ratings"], 2)
        self.assertAlmostEqual(per_user.loc[1, "avg_rating"], 4.5)
        self.assertAlmostEqual(per_user.loc[2, "avg_rating"], 2.0)
        self.assertEqual(per_movie.loc[10, "num_ratings"], 3)
        self.assertAlmostEqual(per_movie.loc[10, "avg_rating"], 3.0)

    def test_includes_unrated_catalogue_movies(self):
        tables = sample_tables()
        _, per_movie = interaction_tables(tables["movies"], tables["ratings"])

        self.assertEqual(set(per_movie.index), {10, 20, 30})
        self.assertEqual(per_movie.loc[30, "num_ratings"], 0)
        self.assertTrue(pd.isna(per_movie.loc[30, "avg_rating"]))


class GenreCountsTests(unittest.TestCase):
    def test_splits_genres_and_excludes_placeholders_and_blank_values(self):
        movies = pd.DataFrame(
            {
                "movieId": range(7),
                "genres": [
                    "Drama|Comedy",
                    "Drama|Action",
                    "Drama",
                    "(no genres listed)",
                    "",
                    None,
                    "  |Comedy|",
                ],
            }
        )

        counts = genre_counts(movies)

        self.assertIsInstance(counts, pd.Series)
        self.assertEqual(counts.to_dict(), {"Drama": 3, "Comedy": 2, "Action": 1})
        self.assertEqual(counts.tolist(), [3, 2, 1])

    def test_all_missing_or_unlisted_genres_produces_an_empty_series(self):
        counts = genre_counts(
            pd.DataFrame(
                {"movieId": [10, 20, 30, 40], "genres": [None, "", " ", "(no genres listed)"]}
            )
        )
        self.assertIsInstance(counts, pd.Series)
        self.assertTrue(counts.empty)


class DatasetSummaryTests(unittest.TestCase):
    def test_uses_the_complete_catalogue_and_left_join_for_tmdb_coverage(self):
        summary = summarize_dataset(sample_tables())

        self.assertEqual(
            set(summary),
            {
                "num_movies",
                "num_users",
                "num_ratings",
                "avg_ratings_per_user",
                "avg_ratings_per_movie",
                "tmdb_coverage",
            },
        )
        self.assertEqual(summary["num_movies"], 3)
        self.assertEqual(summary["num_users"], 2)
        self.assertEqual(summary["num_ratings"], 4)
        self.assertAlmostEqual(summary["avg_ratings_per_user"], 2.0)
        self.assertAlmostEqual(summary["avg_ratings_per_movie"], 4 / 3)
        self.assertAlmostEqual(summary["tmdb_coverage"], 1 / 3)

    def test_unmatched_links_do_not_count_toward_coverage(self):
        tables = sample_tables()
        tables["links"] = tables["links"].loc[tables["links"]["movieId"] == 40]
        self.assertEqual(summarize_dataset(tables)["tmdb_coverage"], 0.0)

    def test_duplicate_movie_or_link_ids_fail_the_one_to_one_join(self):
        for dataset in ("movies", "links"):
            with self.subTest(dataset=dataset):
                tables = sample_tables()
                tables[dataset] = pd.concat(
                    [tables[dataset], tables[dataset].iloc[[0]]], ignore_index=True
                )
                with self.assertRaises(pd.errors.MergeError):
                    summarize_dataset(tables)

    def test_empty_movies_or_ratings_raise_a_clear_value_error(self):
        for dataset in ("movies", "ratings"):
            with self.subTest(dataset=dataset):
                tables = sample_tables()
                tables[dataset] = tables[dataset].iloc[0:0].copy()
                with self.assertRaises(ValueError):
                    summarize_dataset(tables)


if __name__ == "__main__":
    unittest.main()

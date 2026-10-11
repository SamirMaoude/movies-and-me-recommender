"""Small, hand-calculated checks for the evaluation audit report."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.prepare import build_report, split_summary, validation_coverage


def interactions(rows):
    return pd.DataFrame(
        rows,
        columns=["userId", "movieId", "tmdbId", "rating", "timestamp", "is_positive"],
    )


class EvaluationReportTests(unittest.TestCase):
    def test_split_summary_counts_all_feedback_and_uses_chronological_extrema(self):
        frame = interactions(
            [
                (1, 10, 100, 1.0, 200, False),
                (1, 20, 200, 4.0, 100, True),
                (2, 10, 100, 5.0, 300, True),
            ]
        )

        self.assertEqual(
            split_summary(frame),
            {
                "interactions": 3,
                "positive_interactions": 2,
                "other_interactions": 1,
                "users": 2,
                "movies": 2,
                "first_interaction": "1970-01-01T00:01:40+00:00",
                "last_interaction": "1970-01-01T00:05:00+00:00",
            },
        )

    def test_empty_groups_and_missing_coverage_are_json_null(self):
        empty = interactions([])
        summary = split_summary(empty)
        self.assertEqual(
            summary,
            {
                "interactions": 0,
                "positive_interactions": 0,
                "other_interactions": 0,
                "users": 0,
                "movies": 0,
                "first_interaction": None,
                "last_interaction": None,
            },
        )
        # Slicing a typed frame retains the boolean feedback column.
        typed_empty = interactions([(1, 10, 100, 4.0, 0, True)]).iloc[:0]
        coverage = validation_coverage(typed_empty, typed_empty)
        self.assertIsNone(coverage["catalog_coverage"])
        self.assertIsNone(coverage["eligible_user_coverage"])
        self.assertEqual(coverage["positive_targets"], 0)
        encoded = json.dumps({"split": summary, "coverage": coverage}, allow_nan=False)
        self.assertIn('"first_interaction": null', encoded)
        self.assertIn('"catalog_coverage": null', encoded)

    def test_validation_coverage_uses_only_positive_train_history_and_targets(self):
        train = interactions(
            [
                (1, 10, 100, 4.0, 10, True),
                (1, 20, 200, 1.0, 20, False),
                (2, 30, 300, 2.0, 30, False),
                (5, 40, 400, 5.0, 40, True),
            ]
        )
        validation = interactions(
            [
                (1, 40, 400, 5.0, 100, True),
                (1, 50, 500, 4.0, 101, True),
                (3, 10, 100, 4.5, 102, True),
                (3, 60, 600, 5.0, 103, True),
                (2, 20, 200, 4.0, 104, True),
                (5, 10, 100, 4.0, 105, True),
                (99, 70, 700, 1.0, 106, False),
            ]
        )

        self.assertEqual(
            validation_coverage(train, validation),
            {
                "reference": "positive train interactions only",
                "candidate_movies": 2,
                "positive_targets": 6,
                "users_with_positive_targets": 4,
                "users_with_positive_history": 2,
                "eligible_users": 2,
                "positive_targets_in_catalog": 3,
                "positive_targets_outside_catalog": 3,
                "targets_known_user_known_movie": 2,
                "targets_known_user_unknown_movie": 1,
                "targets_unknown_user_known_movie": 1,
                "targets_unknown_user_unknown_movie": 2,
                "catalog_coverage": 0.5,
                "eligible_user_coverage": 0.5,
            },
        )


class BuildReportTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.directory = Path(self.temp_directory.name)
        self.data_dir = self.directory / "raw"
        self.data_dir.mkdir()
        self.config_path = self.directory / "evaluation.json"
        self.csv_files = {
            "movies.csv": (
                "movieId,title,genres\n10,First,Drama\n20,Second,Comedy\n"
                "30,First alias,Drama\n40,Unlinked,Drama\n"
            ),
            "ratings.csv": (
                "userId,movieId,rating,timestamp\n"
                "1,30,1.0,50\n1,10,5.0,250\n2,20,4.0,99\n"
                "2,30,5.0,100\n3,20,4.5,200\n4,40,4.0,100\n5,999,5.0,100\n"
            ),
            "links.csv": (
                "movieId,imdbId,tmdbId\n10,0000010,100\n20,0000020,200\n"
                "30,0000030,100\n40,0000040,\n"
            ),
            "tags.csv": "userId,movieId,tag,timestamp\n",
        }
        for filename, contents in self.csv_files.items():
            (self.data_dir / filename).write_text(contents, encoding="utf-8")
        self.config = {
            "positive_threshold": 4.0,
            "train_end": "1970-01-01T01:01:40+01:00",
            "validation_end": "1970-01-01T00:03:20Z",
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_csv_pipeline_produces_reproducible_counts_with_normalized_cutoffs(self):
        original_files = {
            path: path.read_bytes()
            for path in self.directory.rglob("*")
            if path.is_file()
        }
        report = build_report(self.data_dir, self.config_path)

        self.assertEqual(
            report["config"],
            {
                "positive_threshold": 4.0,
                "train_end": "1970-01-01T00:01:40+00:00",
                "validation_end": "1970-01-01T00:03:20+00:00",
            },
        )
        self.assertEqual(
            report["preparation"],
            {
                "input_ratings": 7,
                "ratings_without_tmdb": 2,
                "ratings_with_tmdb": 5,
                "remapped_ratings": 2,
                "duplicate_user_movies_removed": 1,
                "prepared_interactions": 4,
                "positive_interactions": 3,
                "other_interactions": 1,
                "users": 3,
                "movies": 2,
                "ambiguous_tmdb_ids": 1,
            },
        )
        self.assertEqual(
            {
                name: summary["interactions"]
                for name, summary in report["splits"].items()
            },
            {"train": 2, "validation": 1, "test": 1},
        )
        self.assertEqual(report["splits"]["train"]["positive_interactions"], 1)
        self.assertEqual(
            report["validation_coverage"]["targets_known_user_unknown_movie"], 1
        )
        self.assertEqual(report["validation_coverage"]["catalog_coverage"], 0.0)
        self.assertEqual(report["validation_coverage"]["eligible_user_coverage"], 0.0)
        for filename in ("ratings.csv", "links.csv"):
            expected_digest = hashlib.sha256(
                original_files[self.data_dir / filename]
            ).hexdigest()
            self.assertEqual(report["sources"][filename]["sha256"], expected_digest)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertEqual(build_report(self.data_dir, self.config_path), report)
        self.assertEqual(
            {
                path: path.read_bytes()
                for path in self.directory.rglob("*")
                if path.is_file()
            },
            original_files,
        )

    def test_missing_extra_or_nonobject_config_fields_are_rejected(self):
        invalid_configs = [
            {key: value for key, value in self.config.items() if key != "train_end"},
            {**self.config, "unexpected": True},
            [],
            {**self.config, "positive_threshold": -1},
            {**self.config, "train_end": "1970-01-01T00:01:40"},
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                self.config_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_report(self.data_dir, self.config_path)


if __name__ == "__main__":
    unittest.main()

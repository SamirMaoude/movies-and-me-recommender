"""Small end-to-end ALS experiments, selection rules and held-out data guards."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from src.models.collaborative import ImplicitRecommender
from src.training.train import load_experiment_config, run_experiment, select_trial


def trial_result(name, ndcg, recall):
    return {"name": name, "evaluation": {"metrics": {"ndcg_at_10": ndcg, "recall_at_10": recall}}}


class TrialSelectionTests(unittest.TestCase):
    def test_ndcg_is_primary_even_when_another_trial_has_higher_recall(self):
        trials = [trial_result("higher_recall", 0.2, 0.8), trial_result("higher_ndcg", 0.3, 0.4)]
        self.assertIs(select_trial(trials), trials[1])

    def test_recall_breaks_ndcg_ties_then_the_first_declared_trial_wins(self):
        trials = [trial_result("first", 0.3, 0.2), trial_result("second", 0.3, 0.4)]
        self.assertIs(select_trial(trials), trials[1])
        trials.append(trial_result("third", 0.3, 0.4))
        self.assertIs(select_trial(trials), trials[1])

    def test_missing_cohort_and_nonfinite_scores_cannot_select_a_winner(self):
        with self.assertRaises(ValueError):
            select_trial([])
        for invalid in (None, math.nan, math.inf, True, "0.2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    select_trial([trial_result("invalid", invalid, 0.2)])
                with self.assertRaises(ValueError):
                    select_trial([trial_result("invalid", 0.2, invalid)])


class TrainingExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.data_dir = self.directory / "raw"
        self.data_dir.mkdir()
        self.evaluation_path = self.directory / "evaluation.json"
        self.als_path = self.directory / "als.json"
        self.evaluation = {
            "positive_threshold": 4.0,
            "train_end": "1970-01-01T00:01:40Z",
            "validation_end": "1970-01-01T00:03:20Z",
        }
        self.experiments = {
            "k": 10,
            "trials": [
                {
                    "name": "small", "factors": 2, "regularization": 0.1,
                    "alpha": 10.0, "iterations": 3, "random_state": 7, "num_threads": 1,
                },
                {
                    "name": "larger", "factors": 3, "regularization": 0.1,
                    "alpha": 10.0, "iterations": 3, "random_state": 7, "num_threads": 1,
                },
            ],
        }
        files = {
            "movies.csv": (
                "movieId,title,genres\n10,First,Drama\n11,Alias,Comedy\n"
                "20,Second,Drama\n30,Third,Drama\n40,Future,Action\n"
            ),
            "links.csv": (
                "movieId,imdbId,tmdbId\n10,0000010,100\n11,0000011,100\n"
                "20,0000020,200\n30,0000030,300\n40,0000040,400\n"
            ),
            "ratings.csv": (
                "userId,movieId,rating,timestamp\n1,11,4.0,10\n2,20,5.0,20\n"
                "3,30,4.0,30\n3,10,4.5,40\n1,30,1.0,50\n"
                "1,20,5.0,100\n2,30,4.0,110\n9,20,4.0,120\n4,40,5.0,200\n"
            ),
            "tags.csv": "userId,movieId,tag,timestamp\n",
        }
        for name, text in files.items():
            (self.data_dir / name).write_text(text, encoding="utf-8")
        self.evaluation_path.write_text(json.dumps(self.evaluation), encoding="utf-8")
        self.als_path.write_text(json.dumps(self.experiments), encoding="utf-8")

    def run_experiment(self, artifact_dir=None):
        return run_experiment(self.data_dir, self.evaluation_path, self.als_path, artifact_dir)

    def test_experiment_scores_the_common_cohort_and_saves_a_self_contained_candidate(self):
        before = {path: path.read_bytes() for path in self.data_dir.iterdir()}
        artifact = self.directory / "candidate"

        report = self.run_experiment(artifact)

        self.assertEqual(report["training_matrix"]["users"], 3)
        self.assertEqual(report["training_matrix"]["movies"], 3)
        self.assertEqual(report["training_matrix"]["positive_pairs"], 4)
        self.assertEqual(report["coverage"]["eligible_users"], 2)
        self.assertEqual(report["selection"]["fit_split"], "train")
        self.assertIs(report["selection"]["final_test_evaluated"], False)
        self.assertEqual(set(report["models"]), {"als", "genres", "popularity"})
        self.assertEqual(len(report["trials"]), 2)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        for model in report["models"].values():
            self.assertEqual([row["user_id"] for row in model["per_user"]], [1, 2])
        selected = select_trial(report["trials"])
        self.assertEqual(report["selection"]["selected_trial"], selected["name"])
        self.assertEqual(report["models"]["als"], selected["evaluation"])
        loaded = ImplicitRecommender.load(artifact)
        self.assertEqual(loaded.metadata["selection"], report["selection"])
        self.assertEqual(loaded.training_data.tmdb_ids, (100, 200, 300))
        self.assertEqual(loaded.training_data.movie_ids, (10, 20, 30))
        for row, favorites in zip(report["models"]["als"]["per_user"], ([100], [200])):
            predicted = loaded.recommend(favorites)
            self.assertEqual([item["tmdb_id"] for item in predicted], row["recommended_tmdb_ids"])
        self.assertEqual({path: path.read_bytes() for path in self.data_dir.iterdir()}, before)

    def test_repeating_the_same_experiment_reproduces_scores_and_selection(self):
        first = self.run_experiment()
        self.assertEqual(self.run_experiment(), first)

    def test_future_test_ratings_cannot_change_candidates_profiles_scores_or_selection(self):
        first = self.run_experiment()
        ratings = self.data_dir / "ratings.csv"
        with ratings.open("a", encoding="utf-8") as stream:
            stream.write("5,40,5.0,210\n6,20,4.0,220\n7,10,1.0,230\n")

        second = self.run_experiment()

        for field in ("models", "trials", "selection", "coverage", "training_matrix"):
            self.assertEqual(second[field], first[field], field)
        self.assertNotEqual(second["sources"]["ratings.csv"], first["sources"]["ratings.csv"])

    def test_existing_artifact_is_preserved_and_no_training_is_attempted(self):
        artifact = self.directory / "candidate"
        artifact.mkdir()
        marker = artifact / "keep.txt"
        marker.write_text("existing model", encoding="utf-8")
        self.als_path.unlink()
        with self.assertRaises(FileExistsError):
            self.run_experiment(artifact)
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing model")

    def test_no_eligible_validation_users_cannot_select_or_save_a_model(self):
        self.evaluation["train_end"] = "1970-01-01T00:03:00Z"
        self.evaluation_path.write_text(json.dumps(self.evaluation), encoding="utf-8")
        artifact = self.directory / "candidate"
        with self.assertRaisesRegex(ValueError, "eligible validation"):
            self.run_experiment(artifact)
        self.assertFalse(artifact.exists())

    def test_invalid_trial_declarations_fail_before_loading_or_training(self):
        invalid = [
            [], {"trials": []}, {"k": 10, "trials": []},
            {**self.experiments, "k": True},
            {**self.experiments, "k": 0},
            {**self.experiments, "trials": [self.experiments["trials"][0]] * 2},
            {**self.experiments, "trials": [{"name": "incomplete", "factors": 2}]},
            {**self.experiments, "trials": [{**self.experiments["trials"][0], "alpha": -1}]},
        ]
        for config in invalid:
            with self.subTest(config=config):
                self.als_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_experiment_config(self.als_path)


if __name__ == "__main__":
    unittest.main()

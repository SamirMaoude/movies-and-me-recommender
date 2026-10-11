"""Small end-to-end ALS experiments, selection rules and held-out data guards."""

import hashlib
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import mlflow
import mlflow.tracking.fluent as fluent
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking import MlflowClient

from src.models.collaborative import ImplicitRecommender
from src.training.train import run_experiment, select_trial


def trial_result(name, ndcg, recall):
    return {
        "name": name,
        "evaluation": {"metrics": {"ndcg_at_10": ndcg, "recall_at_10": recall}},
    }


def reproducible_report(report):
    """Keep every result except identities and timings unique to an execution."""
    normalized = json.loads(json.dumps(report))
    normalized["selection"].pop("selected_run_id")
    for trial in normalized["trials"]:
        trial.pop("run_id")
        trial.pop("training_seconds")
    return normalized


class TrialSelectionTests(unittest.TestCase):
    def test_ndcg_is_primary_even_when_another_trial_has_higher_recall(self):
        trials = [
            trial_result("higher_recall", 0.2, 0.8),
            trial_result("higher_ndcg", 0.3, 0.4),
        ]
        self.assertIs(select_trial(trials), trials[1])

    def test_recall_breaks_ndcg_ties_then_the_first_declared_trial_wins(self):
        trials = [trial_result("first", 0.3, 0.2), trial_result("second", 0.3, 0.4)]
        self.assertIs(select_trial(trials), trials[1])
        trials.append(trial_result("third", 0.3, 0.4))
        self.assertIs(select_trial(trials), trials[1])


class TrainingExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.previous_tracking_uri = mlflow.get_tracking_uri()
        self.previous_experiment_id = fluent._active_experiment_id
        self.previous_experiment_environment = {
            name: os.environ.get(name)
            for name in ("MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME")
        }
        self.tracking_uri = f"sqlite:///{(self.directory / 'tracking.db').as_posix()}"
        self.artifact_root = self.directory / "mlflow-artifacts"
        self.addCleanup(self.restore_mlflow)
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment_id = self.client.create_experiment(
            "movies-and-me-recommender", artifact_location=self.artifact_root.as_uri()
        )
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
                    "name": "small",
                    "factors": 2,
                    "regularization": 0.1,
                    "alpha": 10.0,
                    "iterations": 3,
                    "random_state": 7,
                    "num_threads": 1,
                },
                {
                    "name": "larger",
                    "factors": 3,
                    "regularization": 0.1,
                    "alpha": 10.0,
                    "iterations": 3,
                    "random_state": 7,
                    "num_threads": 1,
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

    def restore_mlflow(self):
        if mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
        mlflow.set_tracking_uri(self.previous_tracking_uri)
        fluent._active_experiment_id = self.previous_experiment_id
        for name, value in self.previous_experiment_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        # Cached SQLite connections hold the database open on Windows.
        with SqlAlchemyStore._engine_map_lock:
            engine = SqlAlchemyStore._engine_map.pop(self.tracking_uri, None)
        if engine is not None:
            engine.dispose()

    def run_experiment(self, artifact_dir=None):
        return run_experiment(
            self.data_dir, self.evaluation_path, self.als_path, artifact_dir
        )

    def test_experiment_scores_the_common_cohort_and_saves_a_self_contained_candidate(
        self,
    ):
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
            self.assertEqual(
                [item["tmdb_id"] for item in predicted], row["recommended_tmdb_ids"]
            )
        self.assertEqual(
            {path: path.read_bytes() for path in self.data_dir.iterdir()}, before
        )

    def test_repeating_the_same_experiment_reproduces_scores_and_selection(self):
        first = self.run_experiment()
        second = self.run_experiment()
        self.assertEqual(reproducible_report(second), reproducible_report(first))
        self.assertTrue(
            {trial["run_id"] for trial in first["trials"]}.isdisjoint(
                trial["run_id"] for trial in second["trials"]
            )
        )

    def test_future_test_ratings_cannot_change_candidates_profiles_scores_or_selection(
        self,
    ):
        first = self.run_experiment()
        ratings = self.data_dir / "ratings.csv"
        with ratings.open("a", encoding="utf-8") as stream:
            stream.write("5,40,5.0,210\n6,20,4.0,220\n7,10,1.0,230\n")

        second = self.run_experiment()

        stable_first = reproducible_report(first)
        stable_second = reproducible_report(second)
        for field in ("models", "trials", "selection", "coverage", "training_matrix"):
            self.assertEqual(stable_second[field], stable_first[field], field)
        self.assertNotEqual(
            second["sources"]["ratings.csv"], first["sources"]["ratings.csv"]
        )

    def test_mlflow_runs_link_provenance_models_scores_and_the_selected_report(self):
        candidate = self.directory / "candidate"
        report = self.run_experiment(candidate)
        selected = select_trial(report["trials"])
        selected_run_id = selected["run_id"]
        self.assertEqual(report["selection"]["selected_run_id"], selected_run_id)
        self.assertIsNone(mlflow.active_run())
        runs = self.client.search_runs([self.experiment_id])
        self.assertEqual(len(runs), len(self.experiments["trials"]))
        self.assertEqual(
            {run.info.run_id for run in runs},
            {trial["run_id"] for trial in report["trials"]},
        )
        expected_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
        self.assertEqual(report["git"]["commit"], expected_commit)
        self.assertIsInstance(report["git"]["dirty"], bool)
        self.assertEqual(report["environment"]["mlflow"], mlflow.__version__)
        for name in ("ratings.csv", "links.csv", "movies.csv"):
            self.assertEqual(
                report["sources"][name]["sha256"],
                hashlib.sha256((self.data_dir / name).read_bytes()).hexdigest(),
            )

        provenance = {name: report[name] for name in ("git", "sources", "environment")}
        local_model = ImplicitRecommender.load(candidate)
        self.assertEqual(local_model.metadata["run_id"], selected_run_id)
        for name, value in provenance.items():
            self.assertEqual(local_model.metadata[name], value)

        for trial in report["trials"]:
            with self.subTest(trial=trial["name"]):
                run_id = trial["run_id"]
                run = self.client.get_run(run_id)
                self.assertEqual(run.info.status, "FINISHED")
                self.assertEqual(run.data.tags["mlflow.runName"], trial["name"])
                self.assertEqual(run.data.tags["git.commit"], expected_commit)
                self.assertEqual(
                    run.data.tags["git.dirty"], str(report["git"]["dirty"]).lower()
                )
                expected_params = {**trial["config"], **self.evaluation, "k": 10}
                for name, value in expected_params.items():
                    self.assertEqual(run.data.params[name], str(value))
                self.assertGreaterEqual(trial["training_seconds"], 0)
                self.assertTrue(math.isfinite(trial["training_seconds"]))
                self.assertEqual(
                    run.data.metrics["training_seconds"], trial["training_seconds"]
                )
                for name, value in trial["evaluation"]["metrics"].items():
                    self.assertEqual(run.data.metrics[f"validation_{name}"], value)
                self.assertEqual(run.data.metrics["validation_users"], 2)

                artifact_names = {
                    item.path for item in self.client.list_artifacts(run_id)
                }
                self.assertTrue(
                    {"evaluation.json", "provenance.json", "model"} <= artifact_names
                )
                self.assertEqual(
                    "comparison.json" in artifact_names, run_id == selected_run_id
                )
                destination = self.directory / f"download-{run_id}"
                destination.mkdir()
                download = lambda path: Path(  # noqa: E731
                    self.client.download_artifacts(run_id, path, str(destination))  # noqa: B023
                )
                stored_provenance = json.loads(download("provenance.json").read_text())
                for name, value in provenance.items():
                    self.assertEqual(stored_provenance[name], value)
                self.assertEqual(
                    json.loads(download("evaluation.json").read_text()),
                    trial["evaluation"],
                )
                model_path = download("model")
                self.assertEqual(
                    {path.name for path in model_path.iterdir()},
                    {"model.npz", "interactions.npz", "catalog.json", "manifest.json"},
                )
                recovered = ImplicitRecommender.load(model_path)
                self.assertEqual(recovered.metadata["run_id"], run_id)
                self.assertEqual(recovered.metadata["trial"], trial["name"])
                self.assertEqual(recovered.metadata["fit_split"], "train")
                self.assertEqual(
                    recovered.metadata["evaluation_config"], self.evaluation
                )
                self.assertEqual(
                    recovered.metadata["validation_metrics"],
                    trial["evaluation"]["metrics"],
                )
                for name, value in provenance.items():
                    self.assertEqual(recovered.metadata[name], value)
                for row, favorites in zip(
                    trial["evaluation"]["per_user"], ([100], [200])
                ):
                    self.assertEqual(
                        [item["tmdb_id"] for item in recovered.recommend(favorites)],
                        row["recommended_tmdb_ids"],
                    )
                if run_id == selected_run_id:
                    self.assertEqual(
                        json.loads(download("comparison.json").read_text()), report
                    )
                    for favorites in ([100], [200], [], [999]):
                        local = local_model.recommend(favorites)
                        restored = recovered.recommend(favorites)
                        self.assertEqual(
                            [item["tmdb_id"] for item in restored],
                            [item["tmdb_id"] for item in local],
                        )
                        for actual, expected in zip(restored, local):
                            self.assertAlmostEqual(
                                actual["score"], expected["score"], places=6
                            )

    def test_artifact_logging_failure_marks_the_run_failed_and_saves_no_candidate(self):
        self.artifact_root.write_text(
            "This file prevents creating artifact directories."
        )
        candidate = self.directory / "candidate"
        with self.assertRaises((OSError, MlflowException)):
            self.run_experiment(candidate)
        self.assertFalse(candidate.exists())
        self.assertIsNone(mlflow.active_run())
        runs = self.client.search_runs([self.experiment_id])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].info.status, "FAILED")

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


if __name__ == "__main__":
    unittest.main()

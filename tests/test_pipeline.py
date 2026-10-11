"""Exercise the DVC stage outputs against small MovieLens inputs and MLflow."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import mlflow
import mlflow.tracking.fluent as fluent
import yaml
from mlflow.store.model_registry.sqlalchemy_store import (
    SqlAlchemyStore as RegistryStore,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking import MlflowClient

from src.evaluation.prepare import build_report
from src.models.collaborative import ImplicitRecommender
from src.pipeline.run import prepare_stage, train_stage


class PipelineStageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pipeline-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.raw = self.directory / "raw"
        self.raw.mkdir()
        self.params_path = self.directory / "params.yaml"
        self.evaluation_path = self.directory / "artifacts/evaluation.json"
        self.audit_path = self.directory / "reports/data.json"
        self.model_dir = self.directory / "artifacts/model"
        self.report_path = self.directory / "reports/comparison.json"
        self.metrics_path = self.directory / "reports/metrics.json"
        self.tracking_uri = f"sqlite:///{(self.directory / 'tracking.db').as_posix()}"
        self.previous_uri = mlflow.get_tracking_uri()
        self.previous_experiment = fluent._active_experiment_id
        self.previous_environment = {
            name: os.environ.get(name)
            for name in (
                "MLFLOW_TRACKING_URI",
                "MLFLOW_EXPERIMENT_ID",
                "MLFLOW_EXPERIMENT_NAME",
            )
        }
        for name in self.previous_environment:
            os.environ.pop(name, None)
        self.addCleanup(self.restore_mlflow)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment_id = self.client.create_experiment(
            "movies-and-me-recommender",
            artifact_location=(self.directory / "mlflow-artifacts").as_uri(),
        )
        self.params = {
            "evaluation": {
                "positive_threshold": 4.0,
                "train_end": "1970-01-01T00:01:40Z",
                "validation_end": "1970-01-01T00:03:20Z",
            },
            "als": {
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
                    }
                ],
            },
            "tracking": {"uri": self.tracking_uri},
        }
        self.write_params()
        files = {
            "movies.csv": "movieId,title,genres\n10,First,Drama\n20,Second,Drama\n30,Third,Comedy\n40,Future,Action\n",
            "links.csv": "movieId,imdbId,tmdbId\n10,0000010,100\n20,0000020,200\n30,0000030,300\n40,0000040,400\n",
            "ratings.csv": (
                "userId,movieId,rating,timestamp\n1,10,4.0,10\n2,20,5.0,20\n"
                "3,30,4.0,30\n3,10,4.5,40\n1,30,1.0,50\n"
                "1,20,5.0,100\n2,30,4.0,110\n9,20,4.0,120\n4,40,5.0,200\n"
            ),
            "tags.csv": "userId,movieId,tag,timestamp\n",
        }
        for name, content in files.items():
            (self.raw / name).write_text(content, encoding="utf-8")

    def restore_mlflow(self):
        if mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
        mlflow.set_tracking_uri(self.previous_uri)
        fluent._active_experiment_id = self.previous_experiment
        for name, value in self.previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for store in (SqlAlchemyStore, RegistryStore):
            with store._engine_map_lock:
                engine = store._engine_map.pop(self.tracking_uri, None)
            if engine is not None:
                engine.dispose()

    def write_params(self):
        self.params_path.write_text(yaml.safe_dump(self.params), encoding="utf-8")

    def prepare(self):
        return prepare_stage(
            self.params_path, self.raw, self.evaluation_path, self.audit_path
        )

    def train(self):
        return train_stage(
            self.params_path,
            self.raw,
            self.evaluation_path,
            self.model_dir,
            self.report_path,
            self.metrics_path,
        )

    def test_preparation_exports_the_existing_audit_without_modifying_raw_files(self):
        before = {path.name: path.read_bytes() for path in self.raw.iterdir()}

        report = self.prepare()

        self.assertEqual(report, build_report(self.raw, self.evaluation_path))
        self.assertEqual(json.loads(self.audit_path.read_text()), report)
        self.assertEqual(
            json.loads(self.evaluation_path.read_text()), self.params["evaluation"]
        )
        self.assertEqual(report["splits"]["train"]["interactions"], 5)
        self.assertEqual(report["splits"]["validation"]["interactions"], 3)
        self.assertEqual(report["splits"]["test"]["interactions"], 1)
        self.assertEqual(
            before, {path.name: path.read_bytes() for path in self.raw.iterdir()}
        )
        self.assertEqual(self.client.search_runs([self.experiment_id]), [])

    def test_training_produces_reloadable_model_comparison_and_numeric_validation_metrics(
        self,
    ):
        self.prepare()

        report = self.train()

        self.assertEqual(json.loads(self.report_path.read_text()), report)
        self.assertEqual(mlflow.get_tracking_uri(), self.previous_uri)
        metrics = json.loads(self.metrics_path.read_text())
        self.assertNotIn(b"\r\n", self.metrics_path.read_bytes())
        self.assertEqual(set(metrics), {"als", "popularity", "genres"})
        for name, values in metrics.items():
            self.assertEqual(values, report["models"][name]["metrics"])
            self.assertTrue(all(isinstance(value, float) for value in values.values()))
        model = ImplicitRecommender.load(self.model_dir)
        self.assertEqual(
            model.metadata["run_id"], report["selection"]["selected_run_id"]
        )
        self.assertEqual(model.training_data.tmdb_ids, (100, 200, 300))
        self.assertFalse(report["selection"]["final_test_evaluated"])
        self.assertEqual(report["selection"]["fit_split"], "train")
        self.assertEqual(self.client.search_registered_models(), [])
        runs = self.client.search_runs([self.experiment_id])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].info.status, "FINISHED")
        for row in report["models"]["als"]["per_user"]:
            favorites = [100] if row["user_id"] == 1 else [200]
            self.assertEqual(
                [item["tmdb_id"] for item in model.recommend(favorites)],
                row["recommended_tmdb_ids"],
            )

    def test_environment_can_select_the_tracking_backend_and_metrics_remain_stable(
        self,
    ):
        self.prepare()
        first = self.train()
        first_metrics = self.metrics_path.read_bytes()
        self.params["tracking"]["uri"] = "sqlite:///must-not-be-created.db"
        self.write_params()
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        self.model_dir = self.directory / "artifacts/second-model"
        self.report_path = self.directory / "reports/second-comparison.json"
        self.metrics_path = self.directory / "reports/second-metrics.json"

        second = self.train()

        self.assertNotEqual(
            first["selection"]["selected_run_id"],
            second["selection"]["selected_run_id"],
        )
        self.assertEqual(first_metrics, self.metrics_path.read_bytes())
        self.assertEqual(len(self.client.search_runs([self.experiment_id])), 2)
        self.assertFalse((Path.cwd() / "must-not-be-created.db").exists())

    def test_existing_outputs_fail_before_reading_configuration_or_starting_runs(self):
        self.params_path.unlink()
        for output in (self.model_dir, self.report_path, self.metrics_path):
            with self.subTest(output=output):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("keep this output", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    self.train()
                self.assertEqual(output.read_text(), "keep this output")
                output.unlink()
        self.evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        self.evaluation_path.write_text("keep this configuration", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(self.evaluation_path.read_text(), "keep this configuration")
        self.assertEqual(self.client.search_runs([self.experiment_id]), [])

    def test_training_rejects_stale_preparation_before_loading_data(self):
        self.prepare()
        self.params["evaluation"]["positive_threshold"] = 4.5
        self.write_params()
        (self.raw / "ratings.csv").unlink()

        with self.assertRaisesRegex(ValueError, "differs from params.yaml"):
            self.train()

        self.assertFalse(self.model_dir.exists())
        self.assertFalse(self.report_path.exists())
        self.assertEqual(self.client.search_runs([self.experiment_id]), [])

    def test_invalid_evaluation_leaves_no_partial_preparation(self):
        for updates in (
            {"positive_threshold": True},
            {"positive_threshold": 6.0},
            {"train_end": "1970-01-01"},
            {"train_end": "1971-01-01T00:00:00Z"},
        ):
            with self.subTest(updates=updates):
                params = {
                    **self.params,
                    "evaluation": {**self.params["evaluation"], **updates},
                }
                self.params_path.write_text(yaml.safe_dump(params), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.evaluation_path.exists())
                self.assertFalse(self.audit_path.exists())

    def test_invalid_trials_fail_without_creating_models_or_runs(self):
        self.prepare()
        self.params["als"]["trials"][0]["alpha"] = -1
        self.write_params()

        with self.assertRaises(ValueError):
            self.train()

        self.assertFalse(self.model_dir.exists())
        self.assertFalse(self.report_path.exists())
        self.assertFalse(self.metrics_path.exists())
        self.assertEqual(self.client.search_runs([self.experiment_id]), [])


if __name__ == "__main__":
    unittest.main()

"""Real SQLite registry versions, native ALS reloads and safe alias updates."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import urlsplit
from urllib.request import url2pathname

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from src.models.collaborative import ALSConfig, ImplicitRecommender
from src.models.registry import load_recommender, register_model


class ModelRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        links = pd.DataFrame({
            "movieId": [10, 20, 30, 40, 50],
            "tmdbId": [100, 200, 300, 400, 500],
        })
        train = pd.DataFrame([
            (1, 10, 100, True), (1, 20, 200, True),
            (2, 10, 100, True), (2, 30, 300, True),
            (3, 40, 400, True), (3, 50, 500, True),
        ], columns=["userId", "movieId", "tmdbId", "is_positive"])
        cls.models = [
            ImplicitRecommender.fit(
                train, links,
                ALSConfig(factors=3, iterations=3, random_state=seed),
            )
            for seed in (42, 7)
        ]

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.tracking_uri = f"sqlite:///{(self.directory / 'tracking.db').as_posix()}"
        self.registry_uri = self.tracking_uri
        self.extra_uris = []
        self.addCleanup(self.dispose_engines)
        self.client = MlflowClient(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )
        self.experiment_id = self.client.create_experiment(
            "registry-tests",
            artifact_location=(self.directory / "artifacts").as_uri(),
        )
        self.name = "movies-and-me-als"
        self.original_tracking_uri = mlflow.get_tracking_uri()
        self.original_registry_uri = mlflow.get_registry_uri()

    def dispose_engines(self):
        # Both SQLAlchemy stores cache connections that lock SQLite on Windows.
        for uri in {self.tracking_uri, self.registry_uri, *self.extra_uris}:
            for store in (TrackingStore, RegistryStore):
                with store._engine_map_lock:
                    engine = store._engine_map.pop(uri, None)
                if engine is not None:
                    engine.dispose()

    def log_model(self, index=0, status="FINISHED", metadata=None):
        run = self.client.create_run(self.experiment_id)
        run_id = run.info.run_id
        bundle = self.directory / f"bundle-{run_id}"
        self.models[index].save(
            bundle,
            metadata={"run_id": run_id, "trial": f"trial-{index}"}
            if metadata is None else metadata,
        )
        self.client.log_artifacts(run_id, str(bundle), artifact_path="model")
        if status != "RUNNING":
            self.client.set_terminated(run_id, status=status)
        return run_id, bundle

    def register(self, run_id, alias=None):
        return register_model(
            run_id, self.name, alias=alias,
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri,
        )

    def load(self, source):
        return load_recommender(
            source, tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )

    def assert_recommendations(self, loaded, original):
        for favorites in ([], [999], [100], [100, 300], [100, 200, 300, 400, 500]):
            with self.subTest(favorites=favorites):
                actual = loaded.recommend_with_method(favorites, n=3)
                expected = original.recommend_with_method(favorites, n=3)
                self.assertEqual(actual["method"], expected["method"])
                self.assertEqual(
                    [row["tmdb_id"] for row in actual["recommendations"]],
                    [row["tmdb_id"] for row in expected["recommendations"]],
                )
                np.testing.assert_allclose(
                    [row["score"] for row in actual["recommendations"]],
                    [row["score"] for row in expected["recommendations"]],
                    rtol=1e-6, atol=1e-7,
                )
                self.assertEqual(
                    loaded.recommend(favorites, n=3), actual["recommendations"]
                )

    def champion(self):
        return self.client.get_model_version_by_alias(self.name, "champion")

    def artifact_file(self, run_id, filename):
        uri = urlsplit(self.client.get_run(run_id).info.artifact_uri)
        self.assertEqual(uri.scheme, "file")
        return Path(url2pathname(uri.path)) / "model" / filename

    def test_local_and_run_sources_reload_native_favorites_without_csv_data(self):
        run_id, bundle = self.log_model()
        for source in (bundle, str(bundle), f"runs:/{run_id}/model"):
            with self.subTest(source=source):
                loaded = self.load(source)
                self.assertIsInstance(loaded.model, ImplicitRecommender)
                self.assertIsNone(loaded.name)
                self.assertIsNone(loaded.version)
                self.assertEqual(loaded.run_id, run_id)
                self.assert_recommendations(loaded, self.models[0])
                with self.assertRaises(FrozenInstanceError):
                    loaded.run_id = "changed"
        self.assertEqual(mlflow.get_tracking_uri(), self.original_tracking_uri)
        self.assertEqual(mlflow.get_registry_uri(), self.original_registry_uri)
        self.assertIsNone(mlflow.active_run())

    def test_registration_creates_an_exact_version_and_repeat_reuses_it(self):
        run_id, _ = self.log_model()
        registered = self.register(run_id)
        self.assertEqual(registered["name"], self.name)
        self.assertEqual(registered["version"], "1")
        self.assertEqual(registered["run_id"], run_id)
        self.assertEqual(registered["model_uri"], f"models:/{self.name}/1")
        self.assertIsNone(registered["alias"])
        self.assertEqual(self.register(run_id), registered)
        versions = self.client.search_model_versions(f"name = '{self.name}'")
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0].run_id, run_id)
        with self.assertRaises(MlflowException):
            self.champion()
        loaded = self.load(registered["model_uri"])
        self.assertEqual((loaded.name, loaded.version, loaded.run_id),
                         (self.name, "1", run_id))
        self.assertEqual(loaded.model_uri, registered["model_uri"])
        self.assert_recommendations(loaded, self.models[0])

    def test_explicit_champion_changes_leave_old_versions_and_loaded_aliases_stable(self):
        first_run, _ = self.log_model(0)
        first = self.register(first_run, alias="champion")
        self.assertEqual(first["alias"], "champion")
        pinned = self.load(f"models:/{self.name}@champion")
        self.assertEqual(pinned.model_uri, first["model_uri"])

        second_run, _ = self.log_model(1)
        second = self.register(second_run)
        self.assertEqual(second["version"], "2")
        self.assertEqual(str(self.champion().version), "1")
        promoted = self.register(second_run, alias="champion")
        self.assertEqual(promoted["version"], "2")
        self.assertEqual(str(self.champion().version), "2")
        self.assertEqual(len(self.client.search_model_versions(f"name = '{self.name}'")), 2)

        current = self.load(f"models:/{self.name}@champion")
        self.assertEqual(current.model_uri, second["model_uri"])
        self.assertEqual(current.run_id, second_run)
        self.assert_recommendations(current, self.models[1])
        self.assertEqual((pinned.version, pinned.run_id), ("1", first_run))
        self.assert_recommendations(pinned, self.models[0])
        self.assert_recommendations(self.load(first["model_uri"]), self.models[0])

    def test_unfinished_and_failed_runs_cannot_replace_champion(self):
        good_run, _ = self.log_model()
        self.register(good_run, alias="champion")
        for status in ("RUNNING", "FAILED", "KILLED"):
            run_id, _ = self.log_model(status=status)
            with self.subTest(status=status):
                with self.assertRaises((ValueError, MlflowException)):
                    self.register(run_id, alias="champion")
                self.assertEqual(self.champion().run_id, good_run)
        self.assertEqual(len(self.client.search_model_versions(f"name = '{self.name}'")), 1)

    def test_corrupt_candidate_is_rejected_before_registration_and_alias_change(self):
        good_run, _ = self.log_model()
        self.register(good_run, alias="champion")
        broken_run, _ = self.log_model(1)
        corrupted = self.artifact_file(broken_run, "model.npz")
        corrupted.write_bytes(corrupted.read_bytes() + b"changed")

        with self.assertRaisesRegex(ValueError, "checksum"):
            self.register(broken_run, alias="champion")

        self.assertEqual(self.champion().run_id, good_run)
        self.assertEqual(len(self.client.search_model_versions(f"name = '{self.name}'")), 1)

    def test_idempotent_reuse_rechecks_integrity_before_champion_assignment(self):
        good_run, _ = self.log_model()
        self.register(good_run, alias="champion")
        other_run, _ = self.log_model(1)
        other = self.register(other_run)
        corrupted = self.artifact_file(other_run, "model.npz")
        corrupted.write_bytes(corrupted.read_bytes() + b"changed")

        with self.assertRaisesRegex(ValueError, "checksum"):
            self.register(other_run, alias="champion")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load(other["model_uri"])

        self.assertEqual(self.champion().run_id, good_run)
        self.assertEqual(len(self.client.search_model_versions(f"name = '{self.name}'")), 2)

    def test_manifest_must_name_the_run_that_owns_the_bundle(self):
        good_run, _ = self.log_model()
        self.register(good_run, alias="champion")
        for metadata in ({}, {"run_id": good_run}, {"run_id": 123}):
            run_id, _ = self.log_model(metadata=metadata)
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    self.register(run_id, alias="champion")
                self.assertEqual(self.champion().run_id, good_run)
        self.assertEqual(len(self.client.search_model_versions(f"name = '{self.name}'")), 1)

    def test_missing_bundle_cannot_create_a_registry_version(self):
        run = self.client.create_run(self.experiment_id)
        self.client.set_terminated(run.info.run_id)
        with self.assertRaises((ValueError, OSError, MlflowException)):
            self.register(run.info.run_id)
        self.assertEqual(self.client.search_registered_models(), [])

    def test_missing_name_version_alias_or_run_reports_failure(self):
        run_id, _ = self.log_model()
        self.register(run_id)
        for source in (
            "models:/missing/1", f"models:/{self.name}/999",
            f"models:/{self.name}@champion", "runs:/00000000000000000000000000000000/model",
        ):
            with self.subTest(source=source):
                with self.assertRaises((ValueError, MlflowException)):
                    self.load(source)

    def test_invalid_source_forms_and_versions_are_rejected(self):
        for source in (
            None, True, 12, "", "models:/", "models:/name", "models:/name/0",
            "models:/name/-1", "models:/name/1.5", "models:/name/latest",
            "models:/name@", "models:/name/1/extra", "runs:/", "runs:/id",
            "https://example.com/model", "s3://bucket/model",
        ):
            with self.subTest(source=source):
                with self.assertRaises((ValueError, MlflowException)):
                    self.load(source)

    def test_tracking_and_registry_can_use_separate_databases_without_global_changes(self):
        self.registry_uri = f"sqlite:///{(self.directory / 'registry.db').as_posix()}"
        self.client = MlflowClient(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )
        run_id, _ = self.log_model()
        registered = self.register(run_id, alias="champion")

        loaded = self.load(f"models:/{self.name}@champion")

        self.assertEqual(loaded.model_uri, registered["model_uri"])
        self.assertEqual(loaded.run_id, run_id)
        self.assert_recommendations(loaded, self.models[0])
        self.assertEqual(mlflow.get_tracking_uri(), self.original_tracking_uri)
        self.assertEqual(mlflow.get_registry_uri(), self.original_registry_uri)
        same_database_client = MlflowClient(
            tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri
        )
        self.assertEqual(same_database_client.search_registered_models(), [])

    def test_cli_registers_a_version_and_recommends_from_an_alias_as_json(self):
        run_id, _ = self.log_model()
        common = [
            "--tracking-uri", self.tracking_uri,
            "--registry-uri", self.registry_uri,
        ]
        command = [sys.executable, "-m", "src.models.registry"]
        options = {
            "cwd": Path(__file__).resolve().parents[1],
            "capture_output": True,
            "text": True,
            "timeout": 60,
        }
        registered_process = subprocess.run(
            [*command, "register", "--run-id", run_id, "--alias", "champion", *common],
            **options,
        )
        self.assertEqual(registered_process.returncode, 0, registered_process.stderr)
        registered = json.loads(registered_process.stdout)
        self.assertEqual(registered, {
            "name": self.name, "version": "1", "run_id": run_id,
            "model_uri": f"models:/{self.name}/1", "alias": "champion",
        })

        recommendations_process = subprocess.run(
            [*command, "recommend", "--model-uri", f"models:/{self.name}@champion",
             "--favorites", "100", "300", "--n", "2", *common],
            **options,
        )
        self.assertEqual(
            recommendations_process.returncode, 0, recommendations_process.stderr
        )
        recommendations = json.loads(recommendations_process.stdout)
        for name in ("name", "version", "run_id", "model_uri"):
            self.assertEqual(recommendations[name], registered[name])
        expected = self.models[0].recommend_with_method([100, 300], n=2)
        self.assertEqual(recommendations["method"], expected["method"])
        self.assertEqual(
            [row["tmdb_id"] for row in recommendations["recommendations"]],
            [row["tmdb_id"] for row in expected["recommendations"]],
        )
        np.testing.assert_allclose(
            [row["score"] for row in recommendations["recommendations"]],
            [row["score"] for row in expected["recommendations"]],
            rtol=1e-6, atol=1e-7,
        )

        invalid_process = subprocess.run(
            [*command, "recommend", "--model-uri", registered["model_uri"],
             "--n", "0", *common],
            **options,
        )
        self.assertNotEqual(invalid_process.returncode, 0)
        self.assertEqual(str(self.champion().version), "1")


if __name__ == "__main__":
    unittest.main()

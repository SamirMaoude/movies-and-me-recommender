"""Register native ALS bundles and load a pinned model for evaluation or serving."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

from src.models.collaborative import ImplicitRecommender

# MLflow takes seconds to import, so it is imported where it is used: the API
# serves a local bundle and never needs it.
if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

DEFAULT_MODEL_NAME = "movies-and-me-als"


@dataclass(frozen=True)
class LoadedRecommender:
    """A loaded model and the registry identity it was resolved from."""

    model: ImplicitRecommender
    model_uri: str
    name: str | None = None
    version: str | None = None
    run_id: str | None = None

    def recommend(self, favorites: Iterable[int], n: int = 10) -> list[dict]:
        return self.model.recommend(favorites, n=n)

    def recommend_with_method(self, favorites: Iterable[int], n: int = 10) -> dict:
        return self.model.recommend_with_method(favorites, n=n)


def _client(tracking_uri: str | None, registry_uri: str | None) -> MlflowClient:
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)


def _load_run_model(client: MlflowClient, uri: str, run_id: str) -> ImplicitRecommender:
    """Load a bundle logged by a finished run, whose manifest must name that run."""
    from mlflow.artifacts import download_artifacts

    if client.get_run(run_id).info.status != "FINISHED":
        raise ValueError("Only a FINISHED MLflow run can supply a model.")
    # Load all arrays before the temporary download directory is removed.
    with TemporaryDirectory(prefix="als-download-") as temporary:
        directory = download_artifacts(
            artifact_uri=uri, dst_path=temporary, tracking_uri=client.tracking_uri
        )
        model = ImplicitRecommender.load(directory)
    if model.metadata.get("run_id") != run_id:
        raise ValueError("The model manifest does not belong to the requested run.")
    return model


def load_recommender(
    source: str | Path,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> LoadedRecommender:
    """Load a local bundle, a run's bundle, or a registered version or alias."""
    uri = str(source)
    if uri.startswith("models:/"):
        client = _client(tracking_uri, registry_uri)
        if match := re.fullmatch(r"models:/([^/@?#]+)/([1-9]\d*)", uri):
            name, version = match.groups()
            registered = client.get_model_version(name, version)
        elif match := re.fullmatch(r"models:/([^/@?#]+)@([^/@?#]+)", uri):
            name, alias = match.groups()
            registered = client.get_model_version_by_alias(name, alias)
        else:
            raise ValueError(
                "Use models:/<name>/<positive version> or models:/<name>@<alias>."
            )
        if registered.status != "READY" or not registered.run_id:
            raise ValueError("The registered model must be READY and linked to a run.")
        # Pin the version an alias resolves to now, and download it with this
        # client: the registry can use a different backend from tracking.
        version = str(registered.version)
        download_uri = client.get_model_version_download_uri(name, version)
        model = _load_run_model(client, download_uri, registered.run_id)
        return LoadedRecommender(
            model, f"models:/{name}/{version}", name, version, registered.run_id
        )
    if uri.startswith("runs:/"):
        match = re.fullmatch(r"runs:/([0-9a-fA-F]{32})/model", uri)
        if match is None:
            raise ValueError("Use runs:/<run_id>/model for a logged ALS bundle.")
        client = _client(tracking_uri, registry_uri)
        model = _load_run_model(client, uri, match.group(1))
        return LoadedRecommender(model, uri, run_id=match.group(1))
    directory = Path(uri).resolve()
    model = ImplicitRecommender.load(directory)
    return LoadedRecommender(model, str(directory), run_id=model.metadata.get("run_id"))


def _existing_version(client: MlflowClient, name: str, run_id: str, source: str):
    page_token = None
    while True:
        versions = client.search_model_versions(
            filter_string=f"run_id = '{run_id}'",
            max_results=1000,
            page_token=page_token,
        )
        for registered in versions:
            if registered.name == name and registered.source == source:
                if registered.status != "READY":
                    raise ValueError("An existing version for this run is not READY.")
                return registered
        page_token = versions.token
        if not page_token:
            return None


def _check_recommendations(
    expected: ImplicitRecommender, actual: ImplicitRecommender
) -> None:
    """Check that a reloaded model ranks a few probe profiles like the original."""
    catalog = expected.training_data.tmdb_ids
    profiles = ([catalog[0]], [catalog[-1]], [], [max(catalog) + 1])
    for favorites in profiles:
        before = expected.recommend_with_method(favorites)
        after = actual.recommend_with_method(favorites)
        if before["method"] != after["method"] or (
            [row["tmdb_id"] for row in before["recommendations"]]
            != [row["tmdb_id"] for row in after["recommendations"]]
        ):
            raise ValueError(
                "Registered model recommendations differ from the run bundle."
            )
        if not np.allclose(
            [row["score"] for row in before["recommendations"]],
            [row["score"] for row in after["recommendations"]],
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError("Registered model scores differ from the run bundle.")


def register_model(
    run_id: str,
    name: str = DEFAULT_MODEL_NAME,
    alias: str | None = None,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> dict:
    """Register a run's bundle, verify the registered copy, then move the alias."""
    from mlflow.exceptions import MlflowException

    client = _client(tracking_uri, registry_uri)
    source = f"runs:/{run_id}/model"
    # Loading the runs:/ URI first also validates run_id before the search
    # filter below interpolates it.
    original = load_recommender(
        source, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    try:
        client.create_registered_model(
            name,
            description="Native implicit ALS model trained from simulated TMDB favorites.",
        )
    except MlflowException as exception:
        if exception.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    # A retry reuses the version already created for this run.
    registered = _existing_version(client, name, run_id, source)
    if registered is None:
        registered = client.create_model_version(
            name,
            source=source,
            run_id=run_id,
            await_creation_for=30,
            tags={"format": "implicit-als-bundle", "validation_status": "pending"},
            description="Train-only ALS candidate; validation metrics remain linked to its run.",
        )
    model_uri = f"models:/{name}/{registered.version}"
    recovered = load_recommender(
        model_uri, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    _check_recommendations(original.model, recovered.model)
    client.set_model_version_tag(
        name, registered.version, "validation_status", "passed"
    )
    if alias is not None:
        client.set_registered_model_alias(name, alias, registered.version)
    return {
        "name": name,
        "version": str(registered.version),
        "run_id": run_id,
        "model_uri": model_uri,
        "alias": alias,
    }


def main() -> None:
    from mlflow.exceptions import MlflowException

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser(
        "register", help="Register an existing, finished training run."
    )
    register.add_argument("--run-id", required=True)
    register.add_argument("--name", default=DEFAULT_MODEL_NAME)
    register.add_argument("--alias")
    recommend = commands.add_parser(
        "recommend", help="Load one model and recommend from TMDB favorites."
    )
    recommend.add_argument("--model-uri", required=True)
    recommend.add_argument("--favorites", type=int, nargs="*", default=[])
    recommend.add_argument("--n", type=int, default=10)
    for command in (register, recommend):
        command.add_argument("--tracking-uri")
        command.add_argument("--registry-uri")
    args = parser.parse_args()
    if args.command == "recommend" and args.n < 1:
        recommend.error("--n must be at least 1")
    try:
        if args.command == "register":
            result = register_model(
                args.run_id,
                args.name,
                args.alias,
                tracking_uri=args.tracking_uri,
                registry_uri=args.registry_uri,
            )
        else:
            loaded = load_recommender(
                args.model_uri,
                tracking_uri=args.tracking_uri,
                registry_uri=args.registry_uri,
            )
            result = {
                "name": loaded.name,
                "version": loaded.version,
                "run_id": loaded.run_id,
                "model_uri": loaded.model_uri,
                **loaded.recommend_with_method(args.favorites, n=args.n),
            }
    except (OSError, ValueError, MlflowException) as exception:
        parser.exit(1, f"{exception}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

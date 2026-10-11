"""Register native ALS bundles and load a pinned model for evaluation or serving."""

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.models.collaborative import ImplicitRecommender


DEFAULT_MODEL_NAME = "movies-and-me-als"


@dataclass(frozen=True)
class LoadedRecommender:
    """Keep the resolved identity alongside a fully loaded, independent model."""

    model: ImplicitRecommender
    model_uri: str
    name: str | None = None
    version: str | None = None
    run_id: str | None = None

    def recommend(self, favorites: Iterable[int], n: int = 10) -> list[dict]:
        return self.model.recommend(favorites, n=n)

    def recommend_with_method(self, favorites: Iterable[int], n: int = 10) -> dict:
        return self.model.recommend_with_method(favorites, n=n)


def _check_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", run_id):
        raise ValueError("run_id must be a 32-character MLflow run ID.")


def _check_name(name: str) -> None:
    if (
        not isinstance(name, str) or not name.strip() or name != name.strip()
        or any(character in name for character in "/@?#")
    ):
        raise ValueError("Model name must be nonempty and contain no /, @, ? or #.")


def _check_alias(alias: str | None) -> None:
    if alias is not None and (
        not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", alias)
        or alias.lower() == "latest" or re.fullmatch(r"[vV]\d+", alias)
    ):
        raise ValueError("Use an alias such as champion; latest and v<number> are reserved.")


def _load_artifact(uri: str, client: MlflowClient) -> ImplicitRecommender:
    # Load all arrays before the temporary download directory is removed.
    with TemporaryDirectory(prefix="als-download-") as temporary:
        directory = download_artifacts(
            artifact_uri=uri, dst_path=temporary, tracking_uri=client.tracking_uri,
        )
        return ImplicitRecommender.load(directory)


def _check_run_model(model: ImplicitRecommender, run_id: str) -> None:
    if model.metadata.get("run_id") != run_id:
        raise ValueError("The model manifest does not belong to the requested MLflow run.")


def _require_finished_run(client: MlflowClient, run_id: str) -> None:
    if client.get_run(run_id).info.status != "FINISHED":
        raise ValueError("Only a FINISHED MLflow run can supply a model.")


def load_recommender(
    source: str | Path,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> LoadedRecommender:
    """Load a local bundle, a run artifact or an exact registered model version.

    An alias is resolved once. The returned model_uri always records the numeric
    version, and the in-memory model no longer needs MLflow or the downloaded files.
    """
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Provide a bundle path, runs:/ URI or models:/ URI.")
    uri = str(source)
    if uri.startswith("models:/"):
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        version_match = re.fullmatch(r"models:/([^/@?#]+)/([1-9]\d*)", uri)
        alias_match = re.fullmatch(r"models:/([^/@?#]+)@([^/@?#]+)", uri)
        if version_match:
            name, version = version_match.groups()
            _check_name(name)
            registered = client.get_model_version(name, version)
        elif alias_match:
            name, alias = alias_match.groups()
            _check_name(name)
            _check_alias(alias)
            registered = client.get_model_version_by_alias(name, alias)
        else:
            raise ValueError("Use models:/<name>/<positive version> or models:/<name>@<alias>.")
        if registered.status != "READY" or not registered.run_id:
            raise ValueError("The registered model must be READY and linked to a run.")
        version = str(registered.version)
        _require_finished_run(client, registered.run_id)
        # Resolve with this client before downloading: the registry can use a
        # different backend from tracking, and an alias may move concurrently.
        download_uri = client.get_model_version_download_uri(name, version)
        model = _load_artifact(download_uri, client)
        _check_run_model(model, registered.run_id)
        return LoadedRecommender(
            model, f"models:/{name}/{version}", name, version, registered.run_id,
        )
    if uri.startswith("runs:/"):
        match = re.fullmatch(r"runs:/([0-9a-fA-F]{32})/model", uri)
        if match is None:
            raise ValueError("Use runs:/<run_id>/model for a logged ALS bundle.")
        run_id = match.group(1)
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        _require_finished_run(client, run_id)
        model = _load_artifact(uri, client)
        _check_run_model(model, run_id)
        return LoadedRecommender(model, uri, run_id=run_id)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", uri) and not re.match(r"^[A-Za-z]:[\\/]", uri):
        raise ValueError("Only local paths, runs:/ URIs and models:/ URIs are supported.")
    directory = Path(source).resolve()
    model = ImplicitRecommender.load(directory)
    return LoadedRecommender(model, str(directory), run_id=model.metadata.get("run_id"))


def _existing_version(client: MlflowClient, name: str, run_id: str, source: str):
    page_token = None
    while True:
        versions = client.search_model_versions(
            filter_string=f"run_id = '{run_id}'", max_results=1000, page_token=page_token,
        )
        for registered in versions:
            if registered.name == name and registered.source == source:
                if registered.status != "READY":
                    raise ValueError("An existing version for this run is not READY.")
                return registered
        page_token = versions.token
        if not page_token:
            return None


def _check_recommendations(expected: ImplicitRecommender, actual: ImplicitRecommender) -> None:
    catalog = expected.training_data.tmdb_ids
    profiles = ([catalog[0]], [catalog[-1]], [], [max(catalog) + 1])
    for favorites in profiles:
        before = expected.recommend_with_method(favorites)
        after = actual.recommend_with_method(favorites)
        if before["method"] != after["method"] or (
            [row["tmdb_id"] for row in before["recommendations"]]
            != [row["tmdb_id"] for row in after["recommendations"]]
        ):
            raise ValueError("Registered model recommendations differ from the run bundle.")
        if not np.allclose(
            [row["score"] for row in before["recommendations"]],
            [row["score"] for row in after["recommendations"]],
            rtol=1e-6, atol=1e-7,
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
    """Register a verified run bundle; move an alias only after loading its version.

    Retrying the same name and run reuses its version. Registering without an
    alias creates a candidate and leaves existing aliases in place.
    """
    _check_run_id(run_id)
    _check_name(name)
    _check_alias(alias)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    source = f"runs:/{run_id}/model"
    original = load_recommender(
        source, tracking_uri=tracking_uri, registry_uri=registry_uri,
    )
    try:
        client.create_registered_model(
            name, description="Native implicit ALS model trained from simulated TMDB favorites.",
        )
    except MlflowException as exception:
        if exception.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    registered = _existing_version(client, name, run_id, source)
    if registered is None:
        registered = client.create_model_version(
            name, source=source, run_id=run_id, await_creation_for=30,
            tags={"format": "implicit-als-bundle", "validation_status": "pending"},
            description="Train-only ALS candidate; validation metrics remain linked to its run.",
        )
    model_uri = f"models:/{name}/{registered.version}"
    recovered = load_recommender(
        model_uri, tracking_uri=tracking_uri, registry_uri=registry_uri,
    )
    _check_recommendations(original.model, recovered.model)
    client.set_model_version_tag(name, registered.version, "validation_status", "passed")
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
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="Register an existing, finished training run.")
    register.add_argument("--run-id", required=True)
    register.add_argument("--name", default=DEFAULT_MODEL_NAME)
    register.add_argument("--alias")
    recommend = commands.add_parser("recommend", help="Load one model and recommend from TMDB favorites.")
    recommend.add_argument("--model-uri", required=True)
    recommend.add_argument("--favorites", type=int, nargs="*", default=[])
    recommend.add_argument("--n", type=int, default=10)
    for command in (register, recommend):
        command.add_argument("--tracking-uri")
        command.add_argument("--registry-uri")
    args = parser.parse_args()
    try:
        if args.command == "register":
            result = register_model(
                args.run_id, args.name, args.alias,
                tracking_uri=args.tracking_uri, registry_uri=args.registry_uri,
            )
        else:
            loaded = load_recommender(
                args.model_uri, tracking_uri=args.tracking_uri, registry_uri=args.registry_uri,
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

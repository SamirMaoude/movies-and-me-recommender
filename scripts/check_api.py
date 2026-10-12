"""Check a running API over HTTP, without cloud or third-party clients."""

import argparse
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(url: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    query = Request(
        url.rstrip("/") + path,
        data=encoded,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(query, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check(
    url: str,
    favorites: list[int],
    timeout: float,
    expected_run_id: str | None = None,
    expected_revision: str | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "No response"
    while True:
        try:
            status, body = request(url, "/health")
            if status == 200 and body == {"status": "ok"}:
                break
            last_error = f"/health returned {status}: {body}"
        except (URLError, TimeoutError, OSError) as error:
            last_error = str(error)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"API did not become ready: {last_error}")
        time.sleep(1)

    status, identity = request(url, "/model/info")
    require(status == 200 and bool(identity.get("model_uri")), "Missing model identity")
    if expected_run_id is not None:
        require(
            identity.get("run_id") == expected_run_id,
            f"Unexpected model run_id: expected {expected_run_id!r}, "
            f"got {identity.get('run_id')!r}",
        )
    details = f"model={identity}"
    if expected_revision is not None:
        status, version = request(url, "/version")
        require(status == 200, f"/version returned {status}")
        require(
            version.get("revision") == expected_revision,
            f"Unexpected revision: expected {expected_revision!r}, "
            f"got {version.get('revision')!r}",
        )
        require(version.get("model") == identity, "Version reports another model")
        details = f"revision={expected_revision}, {details}"
    status, response = request(
        url,
        "/recommendations",
        {
            "favorite_movie_ids": favorites,
            "n": 3,
        },
    )
    require(status == 200, f"Recommendations returned {status}")
    require(response.get("method") == "als", "Known favorites did not use ALS")
    require(response.get("model") == identity, "Model identity changed between routes")
    rows = response["recommendations"]
    require(0 < len(rows) <= 3, "Unexpected number of recommendations")
    identifiers = [row["tmdb_id"] for row in rows]
    require(len(set(identifiers)) == len(identifiers), "Duplicate recommendations")
    require(
        not set(identifiers).intersection(favorites),
        "Favorites returned as recommendations",
    )
    require(
        all(math.isfinite(row["score"]) for row in rows),
        "Invalid recommendation scores",
    )

    status, fallback = request(
        url, "/recommendations", {"favorite_movie_ids": [], "n": 3}
    )
    require(
        status == 200 and fallback.get("method") == "popularity",
        "Missing popularity fallback",
    )
    require(fallback.get("model") == identity, "Fallback changed model identity")
    status, _ = request(url, "/recommendations", {"favorite_movie_ids": [True]})
    require(status == 422, "Invalid favorites were not rejected")
    status, _ = request(
        url, "/recommendations", {"favorite_movie_ids": favorites, "n": 0}
    )
    require(status == 422, "Invalid limit was not rejected")
    print(f"API checks passed: ALS, fallback, validation; {details}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--favorite-id", type=int, action="append")
    parser.add_argument(
        "--expected-run-id", help="Require this MLflow run ID in /model/info"
    )
    parser.add_argument(
        "--expected-revision", help="Require this code revision in /version"
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    check(
        args.url,
        args.favorite_id or [100],
        args.timeout,
        args.expected_run_id,
        args.expected_revision,
    )


if __name__ == "__main__":
    main()

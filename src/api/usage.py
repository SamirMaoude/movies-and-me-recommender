"""Aggregated usage records, which CloudWatch turns into metrics on Lambda."""

import json
import time

NAMESPACE = "MoviesAndMe"
# Requests to staging come from deployment checks: they stay in the logs but do
# not feed the metrics, which also keeps the number of custom metrics small.
METRICS_ALIAS = "live"
METRICS = (
    "Favorites",
    "UnknownFavorites",
    "PopularityFallback",
    "EmptyRecommendations",
)


def lambda_alias(context_header: str | None) -> str | None:
    """Return the alias of the invoked ARN, from the web adapter's context header."""
    try:
        arn = json.loads(context_header)["invoked_function_arn"]
    except (TypeError, ValueError, KeyError):
        return None
    parts = arn.split(":")
    return parts[7] if len(parts) == 8 else None


def usage_record(
    *,
    favorites: int,
    known: int,
    method: str,
    returned: int,
    alias: str | None,
    revision: str | None,
    model_run_id: str | None,
    lambda_version: str | None,
    timestamp: float | None = None,
) -> dict:
    """Describe one recommendation request with counts only, never the IDs."""
    record = {
        "event": "recommendation",
        "Alias": alias,
        "lambda_version": lambda_version,
        "revision": revision,
        "model_run_id": model_run_id,
        "method": method,
        "recommendations": returned,
        "Favorites": favorites,
        "UnknownFavorites": favorites - known,
        "PopularityFallback": int(method == "popularity"),
        "EmptyRecommendations": int(returned == 0),
    }
    if alias == METRICS_ALIAS:
        # Embedded metric format: CloudWatch Logs extracts these metrics itself.
        record["_aws"] = {
            "Timestamp": int(1000 * (time.time() if timestamp is None else timestamp)),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [["Alias"]],
                    "Metrics": [{"Name": name, "Unit": "Count"} for name in METRICS],
                }
            ],
        }
    return record


def log_usage(record: dict) -> None:
    print(json.dumps(record, separators=(",", ":")), flush=True)

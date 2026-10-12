"""Compare the inputs received by the live API with a reference period.

The figures come from the CloudWatch metrics written by the API: requests,
favorites per request, unknown favorites, popularity fallbacks and empty
answers. Exit code 1 means that the inputs changed beyond the thresholds.
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

NAMESPACE = "MoviesAndMe"
PERIOD = 300
# Absolute changes in shares, and a relative change in the mean profile size.
SHARE_THRESHOLDS = {"unknown_share": 0.10, "fallback_rate": 0.10, "empty_rate": 0.05}
MEAN_FAVORITES_CHANGE = 0.5
QUERIES = {
    "requests": ("Favorites", "SampleCount"),
    "favorites": ("Favorites", "Sum"),
    "unknown": ("UnknownFavorites", "Sum"),
    "fallbacks": ("PopularityFallback", "Sum"),
    "empty": ("EmptyRecommendations", "Sum"),
}


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime
    requests: int
    favorites: float
    unknown: float
    fallbacks: float
    empty: float

    def figures(self) -> dict[str, float | None]:
        if not self.requests:
            return dict.fromkeys(("mean_favorites", *SHARE_THRESHOLDS), None)
        return {
            "mean_favorites": self.favorites / self.requests,
            "unknown_share": self.unknown / self.favorites if self.favorites else 0.0,
            "fallback_rate": self.fallbacks / self.requests,
            "empty_rate": self.empty / self.requests,
        }


def changes(reference: Window, current: Window, min_requests: int) -> list[str]:
    """Describe each figure that moved beyond its threshold."""
    if min(reference.requests, current.requests) < min_requests:
        return []
    before, after = reference.figures(), current.figures()
    found = [
        f"{name}: {before[name]:.3f} -> {after[name]:.3f}"
        for name, threshold in SHARE_THRESHOLDS.items()
        if abs(after[name] - before[name]) > threshold
    ]
    if before["mean_favorites"] == 0:
        size_changed = after["mean_favorites"] > 0
    else:
        ratio = after["mean_favorites"] / before["mean_favorites"]
        size_changed = abs(ratio - 1) > MEAN_FAVORITES_CHANGE
    if size_changed:
        found.append(
            f"mean_favorites: {before['mean_favorites']:.2f} -> "
            f"{after['mean_favorites']:.2f}"
        )
    return found


def windows(
    end: datetime, current_hours: int, reference_hours: int
) -> tuple[datetime, datetime, datetime]:
    """Return the reference start, the current start and the end, in UTC."""
    end = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
    # Whole five-minute periods: each metric value falls in exactly one window.
    end = end.replace(minute=end.minute - end.minute % 5, second=0, microsecond=0)
    middle = end - timedelta(hours=current_hours)
    return middle - timedelta(hours=reference_hours), middle, end


def fetch(client, start: datetime, end: datetime) -> Window:
    """Add up the five-minute values of each metric over one window."""
    queries = [
        {
            "Id": key,
            "MetricStat": {
                "Metric": {
                    "Namespace": NAMESPACE,
                    "MetricName": metric,
                    "Dimensions": [{"Name": "Alias", "Value": "live"}],
                },
                "Period": PERIOD,
                "Stat": stat,
            },
        }
        for key, (metric, stat) in QUERIES.items()
    ]
    totals = dict.fromkeys(QUERIES, 0.0)
    token = None
    while True:
        extra = {"NextToken": token} if token else {}
        page = client.get_metric_data(
            MetricDataQueries=queries, StartTime=start, EndTime=end, **extra
        )
        for result in page["MetricDataResults"]:
            totals[result["Id"]] += sum(result["Values"])
        token = page.get("NextToken")
        if not token:
            break
    return Window(start, end, int(totals.pop("requests")), **totals)


def report(
    reference: Window, current: Window, found: list[str], min_requests: int
) -> str:
    before, after = reference.figures(), current.figures()
    lines = [
        "| | Reference | Current |",
        "|---|---|---|",
        f"| Window (UTC) | {reference.start:%Y-%m-%d %H:%M} to "
        f"{reference.end:%Y-%m-%d %H:%M} | {current.start:%Y-%m-%d %H:%M} to "
        f"{current.end:%Y-%m-%d %H:%M} |",
        f"| Requests | {reference.requests} | {current.requests} |",
    ]
    for name in before:
        values = [
            "-" if figures[name] is None else f"{figures[name]:.3f}"
            for figures in (before, after)
        ]
        lines.append(f"| {name} | {values[0]} | {values[1]} |")
    lines.append("")
    if min(reference.requests, current.requests) < min_requests:
        lines.append(f"Not compared: each window needs {min_requests} requests.")
    elif found:
        lines.append("Inputs changed: " + "; ".join(found) + ".")
    else:
        lines.append("No change beyond the thresholds.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-hours", type=int, default=24)
    parser.add_argument("--reference-hours", type=int, default=7 * 24)
    parser.add_argument(
        "--end",
        type=datetime.fromisoformat,
        help="End of the current window, in UTC (default: now)",
    )
    parser.add_argument("--min-requests", type=int, default=30)
    parser.add_argument("--region", default="eu-north-1")
    args = parser.parse_args()
    if min(args.current_hours, args.reference_hours, args.min_requests) < 1:
        parser.error("hours and --min-requests must be positive")

    start, middle, end = windows(
        args.end or datetime.now(UTC), args.current_hours, args.reference_hours
    )

    import botocore.session

    client = botocore.session.get_session().create_client(
        "cloudwatch", region_name=args.region
    )
    reference = fetch(client, start, middle)
    current = fetch(client, middle, end)
    found = changes(reference, current, args.min_requests)
    print(report(reference, current, found, args.min_requests))
    raise SystemExit(1 if found else 0)


if __name__ == "__main__":
    main()

"""Input comparison from CloudWatch values, without AWS access."""

import unittest
from datetime import UTC, datetime, timedelta, timezone

from scripts.compare_inputs import Window, changes, fetch, report, windows

START = datetime(2026, 10, 1, tzinfo=UTC)


def window(requests, favorites, unknown, fallbacks=0, empty=0):
    return Window(
        START,
        START + timedelta(hours=1),
        requests,
        favorites,
        unknown,
        fallbacks,
        empty,
    )


class FakeCloudWatch:
    """Return two pages, as get_metric_data does for long windows."""

    def __init__(self):
        self.calls = []

    def get_metric_data(self, **request):
        self.calls.append(request)
        page = len(self.calls)
        values = {
            "requests": [10, 30] if page == 1 else [20],
            "favorites": [50, 100] if page == 1 else [50],
            "unknown": [5] if page == 1 else [15],
            "fallbacks": [] if page == 1 else [3],
            "empty": [] if page == 1 else [1],
        }
        response = {
            "MetricDataResults": [
                {"Id": key, "Values": numbers} for key, numbers in values.items()
            ]
        }
        if page == 1:
            response["NextToken"] = "next"
        return response


class CompareInputsTests(unittest.TestCase):
    def test_fetch_adds_every_page_of_live_metrics(self):
        client = FakeCloudWatch()
        result = fetch(client, START, START + timedelta(hours=1))
        self.assertEqual(result, window(60, 200, 20, 3, 1))
        self.assertEqual(client.calls[1]["NextToken"], "next")
        query = client.calls[0]["MetricDataQueries"][0]["MetricStat"]
        self.assertEqual(
            query["Metric"]["Dimensions"], [{"Name": "Alias", "Value": "live"}]
        )
        self.assertEqual(query["Period"], 300)

    def test_windows_end_on_a_whole_five_minutes(self):
        end = datetime(2026, 10, 12, 9, 38, 41, 5, tzinfo=timezone(timedelta(hours=2)))
        start, middle, aligned = windows(end, 24, 168)
        self.assertEqual(aligned, datetime(2026, 10, 12, 7, 35, tzinfo=UTC))
        self.assertEqual(middle, aligned - timedelta(hours=24))
        self.assertEqual(start, middle - timedelta(hours=168))
        self.assertEqual(windows(datetime(2026, 10, 12, 7, 35), 1, 1)[2], aligned)

    def test_more_unknown_favorites_is_reported(self):
        found = changes(window(100, 400, 40), window(100, 400, 200), min_requests=30)
        self.assertEqual(found, ["unknown_share: 0.100 -> 0.500"])

    def test_small_variations_are_not_reported(self):
        reference = window(100, 400, 40, fallbacks=10, empty=1)
        current = window(80, 400, 60, fallbacks=15, empty=3)
        self.assertEqual(changes(reference, current, min_requests=30), [])

    def test_profile_size_fallbacks_and_empty_answers_are_compared(self):
        found = changes(
            window(100, 400, 0, fallbacks=5, empty=0),
            window(100, 100, 0, fallbacks=40, empty=10),
            min_requests=30,
        )
        self.assertEqual(
            [line.split(":")[0] for line in found],
            ["fallback_rate", "empty_rate", "mean_favorites"],
        )
        self.assertEqual(
            changes(window(50, 0, 0), window(50, 100, 0), 30)[-1],
            "mean_favorites: 0.00 -> 2.00",
        )

    def test_windows_with_few_requests_are_not_compared(self):
        reference, current = window(29, 100, 0), window(100, 400, 400)
        self.assertEqual(changes(reference, current, min_requests=30), [])
        text = report(reference, current, [], 30)
        self.assertIn("Not compared: each window needs 30 requests.", text)

    def test_report_lists_both_windows(self):
        reference, current = window(100, 400, 40), window(100, 400, 200)
        text = report(reference, current, changes(reference, current, 30), 30)
        self.assertIn("| unknown_share | 0.100 | 0.500 |", text)
        self.assertIn("| Requests | 100 | 100 |", text)
        self.assertTrue(text.endswith("Inputs changed: unknown_share: 0.100 -> 0.500."))
        empty = report(window(0, 0, 0), reference, [], 30)
        self.assertIn("| mean_favorites | - | 4.000 |", empty)


if __name__ == "__main__":
    unittest.main()

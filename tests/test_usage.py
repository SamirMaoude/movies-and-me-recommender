"""Usage records: counts only, and metrics for the live alias alone."""

import json
import unittest

from src.api.usage import METRICS, NAMESPACE, lambda_alias, usage_record

ARN = "arn:aws:lambda:eu-north-1:123456789012:function:movies-and-me-recommender"


def context(arn):
    return json.dumps({"request_id": "r", "deadline": 1, "invoked_function_arn": arn})


def record(alias="live", **overrides):
    values = {
        "favorites": 4,
        "known": 3,
        "method": "als",
        "returned": 10,
        "alias": alias,
        "revision": "c" * 40,
        "model_run_id": "a" * 32,
        "lambda_version": "5",
        "timestamp": 1_700_000_000.5,
    }
    return usage_record(**{**values, **overrides})


class LambdaAliasTests(unittest.TestCase):
    def test_alias_comes_from_the_invoked_arn(self):
        self.assertEqual(lambda_alias(context(f"{ARN}:live")), "live")
        self.assertEqual(lambda_alias(context(f"{ARN}:staging")), "staging")
        self.assertEqual(lambda_alias(context(f"{ARN}:5")), "5")

    def test_missing_or_unexpected_context_has_no_alias(self):
        for header in (None, "", "not json", "[]", "{}", context(ARN)):
            with self.subTest(header=header):
                self.assertIsNone(lambda_alias(header))


class UsageRecordTests(unittest.TestCase):
    def test_live_record_declares_the_metrics(self):
        value = record()
        self.assertEqual(
            value["_aws"],
            {
                "Timestamp": 1_700_000_000_500,
                "CloudWatchMetrics": [
                    {
                        "Namespace": NAMESPACE,
                        "Dimensions": [["Alias"]],
                        "Metrics": [
                            {"Name": name, "Unit": "Count"} for name in METRICS
                        ],
                    }
                ],
            },
        )
        # Every declared metric and dimension needs a value at the top level.
        for name in (*METRICS, "Alias"):
            self.assertIn(name, value)
        self.assertEqual(value["Favorites"], 4)
        self.assertEqual(value["UnknownFavorites"], 1)
        self.assertEqual(value["PopularityFallback"], 0)
        self.assertEqual(value["EmptyRecommendations"], 0)
        self.assertEqual(value["model_run_id"], "a" * 32)
        self.assertEqual(value["revision"], "c" * 40)

    def test_staging_and_unknown_aliases_only_log(self):
        for alias in ("staging", "5", None):
            with self.subTest(alias=alias):
                value = record(alias=alias)
                self.assertNotIn("_aws", value)
                self.assertEqual(value["Alias"], alias)

    def test_fallback_and_empty_answers_are_flagged(self):
        value = record(favorites=2, known=0, method="popularity", returned=0)
        self.assertEqual(value["UnknownFavorites"], 2)
        self.assertEqual(value["PopularityFallback"], 1)
        self.assertEqual(value["EmptyRecommendations"], 1)

    def test_record_holds_no_list_of_favorites(self):
        value = record()
        value.pop("_aws")
        self.assertTrue(
            all(isinstance(item, str | int | None) for item in value.values())
        )
        json.dumps(value, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

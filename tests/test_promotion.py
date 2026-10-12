"""Promotion of a retrained bundle against the model in service."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.evaluation.prepare import load_splits
from src.models.collaborative import ALSConfig, ImplicitRecommender
from src.training.promotion import compare, decide, main, summary

EVALUATION = {
    "positive_threshold": 4.0,
    "train_end": "1970-01-01T00:01:40+00:00",
    "validation_end": "1970-01-01T00:03:20+00:00",
}
RULES = {
    "k": 10,
    "ndcg_tolerance": 0.025,
    "recall_tolerance": 0.009,
    "min_coverage_ratio": 0.8,
    "max_p95_ms": 60.0,
    "min_validation_users": 1,
}
FILES = {
    "movies.csv": (
        "movieId,title,genres\n10,First,Drama\n20,Second,Drama\n"
        "30,Third,Drama\n40,Future,Action\n"
    ),
    "links.csv": (
        "movieId,imdbId,tmdbId\n10,0000010,100\n20,0000020,200\n"
        "30,0000030,300\n40,0000040,400\n"
    ),
    "ratings.csv": (
        "userId,movieId,rating,timestamp\n1,10,4.0,10\n2,20,5.0,20\n"
        "3,30,4.0,30\n3,10,4.5,40\n1,20,5.0,100\n2,30,4.0,110\n4,40,5.0,200\n"
    ),
    "tags.csv": "userId,movieId,tag,timestamp\n",
}


def metrics(ndcg, recall, coverage, p95, users=20):
    return {
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
        "catalog_coverage": coverage,
        "p95_ms": p95,
        "users_evaluated": users,
    }


class DecisionTests(unittest.TestCase):
    served = metrics(0.062, 0.048, 0.0306, 6.0)

    def test_candidate_within_the_seed_variation_is_accepted(self):
        candidate = metrics(0.044, 0.042, 0.0295, 6.2)
        checks = decide(candidate, self.served, {**RULES, "min_validation_users": 20})
        self.assertTrue(all(check["passed"] for check in checks))

    def test_degraded_candidate_fails_on_quality_and_coverage(self):
        candidate = metrics(0.026, 0.014, 0.0061, 6.1)
        failed = [
            check["check"]
            for check in decide(candidate, self.served, RULES)
            if not check["passed"]
        ]
        self.assertEqual(failed, ["ndcg_at_10", "recall_at_10", "catalog coverage"])

    def test_slower_ranking_of_a_larger_catalog_stays_within_the_budget(self):
        # The MovieLens 32M candidate against the model of the small dataset.
        served = metrics(0.0518, 0.0274, 0.0156, 12.3)
        candidate = metrics(0.0728, 0.0415, 0.0474, 28.6)
        checks = decide(candidate, served, RULES)
        self.assertTrue(all(check["passed"] for check in checks))

    def test_slow_candidate_and_small_cohort_are_refused(self):
        candidate = metrics(0.062, 0.048, 0.0306, 75.0, users=19)
        failed = [
            check["check"]
            for check in decide(
                candidate, self.served, {**RULES, "min_validation_users": 20}
            )
            if not check["passed"]
        ]
        self.assertEqual(failed, ["validation users", "p95 latency (ms)"])


class CompareTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.data_dir = self.directory / "raw"
        self.data_dir.mkdir()
        for name, text in FILES.items():
            (self.data_dir / name).write_text(text, encoding="utf-8")
        tables, _, splits = load_splits(self.data_dir, EVALUATION)
        self.train, self.links = splits["train"], tables["links"]

    def bundle(self, name, factors=2, iterations=3, config=EVALUATION):
        model = ImplicitRecommender.fit(
            self.train,
            self.links,
            ALSConfig(factors=factors, iterations=iterations, random_state=7),
        )
        path = self.directory / name
        model.save(path, metadata={"run_id": name * 4, "config": config})
        return path

    def test_same_model_is_accepted_on_the_candidate_validation(self):
        served = self.bundle("a")
        decision = compare(served, self.bundle("b"), self.data_dir, RULES)
        self.assertEqual(decision["served_run_id"], "aaaa")
        self.assertEqual(decision["candidate_run_id"], "bbbb")
        self.assertEqual(decision["results"]["served"]["users_evaluated"], 2)
        self.assertEqual(
            decision["results"]["candidate"]["ndcg_at_10"],
            decision["results"]["served"]["ndcg_at_10"],
        )
        self.assertIn("popularity", decision["results"])
        # Latency on two tiny profiles is too noisy to assert here.
        quality = [
            check
            for check in decision["checks"]
            if check["check"] != "p95 latency (ms)"
        ]
        self.assertTrue(all(check["passed"] for check in quality))
        json.dumps(decision, allow_nan=False)

    def test_served_model_that_saw_the_validation_period_cannot_be_compared(self):
        later = {**EVALUATION, "train_end": "1970-01-01T00:02:00+00:00"}
        served = self.bundle("a", config=later)
        with self.assertRaisesRegex(ValueError, "validation period"):
            compare(served, self.bundle("b"), self.data_dir, RULES)

    def test_command_writes_the_decision_and_exits_with_its_verdict(self):
        params = self.directory / "params.yaml"
        params.write_text(
            json.dumps({"promotion": {**RULES, "min_validation_users": 3}}),
            encoding="utf-8",
        )
        output = self.directory / "promotion.json"
        argv = [
            "promotion",
            "--served",
            str(self.bundle("a")),
            "--candidate",
            str(self.bundle("b")),
            "--data-dir",
            str(self.data_dir),
            "--params",
            str(params),
            "--output",
            str(output),
        ]
        printed = io.StringIO()
        with patch("sys.argv", argv), redirect_stdout(printed):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)
        decision = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(decision["accepted"])
        self.assertIn("is **refused** against `aaaa`", printed.getvalue())
        self.assertIn(
            "| validation users | 2 | 2 | >= 3 | **fail** |", printed.getvalue()
        )

    def test_summary_shows_the_popularity_baseline(self):
        decision = compare(self.bundle("a"), self.bundle("b"), self.data_dir, RULES)
        self.assertIn("Popularity baseline: NDCG@10", summary(decision))


if __name__ == "__main__":
    unittest.main()

"""Decide whether a retrained ALS bundle may replace the model in service.

Both bundles and the popularity baseline are scored on the validation split of
the candidate, so the comparison uses the same users. The final test split is
never read.
"""

import argparse
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

import yaml

from src.evaluation.compare import build_validation_cases, evaluate_recommender
from src.evaluation.prepare import load_splits
from src.files import write_json
from src.models.collaborative import ImplicitRecommender
from src.models.popularity import PopularityRecommender

# Timing a fixed sample keeps the comparison short on a large validation set.
LATENCY_SAMPLE = 200


def measure(recommender, cases, candidate_ids, k: int, repeats: int = 5) -> dict:
    """Validation metrics, top-k catalog coverage and p95 latency of one model."""
    evaluation = evaluate_recommender(recommender, cases, candidate_ids, k)
    recommended = {
        tmdb for row in evaluation["per_user"] for tmdb in row["recommended_tmdb_ids"]
    }
    timed = random.Random(0).sample(cases, min(len(cases), LATENCY_SAMPLE))
    timings = []
    for _ in range(repeats):
        for case in timed:
            start = perf_counter()
            recommender.recommend(case.favorites, n=k)
            timings.append(perf_counter() - start)
    timings.sort()
    return {
        **evaluation["metrics"],
        "catalog_coverage": len(recommended) / len(candidate_ids),
        "p95_ms": 1000 * timings[round(0.95 * (len(timings) - 1))],
        "users_evaluated": evaluation["users_evaluated"],
    }


def decide(candidate: dict, served: dict, rules: dict) -> list[dict]:
    """Apply each rule; the candidate is accepted only if every check passes."""
    k = rules["k"]
    ndcg, recall = f"ndcg_at_{k}", f"recall_at_{k}"
    limits = [
        ("validation users", "users_evaluated", ">=", rules["min_validation_users"]),
        (ndcg, ndcg, ">=", served[ndcg] - rules["ndcg_tolerance"]),
        (recall, recall, ">=", served[recall] - rules["recall_tolerance"]),
        (
            "catalog coverage",
            "catalog_coverage",
            ">=",
            served["catalog_coverage"] * rules["min_coverage_ratio"],
        ),
        # An absolute budget: a larger catalog is slower to rank by nature, and a
        # limit relative to the served model would block it.
        ("p95 latency (ms)", "p95_ms", "<=", rules["max_p95_ms"]),
    ]
    return [
        {
            "check": name,
            "candidate": candidate[key],
            "served": served[key],
            "limit": f"{operator} {limit:.4g}",
            "passed": candidate[key] >= limit
            if operator == ">="
            else candidate[key] <= limit,
        }
        for name, key, operator, limit in limits
    ]


def _train_end(model: ImplicitRecommender) -> datetime:
    return datetime.fromisoformat(model.metadata["config"]["train_end"])


def compare(served_dir: Path, candidate_dir: Path, data_dir: Path, rules: dict) -> dict:
    served = ImplicitRecommender.load(served_dir)
    candidate = ImplicitRecommender.load(candidate_dir)
    # A model fitted after the candidate's train period has seen its validation.
    if _train_end(served) > _train_end(candidate):
        raise ValueError("The served model was trained on the validation period.")
    config = candidate.metadata["config"]
    _, _, splits = load_splits(data_dir, config)
    train = splits["train"]
    candidate_ids, cases = build_validation_cases(train, splits["validation"])
    if not cases:
        raise ValueError("No validation user to compare the models.")
    k = rules["k"]
    results = {
        "served": measure(served, cases, candidate_ids, k),
        "candidate": measure(candidate, cases, candidate_ids, k),
        "popularity": measure(PopularityRecommender(train), cases, candidate_ids, k),
    }
    checks = decide(results["candidate"], results["served"], rules)
    return {
        "accepted": all(check["passed"] for check in checks),
        "served_run_id": served.metadata.get("run_id"),
        "candidate_run_id": candidate.metadata.get("run_id"),
        "validation": {
            "train_end": config["train_end"],
            "validation_end": config["validation_end"],
        },
        "results": results,
        "checks": checks,
        "rules": rules,
    }


def summary(decision: dict) -> str:
    verdict = "accepted" if decision["accepted"] else "refused"
    lines = [
        f"Candidate `{decision['candidate_run_id']}` is **{verdict}** against "
        f"`{decision['served_run_id']}`, on the validation users from "
        f"{decision['validation']['train_end']} to "
        f"{decision['validation']['validation_end']}.",
        "",
        "| Check | Candidate | In service | Limit | Result |",
        "|---|---|---|---|---|",
    ]
    for check in decision["checks"]:
        result = "pass" if check["passed"] else "**fail**"
        lines.append(
            f"| {check['check']} | {check['candidate']:.4g} | {check['served']:.4g}"
            f" | {check['limit']} | {result} |"
        )
    popularity = decision["results"]["popularity"]
    k = decision["rules"]["k"]
    lines += [
        "",
        f"Popularity baseline: NDCG@{k} {popularity[f'ndcg_at_{k}']:.4g}, "
        f"Recall@{k} {popularity[f'recall_at_{k}']:.4g}, "
        f"coverage {popularity['catalog_coverage']:.4g}.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--served", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path("artifacts/dvc/model"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rules = yaml.safe_load(args.params.read_text(encoding="utf-8"))["promotion"]
    try:
        decision = compare(args.served, args.candidate, args.data_dir, rules)
    except (OSError, ValueError) as exception:
        parser.exit(1, f"{exception}\n")
    write_json(args.output, decision)
    print(summary(decision))
    raise SystemExit(0 if decision["accepted"] else 1)


if __name__ == "__main__":
    main()

"""Create a small ALS bundle for container checks, using synthetic data."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make src importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.collaborative import ALSConfig, ImplicitRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    links = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40, 50, 60],
            "tmdbId": [100, 200, 300, 400, 500, 600],
        }
    )
    pairs = [
        (1, 10),
        (1, 20),
        (1, 30),
        (2, 10),
        (2, 30),
        (2, 40),
        (3, 20),
        (3, 50),
        (3, 60),
        (4, 40),
        (4, 50),
    ]
    train = pd.DataFrame(
        [(user, movie, movie * 10, True) for user, movie in pairs],
        columns=["userId", "movieId", "tmdbId", "is_positive"],
    )
    model = ImplicitRecommender.fit(
        train,
        links,
        ALSConfig(factors=3, iterations=3, random_state=42),
    )
    model.save(args.output, metadata={"purpose": "container-smoke-test"})
    print(f"Synthetic ALS bundle: {args.output}")


if __name__ == "__main__":
    main()

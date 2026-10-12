"""Write the serving copy of an ALS bundle, without the training users and matrix."""

import argparse
from pathlib import Path

from src.models.collaborative import ImplicitRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Complete bundle from training")
    parser.add_argument("target", type=Path, help="New directory for the serving copy")
    args = parser.parse_args()
    try:
        model = ImplicitRecommender.load(args.source)
        model.serving_copy().save(args.target)
    except (OSError, ValueError) as exception:
        parser.exit(1, f"{exception}\n")


if __name__ == "__main__":
    main()

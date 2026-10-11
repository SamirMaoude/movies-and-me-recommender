"""Recall and binary NDCG for ranked movie IDs."""

from collections.abc import Iterable, Mapping, Set
from math import log2
from numbers import Integral

import numpy as np


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must contain positive integers.")
    return int(value)


def _item_ids(values: Iterable[int], name: str) -> list[int]:
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an iterable of positive integer IDs.")
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of positive integer IDs.") from exc
    return [_positive_integer(value, name) for value in items]


def _validated_inputs(
    recommendations: Iterable[int], relevant_items: Iterable[int], k: int,
) -> tuple[list[int], set[int], int]:
    limit = _positive_integer(k, "k")
    if isinstance(recommendations, Set):
        raise ValueError("recommendations must preserve ranking order.")
    ranking = _item_ids(recommendations, "recommendations")
    if len(set(ranking)) != len(ranking):
        raise ValueError("recommendations must not contain duplicate IDs.")
    relevant = set(_item_ids(relevant_items, "relevant_items"))
    return ranking[:limit], relevant, limit


def recall_at_k(
    recommendations: Iterable[int], relevant_items: Iterable[int], k: int = 10,
) -> float | None:
    """Return the fraction of distinct relevant movies found in the first k.

    Ranking order is preserved, and duplicate recommendations are rejected,
    including after position k. Repeated relevant IDs count once. No relevant
    movies returns None; a missing ranking with relevant movies returns 0.0.
    Inputs are not modified. IDs and k must be positive integers.
    """
    ranking, relevant, _ = _validated_inputs(recommendations, relevant_items, k)
    if not relevant:
        return None
    return float(sum(item in relevant for item in ranking) / len(relevant))


def ndcg_at_k(
    recommendations: Iterable[int], relevant_items: Iterable[int], k: int = 10,
) -> float | None:
    """Return binary discounted gain divided by the ideal gain at k.

    A relevant movie at rank r contributes 1 / log2(r + 1), with ranks starting
    at one. The ideal ranking contains min(k, number of distinct relevant
    movies) hits. No relevant movies returns None; no hits returns 0.0.
    The ranking must contain unique IDs. Inputs are not modified.
    """
    ranking, relevant, limit = _validated_inputs(recommendations, relevant_items, k)
    if not relevant:
        return None
    gain = sum(
        1.0 / log2(rank + 1)
        for rank, item in enumerate(ranking, start=1)
        if item in relevant
    )
    ideal_gain = sum(1.0 / log2(rank + 1) for rank in range(1, min(limit, len(relevant)) + 1))
    return float(gain / ideal_gain)

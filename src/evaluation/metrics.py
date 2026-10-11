"""Recall and binary NDCG for ranked movie IDs."""

from collections.abc import Iterable
from math import log2


def recall_at_k(
    recommendations: Iterable[int], relevant_items: Iterable[int], k: int = 10
) -> float | None:
    """Fraction of the relevant items found in the first k, or None without any."""
    relevant = set(relevant_items)
    if not relevant:
        return None
    hits = sum(item in relevant for item in list(recommendations)[:k])
    return hits / len(relevant)


def ndcg_at_k(
    recommendations: Iterable[int], relevant_items: Iterable[int], k: int = 10
) -> float | None:
    """Binary DCG of the first k over the ideal DCG, or None without relevant items."""
    relevant = set(relevant_items)
    if not relevant:
        return None
    gain = sum(
        1 / log2(rank + 1)
        for rank, item in enumerate(list(recommendations)[:k], start=1)
        if item in relevant
    )
    ideal_gain = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return gain / ideal_gain

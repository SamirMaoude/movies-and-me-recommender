"""Hand-calculated ranking examples."""

import unittest
from math import log2

import numpy as np

from src.evaluation.metrics import ndcg_at_k, recall_at_k

METRICS = (recall_at_k, ndcg_at_k)


class RankingMetricTests(unittest.TestCase):
    def test_recall_uses_all_distinct_targets_as_the_denominator(self):
        # Hits at ranks 1 and 3 out of four future favorites.
        self.assertEqual(recall_at_k([10, 20, 30, 40], [10, 30, 50, 60], k=3), 0.5)

    def test_ndcg_discount_and_ideal_gain_match_a_hand_calculated_example(self):
        # Hits at ranks 1 and 3: 1 + 1/2. Three ideal hits: 1 + 1/log2(3) + 1/2.
        expected = 1.5 / (1.5 + 1.0 / log2(3))
        self.assertAlmostEqual(
            ndcg_at_k([10, 20, 30, 40], [10, 30, 50, 60], k=3), expected
        )

    def test_later_hits_keep_recall_but_reduce_ndcg(self):
        targets = [10, 20]
        first = [10, 20, 30, 40]
        last = [30, 40, 10, 20]
        self.assertEqual(recall_at_k(first, targets, k=4), 1.0)
        self.assertEqual(recall_at_k(last, targets, k=4), 1.0)
        self.assertEqual(ndcg_at_k(first, targets, k=4), 1.0)
        expected = (0.5 + 1.0 / log2(5)) / (1.0 + 1.0 / log2(3))
        self.assertAlmostEqual(ndcg_at_k(last, targets, k=4), expected)

    def test_only_the_first_k_positions_contribute(self):
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                self.assertEqual(metric([20, 30, 10], [10], k=2), 0.0)
                self.assertEqual(metric([10, 30, 20], [10], k=1), 1.0)

    def test_default_limit_is_ten(self):
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                self.assertEqual(metric(range(1, 12), [11]), 0.0)
                self.assertGreater(metric(range(1, 12), [10]), 0.0)

    def test_short_rankings_keep_the_full_ideal_denominator(self):
        self.assertEqual(recall_at_k([10], [10, 20, 30], k=10), 1.0 / 3.0)
        expected = 1.0 / (1.0 + 1.0 / log2(3) + 0.5)
        self.assertAlmostEqual(ndcg_at_k([10], [10, 20, 30], k=10), expected)

    def test_ideal_gain_is_limited_by_k_when_there_are_more_targets(self):
        self.assertEqual(recall_at_k([10, 20], [10, 20, 30, 40], k=2), 0.5)
        self.assertEqual(ndcg_at_k([10, 20], [10, 20, 30, 40], k=2), 1.0)

    def test_repeated_relevant_ids_count_once_and_sets_are_accepted(self):
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                expected = metric([10, 30, 20], {10, 20}, k=3)
                self.assertEqual(metric([10, 30, 20], [20, 10, 20, 10], k=3), expected)

    def test_no_relevant_movies_has_no_score(self):
        for metric in METRICS:
            for ranking in ([], [10, 20]):
                with self.subTest(metric=metric.__name__, ranking=ranking):
                    self.assertIsNone(metric(ranking, []))

    def test_empty_ranking_or_no_hits_scores_zero(self):
        for metric in METRICS:
            for ranking in ([], [30, 40]):
                with self.subTest(metric=metric.__name__, ranking=ranking):
                    result = metric(ranking, [10, 20])
                    self.assertEqual(result, 0.0)
                    self.assertIs(type(result), float)

    def test_generators_numpy_ids_and_numpy_limits_are_accepted(self):
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                result = metric(
                    (value for value in (np.int64(10), np.uint32(20))),
                    (value for value in (np.uint64(20), np.int32(10))),
                    k=np.int64(2),
                )
                self.assertEqual(result, 1.0)
                self.assertIs(type(result), float)

    def test_inputs_are_not_modified(self):
        ranking = [10, 30, 20]
        targets = [20, 10, 20]
        for metric in METRICS:
            metric(ranking, targets, k=2)
            self.assertEqual(ranking, [10, 30, 20])
            self.assertEqual(targets, [20, 10, 20])


if __name__ == "__main__":
    unittest.main()

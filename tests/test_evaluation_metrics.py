"""Hand-calculated ranking examples and invalid evaluation inputs."""

from math import log2
import unittest

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
        self.assertAlmostEqual(ndcg_at_k([10, 20, 30, 40], [10, 30, 50, 60], k=3), expected)

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


class RankingMetricValidationTests(unittest.TestCase):
    def test_duplicates_anywhere_in_the_ranking_are_rejected(self):
        for metric in METRICS:
            for ranking in ([10, 10], [10, 20, 20], [np.int64(10), 10]):
                for targets in ([], [10]):
                    with self.subTest(metric=metric.__name__, ranking=ranking, targets=targets):
                        with self.assertRaises(ValueError):
                            metric(ranking, targets, k=1)

    def test_invalid_ids_are_rejected_even_outside_the_cutoff(self):
        invalid_ids = (0, -1, True, np.bool_(False), 10.0, "10", None, np.nan, np.inf, [10])
        for metric in METRICS:
            for value in invalid_ids:
                with self.subTest(metric=metric.__name__, value=value):
                    with self.assertRaises(ValueError):
                        metric([10, value], [], k=1)
                    with self.assertRaises(ValueError):
                        metric([], [value], k=1)

    def test_invalid_limits_are_rejected_even_without_targets(self):
        invalid_limits = (0, -1, True, np.bool_(False), 1.0, "1", None, np.nan, np.inf)
        for metric in METRICS:
            for limit in invalid_limits:
                with self.subTest(metric=metric.__name__, limit=limit):
                    with self.assertRaises(ValueError):
                        metric([], [], k=limit)

    def test_inputs_must_be_iterables_of_ids(self):
        invalid_inputs = (None, 10, 10.0, True, "", "10", b"", b"10", {10: 1}, np.array(10))
        for metric in METRICS:
            for values in invalid_inputs:
                with self.subTest(metric=metric.__name__, values=values):
                    with self.assertRaises(ValueError):
                        metric(values, [10])
                    with self.assertRaises(ValueError):
                        metric([10], values)

    def test_rankings_must_preserve_order(self):
        for metric in METRICS:
            for ranking in (set(), {10, 20}, frozenset({10, 20})):
                with self.subTest(metric=metric.__name__, ranking=ranking):
                    with self.assertRaises(ValueError):
                        metric(ranking, [10])


if __name__ == "__main__":
    unittest.main()

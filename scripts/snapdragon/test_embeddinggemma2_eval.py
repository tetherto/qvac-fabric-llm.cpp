#!/usr/bin/env python3
"""Regression tests for the embedding evaluation workload."""

import unittest

from embeddinggemma2_eval import partial_batch_indices


class PartialBatchTests(unittest.TestCase):
    def test_multiple_batches_with_partial_tail(self):
        cases = [
            ([18, 31, 27], 256, [246, 31]),
            ([32, 40, 40], 256, [256, 40]),
            ([1, 1, 1], 256, [256, 1]),
            ([256, 1, 1], 256, [256, 2, 256, 2, 256, 1]),
            ([256, 256, 256], 512, [512, 512, 512, 256]),
            ([18, 31, 27, 19, 27, 33, 22, 1011], 1024, [177, 1011, 177, 1011, 18]),
        ]
        for lengths, expected_capacity, expected_batches in cases:
            with self.subTest(lengths=lengths):
                capacity, indices = partial_batch_indices(lengths)
                self.assertEqual(capacity, expected_capacity)
                original_indices = list(range(len(lengths))) * 2 + [0]
                self.assertEqual(indices[:len(original_indices)], original_indices)
                batches = [0]
                for index in indices:
                    if batches[-1] + lengths[index] > capacity:
                        batches.append(0)
                    batches[-1] += lengths[index]
                self.assertEqual(batches, expected_batches)


if __name__ == "__main__":
    unittest.main()

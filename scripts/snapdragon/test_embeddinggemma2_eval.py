#!/usr/bin/env python3
"""Regression tests for the embedding evaluation workload."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from embeddinggemma2_eval import host_device_info, partial_batch_indices


class HostDeviceInfoTests(unittest.TestCase):
    def test_metadata_without_os_uname(self):
        cases = [
            ("Windows", "11", "AMD64"),
            ("Linux", "6.6.87.2", "aarch64"),
            ("Darwin", "25.0.0", "arm64"),
        ]
        for system, release, machine in cases:
            with self.subTest(system=system):
                info = SimpleNamespace(system=system, release=release, machine=machine)
                with patch("embeddinggemma2_eval.platform.uname", return_value=info), \
                     patch("embeddinggemma2_eval.os.uname", None, create=True):
                    self.assertEqual(host_device_info(), {"host": system + " " + release, "machine": machine})


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

from __future__ import annotations

import unittest

import torch

from bug_squash.pytorch.data import load_customer_churn_tensors, make_train_test_split


class TensorDataTest(unittest.TestCase):
    def test_loader_reads_expected_tensor_shapes(self) -> None:
        dataset = load_customer_churn_tensors()

        self.assertEqual(
            dataset.feature_names,
            [
                "tenure_months",
                "support_tickets",
                "weekly_active_days",
                "monthly_bill",
            ],
        )
        self.assertEqual(tuple(dataset.features.shape), (20, 4))
        self.assertEqual(tuple(dataset.labels.shape), (20, 1))
        self.assertEqual(dataset.features.dtype, torch.float32)
        self.assertEqual(dataset.labels.dtype, torch.float32)

    def test_make_train_test_split_is_reproducible(self) -> None:
        dataset = load_customer_churn_tensors()

        split_one = make_train_test_split(dataset.features, dataset.labels, seed=7)
        split_two = make_train_test_split(dataset.features, dataset.labels, seed=7)

        self.assertTrue(torch.equal(split_one.train_features, split_two.train_features))
        self.assertTrue(torch.equal(split_one.test_labels, split_two.test_labels))


if __name__ == "__main__":
    unittest.main()

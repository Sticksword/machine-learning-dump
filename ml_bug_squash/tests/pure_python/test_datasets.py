from __future__ import annotations

import unittest

from bug_squash.pure_python.datasets import load_customer_churn


class LoadCustomerChurnTest(unittest.TestCase):
    def test_loader_reads_expected_shape(self) -> None:
        dataset = load_customer_churn()

        self.assertEqual(
            dataset.feature_names,
            [
                "tenure_months",
                "support_tickets",
                "weekly_active_days",
                "monthly_bill",
            ],
        )
        self.assertEqual(dataset.num_examples, 20)
        self.assertEqual(dataset.num_features, 4)
        self.assertEqual(dataset.labels.count(1), 10)
        self.assertEqual(dataset.labels.count(0), 10)


if __name__ == "__main__":
    unittest.main()

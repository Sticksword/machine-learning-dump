from __future__ import annotations

import unittest

from bug_squash.metrics import accuracy_score, binary_log_loss


class MetricsTest(unittest.TestCase):
    def test_accuracy_score(self) -> None:
        score = accuracy_score([1, 0, 1, 1], [1, 0, 0, 1])
        self.assertAlmostEqual(score, 0.75)

    def test_binary_log_loss_is_low_for_good_predictions(self) -> None:
        loss = binary_log_loss([1, 0, 1], [0.95, 0.05, 0.90])
        self.assertLess(loss, 0.1)


if __name__ == "__main__":
    unittest.main()

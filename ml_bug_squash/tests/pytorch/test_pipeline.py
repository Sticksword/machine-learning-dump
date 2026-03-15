from __future__ import annotations

import unittest

from bug_squash.pytorch.pipeline import TorchChurnPipeline


class TorchChurnPipelineTest(unittest.TestCase):
    def test_pipeline_reaches_expected_quality_bar(self) -> None:
        summary = TorchChurnPipeline().run()

        self.assertEqual(summary.train_size, 14)
        self.assertEqual(summary.test_size, 6)
        self.assertGreaterEqual(summary.test_accuracy, 0.80)
        self.assertLess(summary.test_loss, 0.55)


if __name__ == "__main__":
    unittest.main()

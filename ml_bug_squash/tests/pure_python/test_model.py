from __future__ import annotations

import unittest

from bug_squash.pure_python.metrics import accuracy_score
from bug_squash.pure_python.model import LogisticRegressionGD


class LogisticRegressionTest(unittest.TestCase):
    def test_model_learns_an_easy_separable_dataset(self) -> None:
        rows = [[-2.0], [-1.5], [-1.0], [-0.5], [0.5], [1.0], [1.5], [2.0]]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]

        model = LogisticRegressionGD(learning_rate=0.3, epochs=400)
        model.fit(rows, labels)
        predictions = model.predict(rows)

        self.assertGreaterEqual(accuracy_score(labels, predictions), 0.875)


if __name__ == "__main__":
    unittest.main()

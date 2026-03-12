from __future__ import annotations

import unittest

from bug_squash.preprocessing import StandardScaler


class StandardScalerTest(unittest.TestCase):
    def test_fit_transform_centers_and_scales_training_data(self) -> None:
        rows = [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]

        transformed = StandardScaler().fit_transform(rows)

        for feature_index in range(2):
            column = [row[feature_index] for row in transformed]
            mean = sum(column) / len(column)
            variance = sum((value - mean) ** 2 for value in column) / len(column)

            self.assertAlmostEqual(mean, 0.0, places=7)
            self.assertAlmostEqual(variance, 1.0, places=7)


if __name__ == "__main__":
    unittest.main()

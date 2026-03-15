from __future__ import annotations

import unittest

from bug_squash.pure_python.splitting import train_test_split


class TrainTestSplitTest(unittest.TestCase):
    def test_split_keeps_rows_and_labels_aligned(self) -> None:
        rows = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]
        labels = [0, 1, 2, 3, 4, 5]

        split = train_test_split(rows, labels, test_ratio=0.33, seed=11)

        reconstructed = list(
            zip(
                split.train_rows + split.test_rows,
                split.train_labels + split.test_labels,
            )
        )

        self.assertEqual(
            {(tuple(row), label) for row, label in reconstructed},
            {(tuple(row), label) for row, label in zip(rows, labels)},
        )


if __name__ == "__main__":
    unittest.main()

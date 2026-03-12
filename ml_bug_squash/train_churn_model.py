from __future__ import annotations

from bug_squash.pipeline import ChurnTrainingPipeline


def main() -> None:
    summary = ChurnTrainingPipeline().run()
    print(f"train_size={summary.train_size}")
    print(f"test_size={summary.test_size}")
    print(f"test_accuracy={summary.test_accuracy:.3f}")
    print(f"test_log_loss={summary.test_log_loss:.3f}")


if __name__ == "__main__":
    main()

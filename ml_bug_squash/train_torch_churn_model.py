from __future__ import annotations

from bug_squash.pytorch.pipeline import TorchChurnPipeline


def main() -> None:
    summary = TorchChurnPipeline().run()
    print(f"train_size={summary.train_size}")
    print(f"test_size={summary.test_size}")
    print(f"test_accuracy={summary.test_accuracy:.3f}")
    print(f"test_loss={summary.test_loss:.3f}")


if __name__ == "__main__":
    main()

# ML Bug Squash Practice

This directory contains one interview-style ML debugging repo with two exercises that share a dataset and a package namespace:

- `bug_squash.pure_python`: a small logistic-regression pipeline written with the standard library
- `bug_squash.pytorch`: a tabular PyTorch training loop with a small neural network

Both exercises are intentionally tidy and readable. The tests are correct and should initially fail because of bugs in the library code.

## Shared scenario

Both versions train a lightweight churn-prediction model from `data/customer_churn.csv`:

- the dataset is loaded from CSV
- the data is split into train and test sets
- features are normalized
- a classifier is trained
- evaluation reports classification quality on held-out data

Your goal is to use the tests and the codebase to track down the bugs and get the suite passing.

## Running the exercises

Pure Python tests:

```bash
python3 -m unittest discover -s tests/pure_python -v
```

PyTorch tests:

```bash
../.venv/bin/python -m unittest discover -s tests/pytorch -v
```

All tests with the interpreter that has `torch` installed:

```bash
../.venv/bin/python -m unittest discover -s tests -v
```

End-to-end scripts:

```bash
python3 train_churn_model.py
../.venv/bin/python train_torch_churn_model.py
```

## What to practice

- Forming a debugging hypothesis before making changes
- Narrowing failures from integration tests to specific modules
- Using passing tests to infer intended behavior
- Fixing the implementation quickly without over-engineering

## Suggested approach

1. Run the relevant test suite and group failures by subsystem.
2. Re-run a single failing test while reading the related module.
3. State what you think the code should do before changing it.
4. Make the smallest fix that restores the intended behavior.
5. Re-run the focused test, then the full suite.

## Project layout

- `bug_squash/pure_python/`: standard-library implementation
- `bug_squash/pytorch/`: PyTorch implementation
- `tests/pure_python/`: tests for the pure Python exercise
- `tests/pytorch/`: tests for the PyTorch exercise
- `data/`: shared churn dataset

The bugs are in the implementation, not the tests.

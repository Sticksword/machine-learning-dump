# ML Bug Squash Practice

This directory contains a small, interview-style ML debugging exercise. The codebase is intentionally tidy and readable, but it contains a few bugs in the library code. The tests are correct and should initially fail.

## Scenario

You have inherited a lightweight churn-prediction training pipeline:

- A local CSV dataset is loaded from `data/customer_churn.csv`
- The data is split into train and test sets
- Features are standardized
- A logistic-regression model is trained with gradient descent
- The pipeline reports test accuracy and log loss

Your goal is to use the tests and the codebase to track down the bugs and get the suite passing.

## Running the exercise

From this directory:

```bash
python3 -m unittest discover -s tests -v
```

You can also inspect the end-to-end behavior with:

```bash
python3 train_churn_model.py
```

## What to practice

- Forming a debugging hypothesis before making changes
- Narrowing the failure from integration tests to specific modules
- Using the passing tests to understand intended behavior
- Fixing the implementation quickly without over-engineering

## Suggested approach

1. Run the full test suite and group failures by subsystem.
2. Re-run a single failing test while reading the related module.
3. State what you think the code should do before changing it.
4. Make the smallest fix that restores the intended behavior.
5. Re-run the focused test, then the full suite.

## Project layout

- `bug_squash/datasets.py`: CSV loading
- `bug_squash/splitting.py`: dataset splitting
- `bug_squash/preprocessing.py`: feature scaling
- `bug_squash/model.py`: logistic regression
- `bug_squash/pipeline.py`: end-to-end training entry point
- `tests/`: unit and integration tests

The bugs are all in the library code, not the tests.

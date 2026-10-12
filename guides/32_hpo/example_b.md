# HYPERPARAMETER OPTIMIZATION — Example B

**Random search over tree settings**  
Book page: **133** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/32_hpo/example_b.py) · [All topics](../../README.md)

## What this example teaches

Searches settings that govern training or model structure. Grid search evaluates a defined grid; random search samples settings; multi-fidelity methods allocate different budgets. Selection must use validation, not final test data.

## Python code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
X, y = load_iris(return_X_y=True)
A, B, ya, yb = train_test_split(X, y, stratify=y, random_state=42)
search = RandomizedSearchCV(RandomForestClassifier(random_state=42),
    {"max_depth": [2, 3, 5, None], "min_samples_leaf": [1, 2, 4]},
    n_iter=5, cv=3, random_state=42)
search.fit(A, ya)
print("Selected:", search.best_params_)
print("Final test accuracy:", search.score(B, yb))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import load_iris` | Import load_iris. |
| 2 | `from sklearn.model_selection import (` | Import train_test_split, RandomizedSearchCV. |
| 3 | `    train_test_split,` | Supply arguments to the preceding operation. |
| 4 | `    RandomizedSearchCV,` | Supply arguments to the preceding operation. |
| 5 | `)` | Close the preceding multi-line operation. |
| 6 | `from sklearn.ensemble import RandomForestClassifier` | Import RandomForestClassifier. |
| 7 | `X, y = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 8 | `(A, B, ya, yb) = (` | Reserve unseen data for testing. |
| 9 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `search = (` | Create the training-only hyperparameter search. |
| 12 | `    RandomizedSearchCV(RandomForestClassifier(random_state=42),` | Choose the estimator and its tuning options. |
| 13 | `    {'max_depth': [2, 3, 5, None], 'min_samples_leaf': [1, 2, 4` | Supply arguments to the preceding operation. |
| 14 | `    ]}, n_iter=5, cv=3, random_state=42)` | Fix the seed and other options for repeatability. |
| 15 | `)` | Close the preceding multi-line operation. |
| 16 | `search.fit(A, ya)` | Learn parameters from the training data. |
| 17 | `print('Selected:', search.best_params_)` | Print the result shown in the output box. |
| 18 | `print('Final test accuracy:', search.score(B, yb))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Selected: {'min_samples_leaf': 2, 'max_depth': None}
Final test accuracy: 0.9210526315789473
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Random search samples five configurations from the defined tree setting combinations.

## Try it yourself

Change n_iter and record whether additional computation improves validation performance.

## Important check

A selected setting is best only among the evaluated candidates under this validation scheme.

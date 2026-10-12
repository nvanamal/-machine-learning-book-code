# HYPERPARAMETER OPTIMIZATION — Example A

**Grid search using training-only folds**  
Book page: **132** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/32_hpo/example_a.py) · [All topics](../../README.md)

## What this example teaches

Searches settings that govern training or model structure. Grid search evaluates a defined grid; random search samples settings; multi-fidelity methods allocate different budgets. Selection must use validation, not final test data.

## Python code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
A, B, ya, yb = train_test_split(X, y, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
search = GridSearchCV(model, {"logisticregression__C": [0.1, 1, 10]}, cv=3)
search.fit(A, ya)
print("Selected:", search.best_params_)
print("Final test accuracy:", search.score(B, yb))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import load_iris` | Import load_iris. |
| 2 | `from sklearn.model_selection import (` | Import train_test_split, GridSearchCV. |
| 3 | `    train_test_split,` | Supply arguments to the preceding operation. |
| 4 | `    GridSearchCV,` | Supply arguments to the preceding operation. |
| 5 | `)` | Close the preceding multi-line operation. |
| 6 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 7 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 8 | `from sklearn.linear_model import LogisticRegression` | Import LogisticRegression. |
| 9 | `X, y = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 10 | `(A, B, ya, yb) = (` | Reserve unseen data for testing. |
| 11 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 12 | `)` | Close the preceding multi-line operation. |
| 13 | `model = (` | Chain preprocessing with the estimator. |
| 14 | `    make_pipeline(StandardScaler(), LogisticRegression(max_iter=` | Choose the estimator and its tuning options. |
| 15 | `    1000))` | Supply arguments to the preceding operation. |
| 16 | `)` | Close the preceding multi-line operation. |
| 17 | `search = (` | Create the training-only hyperparameter search. |
| 18 | `    GridSearchCV(model, {'logisticregression__C': [0.1, 1, 10]},` | Supply arguments to the preceding operation. |
| 19 | `    cv=3)` | Supply arguments to the preceding operation. |
| 20 | `)` | Close the preceding multi-line operation. |
| 21 | `search.fit(A, ya)` | Learn parameters from the training data. |
| 22 | `print('Selected:', search.best_params_)` | Print the result shown in the output box. |
| 23 | `print('Final test accuracy:', search.score(B, yb))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Selected: {'logisticregression__C': 1}
Final test accuracy: 0.9210526315789473
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The grid selects C using training folds, then evaluates the selected pipeline once on reserved test rows.

## Try it yourself

Expand the C grid and record both validation scores and the number of fits.

## Important check

The final test is not part of the search and must not guide repeated search revisions.

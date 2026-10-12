# HYPERPARAMETER OPTIMIZATION — Example C

**Validation curve for regularization**  
Book page: **134** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/32_hpo/example_c.py) · [All topics](../../README.md)

## What this example teaches

Searches settings that govern training or model structure. Grid search evaluates a defined grid; random search samples settings; multi-fidelity methods allocate different budgets. Selection must use validation, not final test data.

## Python code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
A, B, ya, yb = train_test_split(X, y, stratify=y, random_state=42)
for C in [0.01, 0.1, 1, 10]:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    print(C, cross_val_score(model, A, ya, cv=3).mean())
# Do not use B/yb to choose C; they remain reserved for a final test.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import load_iris` | Import load_iris. |
| 2 | `from sklearn.model_selection import (` | Import cross_val_score, train_test_split. |
| 3 | `    cross_val_score,` | Supply arguments to the preceding operation. |
| 4 | `    train_test_split,` | Supply arguments to the preceding operation. |
| 5 | `)` | Close the preceding multi-line operation. |
| 6 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 7 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 8 | `from sklearn.linear_model import LogisticRegression` | Import LogisticRegression. |
| 9 | `X, y = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 10 | `(A, B, ya, yb) = (` | Reserve unseen data for testing. |
| 11 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 12 | `)` | Close the preceding multi-line operation. |
| 13 | `for C in [0.01, 0.1, 1, 10]:` | Repeat episodes or steps of interaction. |
| 14 | `    model = (` | Chain preprocessing with the estimator. |
| 15 | `        make_pipeline(StandardScaler(), LogisticRegression(C=C,` | Choose the estimator and its tuning options. |
| 16 | `        max_iter=1000))` | Supply arguments to the preceding operation. |
| 17 | `    )` | Close the preceding multi-line operation. |
| 18 | `    print(C, cross_val_score(model, A, ya, cv=3).mean())` | Print the result shown in the output box. |

## Recorded output from the book

```text
0.01 0.8302513039355145
0.1 0.8845424371740161
1 0.9551920341394026
10 0.9461830251303937
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The printed curve compares training-CV accuracy across regularization strengths; test labels remain unused.

## Try it yourself

Plot the fold means and spreads against log C and identify a stable plateau.

## Important check

A small difference in means may reflect fold variability rather than a meaningful model improvement.

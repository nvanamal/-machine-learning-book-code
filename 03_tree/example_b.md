# DECISION TREE — Example B

**Numeric regression with a tree**  
Book page: **17** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/03_tree/example_b.py) · [All topics](../../README.md)

## What this example teaches

Uses successive feature tests to divide data into regions. Internal nodes are tests, branches are outcomes, and leaves contain a class prediction or numeric value. A shallow tree can be displayed as human-readable rules.

## Python code

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
X, y = make_regression(n_samples=450, n_features=8,
    n_informative=4, noise=15, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
model = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=5, random_state=42))
model.fit(Xtr, ytr)
print("MAE:", mean_absolute_error(yte, model.predict(Xte)))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_regression` | Import a numeric-target data generator. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 4 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 5 | `from sklearn.tree import DecisionTreeRegressor` | Import DecisionTreeRegressor. |
| 6 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 7 | `(X, y) = (` | Generate inputs and numeric targets. |
| 8 | `    make_regression(n_samples=450, n_features=8, n_informative=4` | Set feature count and useful-signal dimensions. |
| 9 | `    , noise=15, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)` | Reserve unseen data for testing. |
| 12 | `model = (` | Chain preprocessing with the estimator. |
| 13 | `    make_pipeline(StandardScaler(), DecisionTreeRegressor(` | Choose the estimator and its tuning options. |
| 14 | `    max_depth=5, random_state=42))` | Fix the seed and other options for repeatability. |
| 15 | `)` | Close the preceding multi-line operation. |
| 16 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 17 | `print('MAE:', mean_absolute_error(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
MAE: 31.648794576732353
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Regression leaves store numeric predictions. MAE evaluates their absolute error; it is not a classification accuracy.

## Try it yourself

Change max_depth and plot prediction against one feature while holding other inputs fixed.

## Important check

A tree is piecewise constant and usually does not extrapolate beyond its training target range.

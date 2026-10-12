# RIDGE, LASSO & ELASTIC NET — Example B

**Lasso numeric regression**  
Book page: **41** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/09_ridge/example_b.py) · [All topics](../../README.md)

## What this example teaches

Penalize coefficients to control model complexity. Ridge shrinks weights; Lasso can set weights to zero; Elastic Net combines both penalties. These extend linear regression while keeping a linear prediction function.

## Python code

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
X, y = make_regression(n_samples=450, n_features=8,
    n_informative=4, noise=15, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
model = make_pipeline(StandardScaler(), Lasso(alpha=0.2, max_iter=5000))
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
| 5 | `from sklearn.linear_model import Lasso` | Import Lasso. |
| 6 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 7 | `(X, y) = (` | Generate inputs and numeric targets. |
| 8 | `    make_regression(n_samples=450, n_features=8, n_informative=4` | Set feature count and useful-signal dimensions. |
| 9 | `    , noise=15, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)` | Reserve unseen data for testing. |
| 12 | `model = (` | Chain preprocessing with the estimator. |
| 13 | `    make_pipeline(StandardScaler(), Lasso(alpha=0.2, max_iter=` | Standardize using statistics fitted on training data. |
| 14 | `    5000))` | Supply arguments to the preceding operation. |
| 15 | `)` | Close the preceding multi-line operation. |
| 16 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 17 | `print('MAE:', mean_absolute_error(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
MAE: 11.865278239954073
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Lasso can shrink some coefficients exactly to zero. MAE describes prediction error, not feature-selection truth.

## Try it yourself

Print the fitted coefficients and count how many are zero as alpha increases.

## Important check

With correlated features, Lasso may choose one of several similar predictors inconsistently.

# RIDGE, LASSO & ELASTIC NET — Example C

**Elastic Net combines two penalties**  
Book page: **42** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/09_ridge/example_c.py) · [All topics](../../README.md)

## What this example teaches

Penalize coefficients to control model complexity. Ridge shrinks weights; Lasso can set weights to zero; Elastic Net combines both penalties. These extend linear regression while keeping a linear prediction function.

## Python code

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
X, y = make_regression(n_samples=450, n_features=8,
    n_informative=4, noise=15, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.2, l1_ratio=0.5, max_iter=5000))
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
| 5 | `from sklearn.linear_model import ElasticNet` | Import ElasticNet. |
| 6 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 7 | `(X, y) = (` | Generate inputs and numeric targets. |
| 8 | `    make_regression(n_samples=450, n_features=8, n_informative=4` | Set feature count and useful-signal dimensions. |
| 9 | `    , noise=15, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)` | Reserve unseen data for testing. |
| 12 | `model = (` | Chain preprocessing with the estimator. |
| 13 | `    make_pipeline(StandardScaler(), ElasticNet(alpha=0.2,` | Standardize using statistics fitted on training data. |
| 14 | `    l1_ratio=0.5, max_iter=5000))` | Supply arguments to the preceding operation. |
| 15 | `)` | Close the preceding multi-line operation. |
| 16 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 17 | `print('MAE:', mean_absolute_error(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
MAE: 14.308866099063021
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Elastic Net combines L1 and L2 penalties. l1_ratio balances sparsity with quadratic shrinkage.

## Try it yourself

Compare l1_ratio=0.2, 0.5 and 0.8 using the same validation folds.

## Important check

Nonzero coefficients are predictive associations, not evidence that the selected inputs cause the target.

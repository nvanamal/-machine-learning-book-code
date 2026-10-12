# RIDGE, LASSO & ELASTIC NET — Example A

**Basic fit and held-out prediction**  
Book page: **40** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/09_ridge/example_a.py) · [All topics](../../README.md)

## What this example teaches

Penalize coefficients to control model complexity. Ridge shrinks weights; Lasso can set weights to zero; Elastic Net combines both penalties. These extend linear regression while keeping a linear prediction function.

## Python code

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
X, y = make_regression(n_samples=600, n_features=12,
                       noise=20, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y,
                       test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), Ridge(alpha=10))
model.fit(Xtr, ytr)
print("MAE:", mean_absolute_error(yte, model.predict(Xte)))
# Substitute Lasso(alpha=...) or ElasticNet(...)
# Tune alpha with cross-validation on Xtr only.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_regression` | Import a numeric-target data generator. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 4 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 5 | `from sklearn.linear_model import Ridge` | Import Ridge. |
| 6 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 7 | `(X, y) = (` | Generate inputs and numeric targets. |
| 8 | `    make_regression(n_samples=600, n_features=12, noise=20,` | Set feature count and useful-signal dimensions. |
| 9 | `    random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 12 | `    train_test_split(X, y, test_size=0.2, random_state=42)` | Fix the seed and other options for repeatability. |
| 13 | `)` | Close the preceding multi-line operation. |
| 14 | `model = make_pipeline(StandardScaler(), Ridge(alpha=10))` | Chain preprocessing with the estimator. |
| 15 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 16 | `print('MAE:', mean_absolute_error(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
MAE: 17.989039122913336
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Ridge penalizes squared coefficient magnitudes. Standardization makes penalties more comparable across features.

## Try it yourself

Compare alpha values 0.1, 1 and 10 on training-only folds.

## Important check

Larger alpha means stronger regularization; this direction differs from classifier parameter C.

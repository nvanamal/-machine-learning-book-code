# GRADIENT BOOSTING — Example B

**Boosted-tree numeric regression**  
Book page: **37** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/08_boost/example_b.py) · [All topics](../../README.md)

## What this example teaches

Adds weak learners in sequence to improve an objective. Each new learner fits the negative gradient of the loss. For squared-error regression this is a residual; classification uses loss-specific gradients.

## Python code

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
X, y = make_regression(n_samples=450, n_features=8,
    n_informative=4, noise=15, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
model = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))
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
| 5 | `from sklearn.ensemble import GradientBoostingRegressor` | Import GradientBoostingRegressor. |
| 6 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 7 | `(X, y) = (` | Generate inputs and numeric targets. |
| 8 | `    make_regression(n_samples=450, n_features=8, n_informative=4` | Set feature count and useful-signal dimensions. |
| 9 | `    , noise=15, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)` | Reserve unseen data for testing. |
| 12 | `model = (` | Chain preprocessing with the estimator. |
| 13 | `    make_pipeline(StandardScaler(), GradientBoostingRegressor(` | Choose the estimator and its tuning options. |
| 14 | `    random_state=42))` | Fix the seed and other options for repeatability. |
| 15 | `)` | Close the preceding multi-line operation. |
| 16 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 17 | `print('MAE:', mean_absolute_error(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
MAE: 16.35157523845117
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

GradientBoostingRegressor uses a numeric target and averages no independent forest; it builds an additive model.

## Try it yourself

Compare depth 1, 2 and 3 and track both training and held-out MAE.

## Important check

Do not describe every boosting step as an ordinary residual: that is specific to squared-error regression.

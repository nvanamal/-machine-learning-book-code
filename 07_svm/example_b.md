# SUPPORT VECTOR MACHINE (SVM) — Example B

**Support-vector numeric regression**  
Book page: **33** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/07_svm/example_b.py) · [All topics](../../README.md)

## What this example teaches

Balances a large separation margin against violations. A kernel can produce nonlinear boundaries. SVC classifies, SVR predicts numeric outcomes, and OneClassSVM supports novelty detection.

## Python code

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
X, y = make_regression(n_samples=450, n_features=8,
    n_informative=4, noise=15, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
model = make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.1))
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
| 5 | `from sklearn.svm import SVR` | Import SVR. |
| 6 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 7 | `(X, y) = (` | Generate inputs and numeric targets. |
| 8 | `    make_regression(n_samples=450, n_features=8, n_informative=4` | Set feature count and useful-signal dimensions. |
| 9 | `    , noise=15, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)` | Reserve unseen data for testing. |
| 12 | `model = make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.1))` | Chain preprocessing with the estimator. |
| 13 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 14 | `print('MAE:', mean_absolute_error(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
MAE: 22.188153166223366
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

SVR predicts numeric targets with an epsilon-insensitive loss. The printed MAE evaluates held-out errors.

## Try it yourself

Compare epsilon values and record how the number of support vectors changes.

## Important check

SVR hyperparameters depend on target scale; input scaling alone does not standardize the target.

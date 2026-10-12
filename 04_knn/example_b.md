# K-NEAREST NEIGHBOURS (KNN) — Example B

**Distance-weighted numeric regression**  
Book page: **21** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/04_knn/example_b.py) · [All topics](../../README.md)

## What this example teaches

Predicts from the k nearest stored training examples. Classification uses voting; regression uses an average. Distance weighting gives closer examples more influence. Fitting stores observations and may build a search structure.

## Python code

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
X, y = make_regression(n_samples=450, n_features=8,
    n_informative=4, noise=15, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7, weights="distance"))
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
| 5 | `from sklearn.neighbors import KNeighborsRegressor` | Import KNeighborsRegressor. |
| 6 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 7 | `(X, y) = (` | Generate inputs and numeric targets. |
| 8 | `    make_regression(n_samples=450, n_features=8, n_informative=4` | Set feature count and useful-signal dimensions. |
| 9 | `    , noise=15, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)` | Reserve unseen data for testing. |
| 12 | `model = (` | Chain preprocessing with the estimator. |
| 13 | `    make_pipeline(StandardScaler(), KNeighborsRegressor(` | Choose the estimator and its tuning options. |
| 14 | `    n_neighbors=7, weights='distance'))` | Set distance weighting for neighbour predictions. |
| 15 | `)` | Close the preceding multi-line operation. |
| 16 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 17 | `print('MAE:', mean_absolute_error(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
MAE: 33.683610712595474
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Regression averages nearby target values. The example uses distance weights rather than fitting a global equation.

## Try it yourself

Compare MAE for several k values and record prediction latency as the dataset grows.

## Important check

Small k can track noise; large k can smooth away meaningful local variation.

# LINEAR REGRESSION — Example C

**OLS cross-validation on training data**  
Book page: **10** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/01_linear/example_c.py) · [All topics](../../README.md)

## What this example teaches

Fits a linear combination of input features to predict a numeric target. Ordinary least squares (OLS) chooses coefficients that minimize squared residuals. The model is linear in its coefficients; features can include polynomial terms.

## Python code

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
X, y = make_regression(n_samples=450, n_features=6,
                       noise=20, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
model = LinearRegression()
scores = cross_val_score(model, Xtr, ytr, cv=3,
                         scoring="neg_mean_absolute_error")
print("CV MAE:", -scores.mean())
model.fit(Xtr, ytr)
print("Test MAE:", mean_absolute_error(yte, model.predict(Xte)))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_regression` | Import a numeric-target data generator. |
| 2 | `from sklearn.model_selection import (` | Import train_test_split, cross_val_score. |
| 3 | `    train_test_split,` | Supply arguments to the preceding operation. |
| 4 | `    cross_val_score,` | Supply arguments to the preceding operation. |
| 5 | `)` | Close the preceding multi-line operation. |
| 6 | `from sklearn.linear_model import LinearRegression` | Import LinearRegression. |
| 7 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 8 | `(X, y) = (` | Generate inputs and numeric targets. |
| 9 | `    make_regression(n_samples=450, n_features=6, noise=20,` | Set feature count and useful-signal dimensions. |
| 10 | `    random_state=42)` | Fix the seed and other options for repeatability. |
| 11 | `)` | Close the preceding multi-line operation. |
| 12 | `Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)` | Reserve unseen data for testing. |
| 13 | `model = LinearRegression()` | Create or fit the selected estimator. |
| 14 | `scores = (` | Compute compatibility, class or preference scores. |
| 15 | `    cross_val_score(model, Xtr, ytr, cv=3, scoring=` | Choose the metric used for validation. |
| 16 | `    'neg_mean_absolute_error')` | Compute mean absolute test-prediction error. |
| 17 | `)` | Close the preceding multi-line operation. |
| 18 | `print('CV MAE:', -scores.mean())` | Print the result shown in the output box. |
| 19 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 20 | `print('Test MAE:', mean_absolute_error(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
CV MAE: 15.860078427524906
Test MAE: 17.00903878511473
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

scikit-learn returns negative MAE for scoring. Negating the mean gives a positive error whose smaller value is better.

## Try it yourself

Change cv from 3 to 5 and examine the individual fold errors as well as their mean.

## Important check

Fold results vary because training subsets differ. Their spread is not automatically a confidence interval.

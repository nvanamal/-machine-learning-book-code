# LINEAR REGRESSION — Example A

**Basic fit and held-out prediction**  
Book page: **8** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/01_linear/example_a.py) · [All topics](../../README.md)

## What this example teaches

Fits a linear combination of input features to predict a numeric target. Ordinary least squares (OLS) chooses coefficients that minimize squared residuals. The model is linear in its coefficients; features can include polynomial terms.

## Python code

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
X, y = make_regression(n_samples=600, n_features=4,
                       noise=15, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("MAE:", mean_absolute_error(yte, pred))
print("R2:", r2_score(yte, pred))
print("Slopes:", model.coef_)
print("Intercept:", model.intercept_)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_regression` | Import a numeric-target data generator. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.linear_model import LinearRegression` | Import LinearRegression. |
| 4 | `from sklearn.metrics import mean_absolute_error, r2_score` | Import mean_absolute_error, r2_score. |
| 5 | `(X, y) = (` | Generate inputs and numeric targets. |
| 6 | `    make_regression(n_samples=600, n_features=4, noise=15,` | Set feature count and useful-signal dimensions. |
| 7 | `    random_state=42)` | Fix the seed and other options for repeatability. |
| 8 | `)` | Close the preceding multi-line operation. |
| 9 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 10 | `    train_test_split(X, y, test_size=0.2, random_state=42)` | Fix the seed and other options for repeatability. |
| 11 | `)` | Close the preceding multi-line operation. |
| 12 | `model = LinearRegression()` | Create or fit the selected estimator. |
| 13 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 14 | `pred = model.predict(Xte)` | Predict from data the model has not fitted. |
| 15 | `print('MAE:', mean_absolute_error(yte, pred))` | Print the result shown in the output box. |
| 16 | `print('R2:', r2_score(yte, pred))` | Print the result shown in the output box. |
| 17 | `print('Slopes:', model.coef_)` | Print the result shown in the output box. |
| 18 | `print('Intercept:', model.intercept_)` | Print the result shown in the output box. |

## Recorded output from the book

```text
MAE: 11.002231138919793
R2: 0.9834570175659038
Slopes: [ 7.71458681 96.03498919 61.15650265  7.71928492]
Intercept: 0.5642966039479007
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

MAE reports average absolute test error in target units. R² compares squared error with a mean-target baseline; slopes describe changes per feature unit.

## Try it yourself

Double the generated noise and rerun the same split. Record how MAE and R² change.

## Important check

Synthetic data follow a linear model here. This does not prove that real measurements have a linear response.

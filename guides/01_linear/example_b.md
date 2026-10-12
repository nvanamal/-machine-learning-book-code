# LINEAR REGRESSION — Example B

**Polynomial regression for a curved target**  
Book page: **9** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/01_linear/example_b.py) · [All topics](../../README.md)

## What this example teaches

Fits a linear combination of input features to predict a numeric target. Ordinary least squares (OLS) chooses coefficients that minimize squared residuals. The model is linear in its coefficients; features can include polynomial terms.

## Python code

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(400, 1))
y = 2 * X[:, 0] ** 2 + rng.normal(size=400)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
model = make_pipeline(PolynomialFeatures(2, include_bias=False),
                      LinearRegression())
model.fit(Xtr, ytr)
print("Polynomial MAE:", mean_absolute_error(yte, model.predict(Xte)))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 4 | `from sklearn.preprocessing import PolynomialFeatures` | Import polynomial feature expansion. |
| 5 | `from sklearn.linear_model import LinearRegression` | Import LinearRegression. |
| 6 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 7 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 8 | `X = rng.uniform(-3, 3, size=(400, 1))` | Build or transform the input feature matrix. |
| 9 | `y = 2 * X[:, 0] ** 2 + rng.normal(size=400)` | Define or generate the corresponding targets. |
| 10 | `Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)` | Reserve unseen data for testing. |
| 11 | `model = (` | Chain preprocessing with the estimator. |
| 12 | `    make_pipeline(PolynomialFeatures(2, include_bias=False),` | Supply arguments to the preceding operation. |
| 13 | `    LinearRegression())` | Supply arguments to the preceding operation. |
| 14 | `)` | Close the preceding multi-line operation. |
| 15 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 16 | `(` | Print the result shown in the output box. |
| 17 | `    print('Polynomial MAE:', mean_absolute_error(yte, model.` | Compute mean absolute test-prediction error. |
| 18 | `    predict(Xte)))` | Supply arguments to the preceding operation. |
| 19 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
Polynomial MAE: 0.8585706110886061
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

PolynomialFeatures creates x and x². The fitted model is linear in its coefficients while its response can curve in x.

## Try it yourself

Compare degrees 1, 2 and 3 using training-only validation; keep the test partition fixed.

## Important check

Higher degree can fit noise and extrapolate badly. Do not select degree by repeatedly inspecting the final test.

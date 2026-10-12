# GAUSSIAN PROCESS REGRESSION — Example C

**Held-out GP regression error**  
Book page: **130** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/31_gp/example_c.py) · [All topics](../../README.md)

## What this example teaches

Places a probability distribution over functions. A mean function and covariance kernel define prior relationships between inputs. Conditioning on noisy observations gives a predictive mean and uncertainty under the model.

## Python code

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_error
X = np.linspace(0, 4, 40)[:, None]; y = np.sin(X[:, 0])
A, B, ya, yb = train_test_split(X, y, random_state=42)
model = GaussianProcessRegressor(kernel=RBF(1.0), alpha=0.01, optimizer=None).fit(A, ya)
print("Held-out MAE:", mean_absolute_error(yb, model.predict(B)))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.gaussian_process import GaussianProcessRegressor` | Import GaussianProcessRegressor. |
| 4 | `from sklearn.gaussian_process.kernels import RBF` | Import RBF. |
| 5 | `from sklearn.metrics import mean_absolute_error` | Import absolute-error evaluation. |
| 6 | `X = np.linspace(0, 4, 40)[:, None]` | Build or transform the input feature matrix. |
| 7 | `y = np.sin(X[:, 0])` | Define or generate the corresponding targets. |
| 8 | `A, B, ya, yb = train_test_split(X, y, random_state=42)` | Reserve unseen data for testing. |
| 9 | `model = (` | Fit model parameters using training inputs. |
| 10 | `    GaussianProcessRegressor(kernel=RBF(1.0), alpha=0.01,` | Choose the estimator and its tuning options. |
| 11 | `    optimizer=None).fit(A, ya)` | Supply arguments to the preceding operation. |
| 12 | `)` | Close the preceding multi-line operation. |
| 13 | `(` | Print the result shown in the output box. |
| 14 | `    print('Held-out MAE:', mean_absolute_error(yb, model.predict` | Compute mean absolute test-prediction error. |
| 15 | `    (B)))` | Supply arguments to the preceding operation. |
| 16 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
Held-out MAE: 0.00804975997357683
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Held-out MAE measures prediction error at omitted input locations; the kernel was fixed in this example.

## Try it yourself

Hold out a whole contiguous input interval and compare with random-row validation.

## Important check

Random interpolation error can underestimate errors for a genuinely new region or design.

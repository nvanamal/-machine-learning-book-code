# GAUSSIAN PROCESS REGRESSION — Example A

**GP mean and standard deviation**  
Book page: **128** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/31_gp/example_a.py) · [All topics](../../README.md)

## What this example teaches

Places a probability distribution over functions. A mean function and covariance kernel define prior relationships between inputs. Conditioning on noisy observations gives a predictive mean and uncertainty under the model.

## Python code

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
X = np.linspace(0, 3, 12)[:, None]; y = np.sin(X[:, 0])
model = GaussianProcessRegressor(kernel=RBF(1.0), alpha=0.01, optimizer=None)
model.fit(X, y)
mean, std = model.predict(np.array([[1.5], [4.0]]), return_std=True)
print("Means:", np.round(mean, 3))
print("Model standard deviations:", np.round(std, 3))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.gaussian_process import GaussianProcessRegressor` | Import GaussianProcessRegressor. |
| 3 | `from sklearn.gaussian_process.kernels import RBF` | Import RBF. |
| 4 | `X = np.linspace(0, 3, 12)[:, None]` | Build or transform the input feature matrix. |
| 5 | `y = np.sin(X[:, 0])` | Define or generate the corresponding targets. |
| 6 | `model = (` | Create or fit the selected estimator. |
| 7 | `    GaussianProcessRegressor(kernel=RBF(1.0), alpha=0.01,` | Choose the estimator and its tuning options. |
| 8 | `    optimizer=None)` | Supply arguments to the preceding operation. |
| 9 | `)` | Close the preceding multi-line operation. |
| 10 | `model.fit(X, y)` | Learn parameters from the training data. |
| 11 | `(mean, std) = (` | Predict GP mean and model standard deviation. |
| 12 | `    model.predict(np.array([[1.5], [4.0]]), return_std=True)` | Supply arguments to the preceding operation. |
| 13 | `)` | Close the preceding multi-line operation. |
| 14 | `print('Means:', np.round(mean, 3))` | Print the result shown in the output box. |
| 15 | `print('Model standard deviations:', np.round(std, 3))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Means: [ 0.999 -0.286]
Model standard deviations: [0.058 0.628]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The GP returns a mean and a model-based standard deviation. Locations farther from observations can have larger uncertainty.

## Try it yourself

Add an observation near the distant query and examine the standard deviation there.

## Important check

The uncertainty is conditional on the kernel and noise model; it is not a guaranteed error bound.

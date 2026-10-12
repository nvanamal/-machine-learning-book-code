# BATCH NORMALIZATION — Example A

**Normalize each feature in a batch**  
Book page: **116** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/28_batchnorm/example_a.py) · [All topics](../../README.md)

## What this example teaches

Normalizes activations using minibatch statistics during training, followed by learned scale and shift. In evaluation it usually uses accumulated running statistics. It is distinct from input scaling and LayerNorm.

## Python code

```python
import numpy as np
X = np.array([[1., 10.], [2., 20.], [3., 30.]])
mean = X.mean(axis=0); var = X.var(axis=0)
Y = (X - mean) / np.sqrt(var + 1e-5)
print("Normalized means:", np.round(Y.mean(axis=0), 5))
print("Normalized variances:", np.round(Y.var(axis=0), 5))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])` | Build or transform the input feature matrix. |
| 3 | `mean = X.mean(axis=0)` | Assign mean for the next step. |
| 4 | `var = X.var(axis=0)` | Assign var for the next step. |
| 5 | `Y = (X - mean) / np.sqrt(var + 1e-05)` | Assign Y for the next step. |
| 6 | `print('Normalized means:', np.round(Y.mean(axis=0), 5))` | Print the result shown in the output box. |
| 7 | `print('Normalized variances:', np.round(Y.var(axis=0), 5))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Normalized means: [0. 0.]
Normalized variances: [0.99999 1.     ]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Each feature is centered and scaled using the current batch. Variance is near one, with a small epsilon effect.

## Try it yourself

Make one feature constant and inspect how epsilon prevents division by zero.

## Important check

This batch computation alone does not provide running statistics for inference.

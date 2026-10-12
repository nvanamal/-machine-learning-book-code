# BATCH NORMALIZATION — Example C

**Running statistics for inference**  
Book page: **118** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/28_batchnorm/example_c.py) · [All topics](../../README.md)

## What this example teaches

Normalizes activations using minibatch statistics during training, followed by learned scale and shift. In evaluation it usually uses accumulated running statistics. It is distinct from input scaling and LayerNorm.

## Python code

```python
import numpy as np
running_mean, running_var = 0.0, 1.0
momentum = 0.1  # fraction of new statistics
batch = np.array([1., 2., 3.])
running_mean = (1 - momentum) * running_mean + momentum * batch.mean()
running_var = (1 - momentum) * running_var + momentum * batch.var()
query = np.array([2.])
y = (query - running_mean) / np.sqrt(running_var + 1e-5)
print("Running mean/variance:", running_mean, running_var)
print("Inference value:", y)
# Toy convention; frameworks can use different variance estimators.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `running_mean, running_var = (0.0, 1.0)` | Assign running_mean, running_var for the next step. |
| 3 | `momentum = 0.1` | Assign momentum for the next step. |
| 4 | `batch = np.array([1.0, 2.0, 3.0])` | Assign batch for the next step. |
| 5 | `running_mean = (` | Assign running_mean for the next step. |
| 6 | `    (1 - momentum) * running_mean + momentum * batch.mean()` | Supply arguments to the preceding operation. |
| 7 | `)` | Close the preceding multi-line operation. |
| 8 | `running_var = (` | Assign running_var for the next step. |
| 9 | `    (1 - momentum) * running_var + momentum * batch.var()` | Supply arguments to the preceding operation. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `query = np.array([2.0])` | Assign query for the next step. |
| 12 | `y = (query - running_mean) / np.sqrt(running_var + 1e-05)` | Define or generate the corresponding targets. |
| 13 | `print('Running mean/variance:', running_mean, running_var)` | Print the result shown in the output box. |
| 14 | `print('Inference value:', y)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Running mean/variance: 0.2 0.9666666666666667
Inference value: [1.83076199]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Running statistics blend old values and current batch moments, then normalize an unseen query.

## Try it yourself

Compare several batches in sequence and inspect how the running values adapt.

## Important check

Momentum and variance-estimator conventions differ across frameworks; this toy states its own convention.

# SGD, MOMENTUM & ADAM — Example C

**Adam with bias correction**  
Book page: **122** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/29_optimizers/example_c.py) · [All topics](../../README.md)

## What this example teaches

Optimization algorithms update model parameters using gradients. SGD uses sampled gradients; momentum averages directional information; Adam tracks first and second moments with bias correction.

## Python code

```python
import numpy as np
theta, m, v = 5.0, 0.0, 0.0
b1, b2 = 0.9, 0.999
for t in range(1, 301):
    gradient = 2 * (theta - 2)
    m = b1 * m + (1 - b1) * gradient
    v = b2 * v + (1 - b2) * gradient ** 2
    mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
    theta -= 0.05 * mhat / (np.sqrt(vhat) + 1e-8)
print("Adam parameter:", round(theta, 6))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `theta, m, v = (5.0, 0.0, 0.0)` | Assign theta, m, v for the next step. |
| 3 | `b1, b2 = (0.9, 0.999)` | Assign b1, b2 for the next step. |
| 4 | `for t in range(1, 301):` | Repeat episodes or steps of interaction. |
| 5 | `    gradient = 2 * (theta - 2)` | Compute the loss gradient at this step. |
| 6 | `    m = b1 * m + (1 - b1) * gradient` | Assign m for the next step. |
| 7 | `    v = b2 * v + (1 - b2) * gradient ** 2` | Assign v for the next step. |
| 8 | `    mhat = m / (1 - b1 ** t)` | Bias-correct the first gradient moment. |
| 9 | `    vhat = v / (1 - b2 ** t)` | Bias-correct the second gradient moment. |
| 10 | `    theta -= 0.05 * mhat / (np.sqrt(vhat) + 1e-08)` | Update theta. |
| 11 | `print('Adam parameter:', round(theta, 6))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Adam parameter: 2.0
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Adam bias-corrects the first and second moments before dividing by the estimated gradient scale.

## Try it yourself

Remove bias correction temporarily and compare the first few parameter updates.

## Important check

A faster quadratic example does not establish that Adam generalizes better on every neural task.

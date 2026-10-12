# SGD, MOMENTUM & ADAM — Example A

**Gradient descent on a quadratic**  
Book page: **120** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/29_optimizers/example_a.py) · [All topics](../../README.md)

## What this example teaches

Optimization algorithms update model parameters using gradients. SGD uses sampled gradients; momentum averages directional information; Adam tracks first and second moments with bias correction.

## Python code

```python
theta = 5.0
for step in range(30):
    gradient = 2 * (theta - 2)
    theta -= 0.1 * gradient
print("Parameter:", round(theta, 6))
print("Loss:", round((theta - 2) ** 2, 8))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `theta = 5.0` | Assign theta for the next step. |
| 2 | `for step in range(30):` | Repeat episodes or steps of interaction. |
| 3 | `    gradient = 2 * (theta - 2)` | Compute the loss gradient at this step. |
| 4 | `    theta -= 0.1 * gradient` | Update theta. |
| 5 | `print('Parameter:', round(theta, 6))` | Print the result shown in the output box. |
| 6 | `print('Loss:', round((theta - 2) ** 2, 8))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Parameter: 2.003714
Loss: 1.379e-05
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The gradient points away from the optimum unless subtracted. The parameter approaches the quadratic minimum at two.

## Try it yourself

Increase the learning rate and find when the iterations oscillate or diverge.

## Important check

These updates use a full deterministic gradient, not sampled stochastic gradients.

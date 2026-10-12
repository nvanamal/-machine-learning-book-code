# SGD, MOMENTUM & ADAM — Example B

**Momentum on the same quadratic**  
Book page: **121** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/29_optimizers/example_b.py) · [All topics](../../README.md)

## What this example teaches

Optimization algorithms update model parameters using gradients. SGD uses sampled gradients; momentum averages directional information; Adam tracks first and second moments with bias correction.

## Python code

```python
theta, velocity = 5.0, 0.0
for step in range(100):
    gradient = 2 * (theta - 2)
    velocity = 0.9 * velocity + 0.1 * gradient
    theta -= 0.1 * velocity
print("Momentum parameter:", round(theta, 6))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `theta, velocity = (5.0, 0.0)` | Assign theta, velocity for the next step. |
| 2 | `for step in range(100):` | Repeat episodes or steps of interaction. |
| 3 | `    gradient = 2 * (theta - 2)` | Compute the loss gradient at this step. |
| 4 | `    velocity = 0.9 * velocity + 0.1 * gradient` | Update the momentum estimate. |
| 5 | `    theta -= 0.1 * velocity` | Update theta. |
| 6 | `print('Momentum parameter:', round(theta, 6))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Momentum parameter: 2.012684
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Momentum smooths the update direction using a running gradient average.

## Try it yourself

Compare convergence for beta=0 and beta=0.9 at the same learning rate.

## Important check

Momentum conventions can place the learning-rate factor in different locations.

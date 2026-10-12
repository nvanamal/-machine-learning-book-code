# RESIDUAL NETWORKS (RESNET) — Example B

**Projection aligns feature dimensions**  
Book page: **85** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/20_resnet/example_b.py) · [All topics](../../README.md)

## What this example teaches

Use skip connections to add a shortcut signal to a learned residual transformation. This eases optimization of deep networks. A projection shortcut is used when feature shapes differ.

## Python code

```python
import numpy as np
x = np.array([[1., 2.]])
P = np.array([[1., 0., 1.], [0., 1., 1.]])
W = np.full((2, 3), 0.1)
y = x @ P + np.maximum(0, x @ W)
print("Input/output shapes:", x.shape, y.shape)
print("Projected residual output:", y)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `x = np.array([[1.0, 2.0]])` | Assign x for the next step. |
| 3 | `P = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])` | Assign P for the next step. |
| 4 | `W = np.full((2, 3), 0.1)` | Assign W for the next step. |
| 5 | `y = x @ P + np.maximum(0, x @ W)` | Define or generate the corresponding targets. |
| 6 | `print('Input/output shapes:', x.shape, y.shape)` | Print the result shown in the output box. |
| 7 | `print('Projected residual output:', y)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Input/output shapes: (1, 2) (1, 3)
Projected residual output: [[1.3 2.3 3.3]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

A projection transforms the shortcut from two features to three so it can be added to the residual.

## Try it yourself

Change the output width and update both the projection and residual branch consistently.

## Important check

A shortcut projection is a transformation, not an identity connection.

# RESIDUAL NETWORKS (RESNET) — Example A

**Identity skip connection**  
Book page: **84** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/20_resnet/example_a.py) · [All topics](../../README.md)

## What this example teaches

Use skip connections to add a shortcut signal to a learned residual transformation. This eases optimization of deep networks. A projection shortcut is used when feature shapes differ.

## Python code

```python
import numpy as np
x = np.array([[1., -2.]])
W = np.array([[0.2, 0.1], [0.1, 0.2]])
F = np.maximum(0, x @ W)
y = x + F
print("Residual:", F)
print("Skip plus residual:", y)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `x = np.array([[1.0, -2.0]])` | Assign x for the next step. |
| 3 | `W = np.array([[0.2, 0.1], [0.1, 0.2]])` | Assign W for the next step. |
| 4 | `F = np.maximum(0, x @ W)` | Assign F for the next step. |
| 5 | `y = x + F` | Define or generate the corresponding targets. |
| 6 | `print('Residual:', F)` | Print the result shown in the output box. |
| 7 | `print('Skip plus residual:', y)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Residual: [[0. 0.]]
Skip plus residual: [[ 1. -2.]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The input is added to a residual transformation. This example isolates the skip operation using fixed weights.

## Try it yourself

Set the residual weights to zero and verify that the output becomes exactly the input.

## Important check

Compatible shapes are required for residual addition.

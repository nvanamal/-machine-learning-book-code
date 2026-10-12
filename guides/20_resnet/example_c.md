# RESIDUAL NETWORKS (RESNET) — Example C

**Identity path appears in the gradient**  
Book page: **86** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/20_resnet/example_c.py) · [All topics](../../README.md)

## What this example teaches

Use skip connections to add a shortcut signal to a learned residual transformation. This eases optimization of deep networks. A projection shortcut is used when feature shapes differ.

## Python code

```python
import numpy as np
W = np.array([[0.2, 0.1], [0.1, 0.2]])
# Linear residual toy: y = x + W @ x.
J = np.eye(2) + W
upstream = np.ones(2)
print("Jacobian:", J)
print("Input gradient:", J.T @ upstream)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `W = np.array([[0.2, 0.1], [0.1, 0.2]])` | Assign W for the next step. |
| 3 | `J = np.eye(2) + W` | Assign J for the next step. |
| 4 | `upstream = np.ones(2)` | Assign upstream for the next step. |
| 5 | `print('Jacobian:', J)` | Print the result shown in the output box. |
| 6 | `print('Input gradient:', J.T @ upstream)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Jacobian: [[1.2 0.1]
 [0.1 1.2]]
Input gradient: [1.3 1.3]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The linear residual Jacobian contains an identity matrix, providing a direct contribution to the backward gradient.

## Try it yourself

Compare the input gradient with and without the identity term.

## Important check

A skip path can help optimization, but does not guarantee stable gradients in every deep model.

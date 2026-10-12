# TRANSFER LEARNING / FINE-TUNING — Example C

**Fine-tune a transferred hidden weight matrix**  
Book page: **126** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/30_transfer/example_c.py) · [All topics](../../README.md)

## What this example teaches

Reuse parameters learned on a source task for a target task. Feature extraction freezes a pretrained representation; fine-tuning updates some or all transferred parameters. A new task head often replaces the original output layer.

## Python code

```python
import numpy as np
X = np.array([[1., 0.], [0., 1.]])
y = np.array([[1.], [0.]])
W = np.array([[0.2, 0.1], [0.1, 0.2]])  # transferred toy weights
V = np.full((2, 1), 0.1)  # a new head
for step in range(80):
    h = np.maximum(0, X @ W); pred = h @ V
    error = 2 * (pred - y) / len(X)
    gV = h.T @ error
    gW = X.T @ ((error @ V.T) * (h > 0))
    V -= 0.1 * gV; W -= 0.02 * gW
print("Fine-tuned predictions:", np.round(np.maximum(0, X @ W) @ V, 3))
# Both the new head and transferred layer are updated.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `X = np.array([[1.0, 0.0], [0.0, 1.0]])` | Build or transform the input feature matrix. |
| 3 | `y = np.array([[1.0], [0.0]])` | Define or generate the corresponding targets. |
| 4 | `W = np.array([[0.2, 0.1], [0.1, 0.2]])` | Assign W for the next step. |
| 5 | `V = np.full((2, 1), 0.1)` | Assign V for the next step. |
| 6 | `for step in range(80):` | Repeat episodes or steps of interaction. |
| 7 | `    h = np.maximum(0, X @ W)` | Compute or initialize the hidden representation. |
| 8 | `    pred = h @ V` | Store predictions for evaluation. |
| 9 | `    error = 2 * (pred - y) / len(X)` | Compute a prediction residual or loss derivative. |
| 10 | `    gV = h.T @ error` | Compute the new-head loss gradient. |
| 11 | `    gW = X.T @ (error @ V.T * (h > 0))` | Compute the transferred-layer gradient. |
| 12 | `    V -= 0.1 * gV` | Update V. |
| 13 | `    W -= 0.02 * gW` | Update W. |
| 14 | `(` | Print the result shown in the output box. |
| 15 | `    print('Fine-tuned predictions:', np.round(np.maximum(0, X @` | Supply arguments to the preceding operation. |
| 16 | `    W) @ V, 3))` | Supply arguments to the preceding operation. |
| 17 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
Fine-tuned predictions: [[0.979]
 [0.078]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Both the new output matrix and transferred hidden matrix receive gradients, using a smaller learning rate for the latter.

## Try it yourself

Freeze W and compare the resulting training fit with the fine-tuned version.

## Important check

The toy transferred weights are specified by hand; no real pretrained checkpoint is claimed.

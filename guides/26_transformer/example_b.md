# TRANSFORMER ARCHITECTURE — Example B

**Layer normalization and a residual branch**  
Book page: **109** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/26_transformer/example_b.py) · [All topics](../../README.md)

## What this example teaches

Combines attention, positionwise feed-forward layers, residual paths and normalization. Encoder, decoder and encoder-decoder variants use different masks and roles. The basic architecture does not require recurrence.

## Python code

```python
import numpy as np
X = np.array([[1., 2., 3.], [2., 4., 6.]])
F = np.maximum(0, X * 0.2)
y = X + F
normalized = (y - y.mean(axis=1, keepdims=True))
normalized /= np.sqrt(y.var(axis=1, keepdims=True) + 1e-5)
print("Normalized rows:", np.round(normalized, 3))
# Normalize features within each token, not across the batch.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `X = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])` | Build or transform the input feature matrix. |
| 3 | `F = np.maximum(0, X * 0.2)` | Assign F for the next step. |
| 4 | `y = X + F` | Define or generate the corresponding targets. |
| 5 | `normalized = y - y.mean(axis=1, keepdims=True)` | Center or normalize the feature activations. |
| 6 | `normalized /= np.sqrt(y.var(axis=1, keepdims=True) + 1e-05)` | Update normalized. |
| 7 | `print('Normalized rows:', np.round(normalized, 3))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Normalized rows: [[-1.225  0.     1.225]
 [-1.225  0.     1.225]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

A residual branch is added, then each token’s features are normalized within that row.

## Try it yourself

Contrast this operation with feature-wise batch normalization on the same input matrix.

## Important check

LayerNorm and BatchNorm use different axes and evaluation behaviour.

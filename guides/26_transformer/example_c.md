# TRANSFORMER ARCHITECTURE — Example C

**A miniature attention-plus-feedforward block**  
Book page: **110** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/26_transformer/example_c.py) · [All topics](../../README.md)

## What this example teaches

Combines attention, positionwise feed-forward layers, residual paths and normalization. Encoder, decoder and encoder-decoder variants use different masks and roles. The basic architecture does not require recurrence.

## Python code

```python
import numpy as np
X = np.array([[1., 0.], [0., 1.]])
scores = X @ X.T / np.sqrt(2)
A = np.exp(scores - scores.max(axis=1, keepdims=True))
A /= A.sum(axis=1, keepdims=True)
y = X + A @ X
norm = (y - y.mean(axis=1, keepdims=True)) / np.sqrt(y.var(axis=1, keepdims=True) + 1e-5)
F = np.maximum(0, norm @ np.eye(2)) @ np.eye(2)
print("Toy block output:", np.round(norm + F, 3))
# Fixed projections, one head; illustrative sublayers only.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `X = np.array([[1.0, 0.0], [0.0, 1.0]])` | Build or transform the input feature matrix. |
| 3 | `scores = X @ X.T / np.sqrt(2)` | Compute compatibility, class or preference scores. |
| 4 | `A = np.exp(scores - scores.max(axis=1, keepdims=True))` | Store normalized attention weights or transformed inputs. |
| 5 | `A /= A.sum(axis=1, keepdims=True)` | Update A. |
| 6 | `y = X + A @ X` | Define or generate the corresponding targets. |
| 7 | `norm = (` | Assign norm for the next step. |
| 8 | `    (y - y.mean(axis=1, keepdims=True)) / np.sqrt(y.var(axis=1,` | Supply arguments to the preceding operation. |
| 9 | `    keepdims=True) + 1e-05)` | Supply arguments to the preceding operation. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `F = np.maximum(0, norm @ np.eye(2)) @ np.eye(2)` | Assign F for the next step. |
| 12 | `print('Toy block output:', np.round(norm + F, 3))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Toy block output: [[ 2. -1.]
 [-1.  2.]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The toy block mixes token values with attention, then applies a residual, normalization and feed-forward branch.

## Try it yourself

Change one input token and inspect which output tokens change through attention.

## Important check

Fixed identity projections and one head do not constitute a complete trained Transformer.

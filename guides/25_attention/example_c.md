# ATTENTION MECHANISMS — Example C

**Attention pooling retrieves similar keys**  
Book page: **106** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/25_attention/example_c.py) · [All topics](../../README.md)

## What this example teaches

Compute a weighted combination of values, with weights determined by query-key compatibility. Self-attention uses representations from one sequence; cross-attention uses queries and keys/values from different sources.

## Python code

```python
import numpy as np
keys = np.array([0., 1., 2.]); values = np.array([0., 1., 4.])
query = 1.2
scores = -0.5 * (keys - query) ** 2
weights = np.exp(scores - scores.max()); weights /= weights.sum()
print("Similarity weights:", np.round(weights, 3))
print("Weighted estimate:", round(weights @ values, 4))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `keys = np.array([0.0, 1.0, 2.0])` | Assign keys for the next step. |
| 3 | `values = np.array([0.0, 1.0, 4.0])` | Assign values for the next step. |
| 4 | `query = 1.2` | Assign query for the next step. |
| 5 | `scores = -0.5 * (keys - query) ** 2` | Compute compatibility, class or preference scores. |
| 6 | `weights = np.exp(scores - scores.max())` | Assign weights for the next step. |
| 7 | `weights /= weights.sum()` | Update weights. |
| 8 | `print('Similarity weights:', np.round(weights, 3))` | Print the result shown in the output box. |
| 9 | `print('Weighted estimate:', round(weights @ values, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Similarity weights: [0.222 0.447 0.331]
Weighted estimate: 1.7714
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Nearby keys receive higher Gaussian-similarity weights, producing a smoothed estimate of the values.

## Try it yourself

Move the query from 0 to 2 and plot the weighted estimate.

## Important check

Attention pooling is not necessarily interpolation through every observed value.

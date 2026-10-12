# SOFTMAX REGRESSION — Example A

**Stable softmax probabilities**  
Book page: **72** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/17_softmax/example_a.py) · [All topics](../../README.md)

## What this example teaches

A linear multiclass classifier. One score is computed for each class; softmax converts scores into a probability vector summing to one. It is a neural network with an output layer and no hidden layer.

## Python code

```python
import numpy as np
scores = np.array([[2., 1., -1.]])
shifted = scores - scores.max(axis=1, keepdims=True)
p = np.exp(shifted)
p /= p.sum(axis=1, keepdims=True)
print("Probabilities:", np.round(p, 4))
print("Row sums:", p.sum(axis=1))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `scores = np.array([[2.0, 1.0, -1.0]])` | Compute compatibility, class or preference scores. |
| 3 | `shifted = scores - scores.max(axis=1, keepdims=True)` | Assign shifted for the next step. |
| 4 | `p = np.exp(shifted)` | Assign p for the next step. |
| 5 | `p /= p.sum(axis=1, keepdims=True)` | Update p. |
| 6 | `print('Probabilities:', np.round(p, 4))` | Print the result shown in the output box. |
| 7 | `print('Row sums:', p.sum(axis=1))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Probabilities: [[0.7054 0.2595 0.0351]]
Row sums: [1.]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Each row is a class probability vector. Subtracting the maximum score preserves softmax while reducing overflow risk.

## Try it yourself

Add the same constant to every score and verify that probabilities remain unchanged.

## Important check

Softmax normalization does not establish probability calibration.

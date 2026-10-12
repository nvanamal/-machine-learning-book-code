# ATTENTION MECHANISMS — Example A

**Scaled dot-product attention**  
Book page: **104** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/25_attention/example_a.py) · [All topics](../../README.md)

## What this example teaches

Compute a weighted combination of values, with weights determined by query-key compatibility. Self-attention uses representations from one sequence; cross-attention uses queries and keys/values from different sources.

## Python code

```python
import numpy as np
Q = np.array([[1., 0.]])
K = np.eye(2); V = np.array([[10.], [20.]])
scores = Q @ K.T / np.sqrt(2)
A = np.exp(scores - scores.max(axis=1, keepdims=True))
A /= A.sum(axis=1, keepdims=True)
print("Weights:", np.round(A, 4))
print("Context:", np.round(A @ V, 4))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `Q = np.array([[1.0, 0.0]])` | Initialize a state-by-action value table. |
| 3 | `K = np.eye(2)` | Assign K for the next step. |
| 4 | `V = np.array([[10.0], [20.0]])` | Assign V for the next step. |
| 5 | `scores = Q @ K.T / np.sqrt(2)` | Compute compatibility, class or preference scores. |
| 6 | `A = np.exp(scores - scores.max(axis=1, keepdims=True))` | Store normalized attention weights or transformed inputs. |
| 7 | `A /= A.sum(axis=1, keepdims=True)` | Update A. |
| 8 | `print('Weights:', np.round(A, 4))` | Print the result shown in the output box. |
| 9 | `print('Context:', np.round(A @ V, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Weights: [[0.6698 0.3302]]
Context: [[13.3024]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Compatibility scores are normalized across keys. The output is a weighted mixture of the values.

## Try it yourself

Swap the query to the other basis vector and observe which value receives more weight.

## Important check

Attention weights indicate routing in this calculation, not causal influence.

# ATTENTION MECHANISMS — Example B

**Causal self-attention mask**  
Book page: **105** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/25_attention/example_b.py) · [All topics](../../README.md)

## What this example teaches

Compute a weighted combination of values, with weights determined by query-key compatibility. Self-attention uses representations from one sequence; cross-attention uses queries and keys/values from different sources.

## Python code

```python
import numpy as np
X = np.eye(3)
scores = X @ X.T / np.sqrt(3)
scores[np.triu_indices(3, 1)] = -np.inf
A = np.exp(scores - scores.max(axis=1, keepdims=True))
A /= A.sum(axis=1, keepdims=True)
print("Causal weights:", np.round(A, 3))
print("Future-position weights are zero:", np.allclose(np.triu(A, 1), 0))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `X = np.eye(3)` | Build or transform the input feature matrix. |
| 3 | `scores = X @ X.T / np.sqrt(3)` | Compute compatibility, class or preference scores. |
| 4 | `scores[np.triu_indices(3, 1)] = -np.inf` | Assign scores[np.triu_indices(3, 1)] for the next step. |
| 5 | `A = np.exp(scores - scores.max(axis=1, keepdims=True))` | Store normalized attention weights or transformed inputs. |
| 6 | `A /= A.sum(axis=1, keepdims=True)` | Update A. |
| 7 | `print('Causal weights:', np.round(A, 3))` | Print the result shown in the output box. |
| 8 | `(` | Print the result shown in the output box. |
| 9 | `    print('Future-position weights are zero:', np.allclose(np.` | Supply arguments to the preceding operation. |
| 10 | `    triu(A, 1), 0))` | Supply arguments to the preceding operation. |
| 11 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
Causal weights: [[1.    0.    0.   ]
 [0.36  0.64  0.   ]
 [0.264 0.264 0.471]]
Future-position weights are zero: True
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The upper triangle is masked before softmax; future-position weights become zero and each row remains normalized.

## Try it yourself

Add a padding mask and check that all forbidden positions have zero probability.

## Important check

An entirely masked query row can produce invalid normalization; define a handling policy.

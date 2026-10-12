# RECOMMENDER SYSTEMS — Example B

**Latent user-item score matrix**  
Book page: **141** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/34_recommender/example_b.py) · [All topics](../../README.md)

## What this example teaches

Rank items for users using preferences, interactions or content. Content-based methods compare item attributes; collaborative methods use interaction patterns. Explicit ratings and implicit clicks require different assumptions.

## Python code

```python
import numpy as np
users = np.array([[1., 0.2], [0.1, 1.]])
items = np.array([[1., 0.], [0., 1.], [0.6, 0.6]])
scores = users @ items.T
print("User-item scores:", scores)
print("Best item per user:", scores.argmax(axis=1))
# Fixed embeddings illustrate scores; no factor training is claimed.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `users = np.array([[1.0, 0.2], [0.1, 1.0]])` | Assign users for the next step. |
| 3 | `items = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.6]])` | Assign items for the next step. |
| 4 | `scores = users @ items.T` | Compute compatibility, class or preference scores. |
| 5 | `print('User-item scores:', scores)` | Print the result shown in the output box. |
| 6 | `print('Best item per user:', scores.argmax(axis=1))` | Print the result shown in the output box. |

## Recorded output from the book

```text
User-item scores: [[1.   0.2  0.72]
 [0.1  1.   0.66]]
Best item per user: [0 1]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The user-item matrix contains dot products between fixed latent vectors. The largest score gives a candidate item.

## Try it yourself

Rescale one item embedding and observe how dot-product ranking differs from cosine ranking.

## Important check

These embeddings are fixed examples, not learned factors or evidence of recommendation quality.

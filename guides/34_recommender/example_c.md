# RECOMMENDER SYSTEMS — Example C

**Train factors only on observed ratings**  
Book page: **142** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/34_recommender/example_c.py) · [All topics](../../README.md)

## What this example teaches

Rank items for users using preferences, interactions or content. Content-based methods compare item attributes; collaborative methods use interaction patterns. Explicit ratings and implicit clicks require different assumptions.

## Python code

```python
import numpy as np
ratings = np.array([[5., np.nan, 1.], [4., 2., np.nan], [np.nan, 5., 2.]])
observed = np.argwhere(~np.isnan(ratings))
rng = np.random.default_rng(42)
U = rng.normal(0, 0.2, (3, 2)); V = rng.normal(0, 0.2, (3, 2))
for epoch in range(400):
    for i, j in observed:
        error = U[i] @ V[j] - ratings[i, j]
        old_u, old_v = U[i].copy(), V[j].copy()
        U[i] -= 0.01 * (error * old_v + 0.01 * old_u)
        V[j] -= 0.01 * (error * old_u + 0.01 * old_v)
print("Factor estimates:", np.round(U @ V.T, 2))
# Training fit only; reserve interactions for ranking evaluation.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `ratings = (` | Assign ratings for the next step. |
| 3 | `    np.array([[5.0, np.nan, 1.0], [4.0, 2.0, np.nan], [np.nan,` | Supply arguments to the preceding operation. |
| 4 | `    5.0, 2.0]])` | Supply arguments to the preceding operation. |
| 5 | `)` | Close the preceding multi-line operation. |
| 6 | `observed = np.argwhere(~np.isnan(ratings))` | Find rating entries that are actually observed. |
| 7 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 8 | `U = rng.normal(0, 0.2, (3, 2))` | Assign U for the next step. |
| 9 | `V = rng.normal(0, 0.2, (3, 2))` | Assign V for the next step. |
| 10 | `for epoch in range(400):` | Repeat episodes or steps of interaction. |
| 11 | `    for i, j in observed:` | Repeat episodes or steps of interaction. |
| 12 | `        error = U[i] @ V[j] - ratings[i, j]` | Compute a prediction residual or loss derivative. |
| 13 | `        old_u, old_v = (U[i].copy(), V[j].copy())` | Assign old_u, old_v for the next step. |
| 14 | `        U[i] -= 0.01 * (error * old_v + 0.01 * old_u)` | Update U[i]. |
| 15 | `        V[j] -= 0.01 * (error * old_u + 0.01 * old_v)` | Update V[j]. |
| 16 | `print('Factor estimates:', np.round(U @ V.T, 2))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Factor estimates: [[4.98 1.63 1.  ]
 [3.99 2.   1.06]
 [1.75 4.98 1.99]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Only observed ratings contribute to updates. The resulting matrix includes estimates for missing entries.

## Try it yourself

Reserve one observed rating before fitting and evaluate the held-out estimate.

## Important check

Never treat an unobserved rating as a true zero or report training fit as independent ranking performance.

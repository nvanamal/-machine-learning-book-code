# RECOMMENDER SYSTEMS — Example A

**Rank items by content similarity**  
Book page: **140** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/34_recommender/example_a.py) · [All topics](../../README.md)

## What this example teaches

Rank items for users using preferences, interactions or content. Content-based methods compare item attributes; collaborative methods use interaction patterns. Explicit ratings and implicit clicks require different assumptions.

## Python code

```python
import numpy as np
items = np.array([[1., 0.], [0., 1.], [0.8, 0.2]])
profile = np.array([1., 0.])
scores = items @ profile / (np.linalg.norm(items, axis=1) * np.linalg.norm(profile))
print("Content scores:", np.round(scores, 3))
print("Ranked item IDs:", np.argsort(-scores))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `items = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])` | Assign items for the next step. |
| 3 | `profile = np.array([1.0, 0.0])` | Assign profile for the next step. |
| 4 | `scores = (` | Compute compatibility, class or preference scores. |
| 5 | `    items @ profile / (np.linalg.norm(items, axis=1) * np.linalg` | Supply arguments to the preceding operation. |
| 6 | `    .norm(profile))` | Supply arguments to the preceding operation. |
| 7 | `)` | Close the preceding multi-line operation. |
| 8 | `print('Content scores:', np.round(scores, 3))` | Print the result shown in the output box. |
| 9 | `print('Ranked item IDs:', np.argsort(-scores))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Content scores: [1.   0.   0.97]
Ranked item IDs: [0 2 1]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Cosine similarity ranks item feature vectors by alignment with a user profile rather than by raw vector magnitude.

## Try it yourself

Change the user profile and compare the ranked item IDs.

## Important check

Filter unavailable or already-consumed items where appropriate; a raw ranking is not a complete recommender.

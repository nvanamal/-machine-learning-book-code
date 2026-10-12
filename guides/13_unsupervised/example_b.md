# UNSUPERVISED LEARNING — Example B

**Density clustering with DBSCAN**  
Book page: **57** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/13_unsupervised/example_b.py) · [All topics](../../README.md)

## What this example teaches

Uses inputs without supplied task labels to learn patterns or representations. Includes clustering, dimensionality reduction and density estimation. Objectives differ across algorithms; there is no universal unsupervised loss.

## Python code

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
X, _ = make_moons(n_samples=400, noise=0.05, random_state=42)
X = StandardScaler().fit_transform(X)
labels = DBSCAN(eps=0.25, min_samples=5).fit_predict(X)
print("Non-noise cluster count:", len(set(labels) - {-1}))
print("Noise points:", np.sum(labels == -1))
# -1 is DBSCAN noise; cluster IDs are arbitrary.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.datasets import make_moons` | Import make_moons. |
| 3 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 4 | `from sklearn.cluster import DBSCAN` | Import DBSCAN. |
| 5 | `X, _ = make_moons(n_samples=400, noise=0.05, random_state=42)` | Load or generate the example dataset. |
| 6 | `X = StandardScaler().fit_transform(X)` | Learn the transformation and apply it. |
| 7 | `labels = DBSCAN(eps=0.25, min_samples=5).fit_predict(X)` | Fit the clustering model and assign groups. |
| 8 | `print('Non-noise cluster count:', len(set(labels) - {-1}))` | Print the result shown in the output box. |
| 9 | `print('Noise points:', np.sum(labels == -1))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Non-noise cluster count: 2
Noise points: 1
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

DBSCAN groups dense regions and assigns label -1 to noise. Nonround groups can be captured.

## Try it yourself

Vary eps and min_samples; track group counts and noise fraction.

## Important check

In this method, -1 means noise, not an unlabeled supervised target.

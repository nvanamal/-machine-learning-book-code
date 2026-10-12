# SEMI-SUPERVISED LEARNING — Example B

**Graph-based label spreading**  
Book page: **65** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/15_semi/example_b.py) · [All topics](../../README.md)

## What this example teaches

Uses both labeled and unlabeled training examples for a task with target labels. Self-training fits a classifier on known labels, adds confident pseudo-labels, and repeats. Graph and consistency methods are other approaches.

## Python code

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
X, y = make_moons(n_samples=400, noise=0.15, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
yp = ytr.copy(); yp[np.random.default_rng(42).random(len(yp)) < 0.8] = -1
model = make_pipeline(StandardScaler(),
    LabelSpreading(kernel="knn", n_neighbors=7, max_iter=100))
model.fit(Xtr, yp)
print("Labeled test accuracy:", model.score(Xte, yte))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.datasets import make_moons` | Import make_moons. |
| 3 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 4 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 5 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 6 | `from sklearn.semi_supervised import LabelSpreading` | Import LabelSpreading. |
| 7 | `X, y = make_moons(n_samples=400, noise=0.15, random_state=42)` | Load or generate the example dataset. |
| 8 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 9 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `yp = ytr.copy()` | Retain known labels; -1 denotes unlabeled. |
| 12 | `yp[np.random.default_rng(42).random(len(yp)) < 0.8] = -1` | Hide selected labels by assigning the sentinel -1. |
| 13 | `model = (` | Chain preprocessing with the estimator. |
| 14 | `    make_pipeline(StandardScaler(), LabelSpreading(kernel='knn',` | Standardize using statistics fitted on training data. |
| 15 | `    n_neighbors=7, max_iter=100))` | Supply arguments to the preceding operation. |
| 16 | `)` | Close the preceding multi-line operation. |
| 17 | `model.fit(Xtr, yp)` | Learn parameters from the training data. |
| 18 | `print('Labeled test accuracy:', model.score(Xte, yte))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Labeled test accuracy: 0.97
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

LabelSpreading uses neighbourhood structure to spread supervision from a subset of labeled rows.

## Try it yourself

Change neighbour count and labeled fraction, evaluating on the same independent labeled test.

## Important check

The assumed similarity graph may connect points from different classes; check feature scaling.

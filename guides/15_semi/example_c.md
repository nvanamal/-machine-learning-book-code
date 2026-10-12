# SEMI-SUPERVISED LEARNING — Example C

**Compare with a labeled-only baseline**  
Book page: **66** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/15_semi/example_c.py) · [All topics](../../README.md)

## What this example teaches

Uses both labeled and unlabeled training examples for a task with target labels. Self-training fits a classifier on known labels, adds confident pseudo-labels, and repeats. Graph and consistency methods are other approaches.

## Python code

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
X, y = make_classification(n_samples=500, n_features=8, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
mask = np.random.default_rng(42).random(len(ytr)) < 0.2
yp = np.where(mask, ytr, -1)
def base(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
plain = base().fit(Xtr[mask], ytr[mask])
semi = SelfTrainingClassifier(base(), threshold=0.9).fit(Xtr, yp)
print("Labeled only:", plain.score(Xte, yte))
print("Self-training:", semi.score(Xte, yte))
# This demonstrates comparison; semi-supervised gains are not guaranteed.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.datasets import make_classification` | Import a labeled synthetic-data generator. |
| 3 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 4 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 5 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 6 | `from sklearn.linear_model import LogisticRegression` | Import LogisticRegression. |
| 7 | `from sklearn.semi_supervised import SelfTrainingClassifier` | Import SelfTrainingClassifier. |
| 8 | `(X, y) = (` | Generate inputs and class labels. |
| 9 | `    make_classification(n_samples=500, n_features=8,` | Set feature count and useful-signal dimensions. |
| 10 | `    random_state=42)` | Fix the seed and other options for repeatability. |
| 11 | `)` | Close the preceding multi-line operation. |
| 12 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 13 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 14 | `)` | Close the preceding multi-line operation. |
| 15 | `mask = np.random.default_rng(42).random(len(ytr)) < 0.2` | Select retained activations or labeled examples. |
| 16 | `yp = np.where(mask, ytr, -1)` | Retain known labels; -1 denotes unlabeled. |
| 17 | `` |  |
| 18 | `def base():` | Define a reusable helper function. |
| 19 | `    return (` | Return the helper result to its caller. |
| 20 | `        make_pipeline(StandardScaler(), LogisticRegression(` | Choose the estimator and its tuning options. |
| 21 | `        max_iter=1000))` | Supply arguments to the preceding operation. |
| 22 | `    )` | Close the preceding multi-line operation. |
| 23 | `plain = base().fit(Xtr[mask], ytr[mask])` | Fit model parameters using training inputs. |
| 24 | `semi = (` | Fit model parameters using training inputs. |
| 25 | `    SelfTrainingClassifier(base(), threshold=0.9).fit(Xtr, yp)` | Choose the estimator and its tuning options. |
| 26 | `)` | Close the preceding multi-line operation. |
| 27 | `print('Labeled only:', plain.score(Xte, yte))` | Print the result shown in the output box. |
| 28 | `print('Self-training:', semi.score(Xte, yte))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Labeled only: 0.88
Self-training: 0.864
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The two scores compare labeled-only fitting and self-training under the same held-out target set.

## Try it yourself

Repeat the labeled-mask selection with several seeds to examine sensitivity.

## Important check

Unlabeled data do not guarantee an improvement; report the labeled-only baseline.

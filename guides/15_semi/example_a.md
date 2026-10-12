# SEMI-SUPERVISED LEARNING — Example A

**Confidence-based self-training**  
Book page: **64** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/15_semi/example_a.py) · [All topics](../../README.md)

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
from sklearn.metrics import balanced_accuracy_score
X, y = make_classification(n_samples=600,
    n_features=8, n_informative=4, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, stratify=y, random_state=42)
rng = np.random.default_rng(42)
y_partial = ytr.copy()
y_partial[rng.random(len(ytr)) < 0.8] = -1
base = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=1000))
model = SelfTrainingClassifier(base, threshold=0.9)
model.fit(Xtr, y_partial)
print(balanced_accuracy_score(yte, model.predict(Xte)))
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
| 8 | `from sklearn.metrics import balanced_accuracy_score` | Import class-balanced accuracy. |
| 9 | `(X, y) = (` | Generate inputs and class labels. |
| 10 | `    make_classification(n_samples=600, n_features=8,` | Set feature count and useful-signal dimensions. |
| 11 | `    n_informative=4, random_state=42)` | Set feature count and useful-signal dimensions. |
| 12 | `)` | Close the preceding multi-line operation. |
| 13 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 14 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 15 | `)` | Close the preceding multi-line operation. |
| 16 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 17 | `y_partial = ytr.copy()` | Retain known labels; -1 denotes unlabeled. |
| 18 | `y_partial[rng.random(len(ytr)) < 0.8] = -1` | Hide selected labels by assigning the sentinel -1. |
| 19 | `base = (` | Chain preprocessing with the estimator. |
| 20 | `    make_pipeline(StandardScaler(), LogisticRegression(max_iter=` | Choose the estimator and its tuning options. |
| 21 | `    1000))` | Supply arguments to the preceding operation. |
| 22 | `)` | Close the preceding multi-line operation. |
| 23 | `model = SelfTrainingClassifier(base, threshold=0.9)` | Create or fit the selected estimator. |
| 24 | `model.fit(Xtr, y_partial)` | Learn parameters from the training data. |
| 25 | `print(balanced_accuracy_score(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
0.7333333333333334
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Known labels seed a classifier; confident predictions add pseudo-labels from the training-only pool.

## Try it yourself

Compare thresholds 0.75 and 0.9 and count newly labeled training rows.

## Important check

Confident errors can reinforce themselves, especially if probabilities are poorly calibrated.

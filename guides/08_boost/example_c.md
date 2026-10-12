# GRADIENT BOOSTING — Example C

**Histogram boosting with missing inputs**  
Book page: **38** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/08_boost/example_c.py) · [All topics](../../README.md)

## What this example teaches

Adds weak learners in sequence to improve an objective. Each new learner fits the negative gradient of the loss. For squared-error regression this is a residual; classification uses loss-specific gradients.

## Python code

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
X, y = make_classification(n_samples=450, n_features=8,
    n_informative=4, random_state=42)
X[np.random.default_rng(42).random(X.shape) < 0.05] = np.nan
Xtr, Xte, ytr, yte = train_test_split(X, y,
    stratify=y, random_state=42)
model = HistGradientBoostingClassifier(max_iter=60, random_state=42)
model.fit(Xtr, ytr)
print(balanced_accuracy_score(yte, model.predict(Xte)))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.datasets import make_classification` | Import a labeled synthetic-data generator. |
| 3 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 4 | `from sklearn.ensemble import HistGradientBoostingClassifier` | Import HistGradientBoostingClassifier. |
| 5 | `from sklearn.metrics import balanced_accuracy_score` | Import class-balanced accuracy. |
| 6 | `(X, y) = (` | Generate inputs and class labels. |
| 7 | `    make_classification(n_samples=450, n_features=8,` | Set feature count and useful-signal dimensions. |
| 8 | `    n_informative=4, random_state=42)` | Set feature count and useful-signal dimensions. |
| 9 | `)` | Close the preceding multi-line operation. |
| 10 | `X[np.random.default_rng(42).random(X.shape) < 0.05] = np.nan` | Insert missing values at selected feature positions. |
| 11 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 12 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 13 | `)` | Close the preceding multi-line operation. |
| 14 | `model = (` | Create or fit the selected estimator. |
| 15 | `    HistGradientBoostingClassifier(max_iter=60, random_state=42)` | Choose the estimator and its tuning options. |
| 16 | `)` | Close the preceding multi-line operation. |
| 17 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 18 | `print(balanced_accuracy_score(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
0.7880639097744361
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Histogram boosting accepts the inserted NaNs. The result tests the fitted classifier on separate rows.

## Try it yourself

Compare native missing-value handling with a median-imputation pipeline on the same split.

## Important check

Native NaN support here does not imply that standard GradientBoostingClassifier accepts missing values.

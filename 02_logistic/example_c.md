# LOGISTIC REGRESSION — Example C

**Training-only CV for binary classification**  
Book page: **14** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/02_logistic/example_c.py) · [All topics](../../README.md)

## What this example teaches

A classification model that maps a linear score to a probability. Binary logistic regression uses a sigmoid; multiclass models commonly use softmax. The name refers to a regression of log-odds, not a numeric target.

## Python code

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
X, y = make_classification(n_samples=450, n_features=8,
    n_informative=4, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y,
    stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(model, Xtr, ytr, cv=3,
    scoring="balanced_accuracy")
print("Training CV:", scores.mean())
model.fit(Xtr, ytr)
print("Test:", balanced_accuracy_score(yte, model.predict(Xte)))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_classification` | Import a labeled synthetic-data generator. |
| 2 | `from sklearn.model_selection import (` | Import train_test_split, cross_val_score. |
| 3 | `    train_test_split,` | Supply arguments to the preceding operation. |
| 4 | `    cross_val_score,` | Supply arguments to the preceding operation. |
| 5 | `)` | Close the preceding multi-line operation. |
| 6 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 7 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 8 | `from sklearn.linear_model import LogisticRegression` | Import LogisticRegression. |
| 9 | `from sklearn.metrics import balanced_accuracy_score` | Import class-balanced accuracy. |
| 10 | `(X, y) = (` | Generate inputs and class labels. |
| 11 | `    make_classification(n_samples=450, n_features=8,` | Set feature count and useful-signal dimensions. |
| 12 | `    n_informative=4, random_state=42)` | Set feature count and useful-signal dimensions. |
| 13 | `)` | Close the preceding multi-line operation. |
| 14 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 15 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 16 | `)` | Close the preceding multi-line operation. |
| 17 | `model = (` | Chain preprocessing with the estimator. |
| 18 | `    make_pipeline(StandardScaler(), LogisticRegression(max_iter=` | Choose the estimator and its tuning options. |
| 19 | `    1000))` | Supply arguments to the preceding operation. |
| 20 | `)` | Close the preceding multi-line operation. |
| 21 | `scores = (` | Compute compatibility, class or preference scores. |
| 22 | `    cross_val_score(model, Xtr, ytr, cv=3, scoring=` | Choose the metric used for validation. |
| 23 | `    'balanced_accuracy')` | Supply arguments to the preceding operation. |
| 24 | `)` | Close the preceding multi-line operation. |
| 25 | `print('Training CV:', scores.mean())` | Print the result shown in the output box. |
| 26 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 27 | `print('Test:', balanced_accuracy_score(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Training CV: 0.7124060150375939
Test: 0.681704260651629
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Cross-validation uses Xtr only; the final test result is computed after fitting the same pipeline on all training rows.

## Try it yourself

Use StratifiedKFold with shuffling and a fixed seed; compare the fold scores.

## Important check

Tuning many variants on the held-out test would turn it into a validation set.

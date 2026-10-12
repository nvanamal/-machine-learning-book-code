# DECISION TREE — Example C

**Training-only classification CV**  
Book page: **18** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/03_tree/example_c.py) · [All topics](../../README.md)

## What this example teaches

Uses successive feature tests to divide data into regions. Internal nodes are tests, branches are outcomes, and leaves contain a class prediction or numeric value. A shallow tree can be displayed as human-readable rules.

## Python code

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
X, y = make_classification(n_samples=450, n_features=8,
    n_informative=4, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y,
    stratify=y, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
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
| 8 | `from sklearn.tree import DecisionTreeClassifier` | Import DecisionTreeClassifier. |
| 9 | `from sklearn.metrics import balanced_accuracy_score` | Import class-balanced accuracy. |
| 10 | `(X, y) = (` | Generate inputs and class labels. |
| 11 | `    make_classification(n_samples=450, n_features=8,` | Set feature count and useful-signal dimensions. |
| 12 | `    n_informative=4, random_state=42)` | Set feature count and useful-signal dimensions. |
| 13 | `)` | Close the preceding multi-line operation. |
| 14 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 15 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 16 | `)` | Close the preceding multi-line operation. |
| 17 | `model = DecisionTreeClassifier(max_depth=4, random_state=42)` | Create or fit the selected estimator. |
| 18 | `scores = (` | Compute compatibility, class or preference scores. |
| 19 | `    cross_val_score(model, Xtr, ytr, cv=3, scoring=` | Choose the metric used for validation. |
| 20 | `    'balanced_accuracy')` | Supply arguments to the preceding operation. |
| 21 | `)` | Close the preceding multi-line operation. |
| 22 | `print('Training CV:', scores.mean())` | Print the result shown in the output box. |
| 23 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 24 | `print('Test:', balanced_accuracy_score(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Training CV: 0.7895258980785297
Test: 0.7534461152882206
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Training-only CV estimates how the depth-limited classifier behaves across different training subsets.

## Try it yourself

Add a DummyClassifier baseline and use the same folds for both models.

## Important check

Do not attribute an improvement to the model if the compared methods used different data splits.

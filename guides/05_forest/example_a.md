# RANDOM FOREST — Example A

**Basic fit and held-out prediction**  
Book page: **24** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/05_forest/example_a.py) · [All topics](../../README.md)

## What this example teaches

Combines many randomized decision trees. Each tree usually sees a bootstrap sample and considers a random subset of features at a split. Averaging reduces variability compared with a single tree.

## Python code

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
X, y = make_classification(n_samples=600,
    n_features=8, n_informative=4, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(n_estimators=150,
        min_samples_leaf=3, random_state=42))
# Train on training data; evaluate on unseen rows
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("Balanced accuracy:",
      balanced_accuracy_score(yte, pred))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_classification` | Import a labeled synthetic-data generator. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 4 | `from sklearn.impute import SimpleImputer` | Import missing-value imputation. |
| 5 | `from sklearn.metrics import balanced_accuracy_score` | Import class-balanced accuracy. |
| 6 | `from sklearn.ensemble import RandomForestClassifier` | Import RandomForestClassifier. |
| 7 | `(X, y) = (` | Generate inputs and class labels. |
| 8 | `    make_classification(n_samples=600, n_features=8,` | Set feature count and useful-signal dimensions. |
| 9 | `    n_informative=4, random_state=42)` | Set feature count and useful-signal dimensions. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 12 | `    train_test_split(X, y, test_size=0.2, stratify=y,` | Preserve class proportions in the split. |
| 13 | `    random_state=42)` | Fix the seed and other options for repeatability. |
| 14 | `)` | Close the preceding multi-line operation. |
| 15 | `model = (` | Chain preprocessing with the estimator. |
| 16 | `    make_pipeline(SimpleImputer(strategy='median'),` | Replace missing inputs using training medians. |
| 17 | `    RandomForestClassifier(n_estimators=150, min_samples_leaf=3` | Choose the estimator and its tuning options. |
| 18 | `    , random_state=42))` | Fix the seed and other options for repeatability. |
| 19 | `)` | Close the preceding multi-line operation. |
| 20 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 21 | `pred = model.predict(Xte)` | Predict from data the model has not fitted. |
| 22 | `print('Balanced accuracy:', balanced_accuracy_score(yte, pred))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Balanced accuracy: 0.8833333333333333
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The forest averages randomized trees. Balanced accuracy summarizes class recall on the held-out partition.

## Try it yourself

Compare min_samples_leaf=1, 3 and 8 on training folds, keeping the seed fixed.

## Important check

More trees reduce Monte Carlo variability, but do not repair label errors or leakage.

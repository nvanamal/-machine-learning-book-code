# DECISION TREE — Example A

**Basic fit and held-out prediction**  
Book page: **16** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/03_tree/example_a.py) · [All topics](../../README.md)

## What this example teaches

Uses successive feature tests to divide data into regions. Internal nodes are tests, branches are outcomes, and leaves contain a class prediction or numeric value. A shallow tree can be displayed as human-readable rules.

## Python code

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
X, y = make_classification(n_samples=600,
    n_features=8, n_informative=4, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
model = make_pipeline(
    SimpleImputer(strategy="median"),
    DecisionTreeClassifier(max_depth=5,
        min_samples_leaf=8, random_state=42))
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
| 6 | `from sklearn.tree import DecisionTreeClassifier` | Import DecisionTreeClassifier. |
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
| 17 | `    DecisionTreeClassifier(max_depth=5, min_samples_leaf=8,` | Choose the estimator and its tuning options. |
| 18 | `    random_state=42))` | Fix the seed and other options for repeatability. |
| 19 | `)` | Close the preceding multi-line operation. |
| 20 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 21 | `pred = model.predict(Xte)` | Predict from data the model has not fitted. |
| 22 | `print('Balanced accuracy:', balanced_accuracy_score(yte, pred))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Balanced accuracy: 0.7416666666666667
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

A depth-limited tree learns feature tests and produces a class at each leaf. Leaf-size limits reduce tiny, unstable regions.

## Try it yourself

Compare max_depth values 2, 4 and 8 using the training partition.

## Important check

A larger tree can improve training accuracy while harming performance on unseen rows.

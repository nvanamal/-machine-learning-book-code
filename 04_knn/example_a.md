# K-NEAREST NEIGHBOURS (KNN) — Example A

**Basic fit and held-out prediction**  
Book page: **20** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/04_knn/example_a.py) · [All topics](../../README.md)

## What this example teaches

Predicts from the k nearest stored training examples. Classification uses voting; regression uses an average. Distance weighting gives closer examples more influence. Fitting stores observations and may build a search structure.

## Python code

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
X, y = make_classification(n_samples=600,
    n_features=8, n_informative=4, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=7,
        weights="distance"))
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
| 5 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 6 | `from sklearn.metrics import balanced_accuracy_score` | Import class-balanced accuracy. |
| 7 | `from sklearn.neighbors import KNeighborsClassifier` | Import KNeighborsClassifier. |
| 8 | `(X, y) = (` | Generate inputs and class labels. |
| 9 | `    make_classification(n_samples=600, n_features=8,` | Set feature count and useful-signal dimensions. |
| 10 | `    n_informative=4, random_state=42)` | Set feature count and useful-signal dimensions. |
| 11 | `)` | Close the preceding multi-line operation. |
| 12 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 13 | `    train_test_split(X, y, test_size=0.2, stratify=y,` | Preserve class proportions in the split. |
| 14 | `    random_state=42)` | Fix the seed and other options for repeatability. |
| 15 | `)` | Close the preceding multi-line operation. |
| 16 | `model = (` | Chain preprocessing with the estimator. |
| 17 | `    make_pipeline(SimpleImputer(strategy='median'),` | Replace missing inputs using training medians. |
| 18 | `    StandardScaler(), KNeighborsClassifier(n_neighbors=7,` | Choose the estimator and its tuning options. |
| 19 | `    weights='distance'))` | Set distance weighting for neighbour predictions. |
| 20 | `)` | Close the preceding multi-line operation. |
| 21 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 22 | `pred = model.predict(Xte)` | Predict from data the model has not fitted. |
| 23 | `print('Balanced accuracy:', balanced_accuracy_score(yte, pred))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Balanced accuracy: 0.8333333333333333
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Distance-weighted voting gives nearby examples more influence. The scaler is learned from training data inside the pipeline.

## Try it yourself

Compare uniform and distance weighting at k=3, 7 and 15 using training-only folds.

## Important check

Unscaled feature units can determine the neighbours even when those units are irrelevant to the task.

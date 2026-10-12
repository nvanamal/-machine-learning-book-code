# SUPPORT VECTOR MACHINE (SVM) — Example C

**Linear SVM on multiclass inputs**  
Book page: **34** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/07_svm/example_c.py) · [All topics](../../README.md)

## What this example teaches

Balances a large separation margin against violations. A kernel can produce nonlinear boundaries. SVC classifies, SVR predicts numeric outcomes, and OneClassSVM supports novelty detection.

## Python code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y,
    stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LinearSVC(C=1, max_iter=5000))
model.fit(Xtr, ytr)
print(classification_report(yte, model.predict(Xte), zero_division=0))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import load_iris` | Import load_iris. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 4 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 5 | `from sklearn.svm import LinearSVC` | Import LinearSVC. |
| 6 | `from sklearn.metrics import classification_report` | Import per-class evaluation metrics. |
| 7 | `X, y = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 8 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 9 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `model = (` | Chain preprocessing with the estimator. |
| 12 | `    make_pipeline(StandardScaler(), LinearSVC(C=1, max_iter=5000` | Standardize using statistics fitted on training data. |
| 13 | `    ))` | Supply arguments to the preceding operation. |
| 14 | `)` | Close the preceding multi-line operation. |
| 15 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 16 | `(` | Print the result shown in the output box. |
| 17 | `    print(classification_report(yte, model.predict(Xte),` | Supply arguments to the preceding operation. |
| 18 | `    zero_division=0))` | Supply arguments to the preceding operation. |
| 19 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
precision    recall  f1-score   support
           0       1.00      0.92      0.96        12
           1       0.83      0.77      0.80        13
           2       0.80      0.92      0.86        13
    accuracy                           0.87        38
   macro avg       0.88      0.87      0.87        38
weighted avg       0.87      0.87      0.87        38
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

LinearSVC uses a linear decision function and reports multiclass performance without an RBF kernel.

## Try it yourself

Compare LinearSVC with logistic regression on the same scaled features and split.

## Important check

LinearSVC does not directly return class probabilities. A decision score is not a calibrated probability.

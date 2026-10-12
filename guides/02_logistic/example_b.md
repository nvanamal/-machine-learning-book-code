# LOGISTIC REGRESSION — Example B

**Multiclass flower classification**  
Book page: **13** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/02_logistic/example_b.py) · [All topics](../../README.md)

## What this example teaches

A classification model that maps a linear score to a probability. Binary logistic regression uses a sigmoid; multiclass models commonly use softmax. The name refers to a regression of log-odds, not a numeric target.

## Python code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y,
    stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
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
| 5 | `from sklearn.linear_model import LogisticRegression` | Import LogisticRegression. |
| 6 | `from sklearn.metrics import classification_report` | Import per-class evaluation metrics. |
| 7 | `X, y = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 8 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 9 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `model = (` | Chain preprocessing with the estimator. |
| 12 | `    make_pipeline(StandardScaler(), LogisticRegression(max_iter=` | Choose the estimator and its tuning options. |
| 13 | `    1000))` | Supply arguments to the preceding operation. |
| 14 | `)` | Close the preceding multi-line operation. |
| 15 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 16 | `(` | Print the result shown in the output box. |
| 17 | `    print(classification_report(yte, model.predict(Xte),` | Supply arguments to the preceding operation. |
| 18 | `    zero_division=0))` | Supply arguments to the preceding operation. |
| 19 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
precision    recall  f1-score   support
           0       1.00      1.00      1.00        12
           1       0.86      0.92      0.89        13
           2       0.92      0.85      0.88        13
    accuracy                           0.92        38
   macro avg       0.92      0.92      0.92        38
weighted avg       0.92      0.92      0.92        38
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The report lists precision, recall, F1 and support for each flower class. Multiclass prediction is supported.

## Try it yourself

Print predict_proba on three held-out rows and verify that each probability row sums to one.

## Important check

A probability vector summing to one does not establish calibration. Class probabilities need separate checks.

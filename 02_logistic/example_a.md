# LOGISTIC REGRESSION — Example A

**Basic fit and held-out prediction**  
Book page: **12** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/02_logistic/example_a.py) · [All topics](../../README.md)

## What this example teaches

A classification model that maps a linear score to a probability. Binary logistic regression uses a sigmoid; multiclass models commonly use softmax. The name refers to a regression of log-odds, not a numeric target.

## Python code

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
X, y = make_classification(n_samples=600,
    n_features=8, n_informative=4, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(C=1, max_iter=1000))
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
| 7 | `from sklearn.linear_model import LogisticRegression` | Import LogisticRegression. |
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
| 18 | `    StandardScaler(), LogisticRegression(C=1, max_iter=1000))` | Choose the estimator and its tuning options. |
| 19 | `)` | Close the preceding multi-line operation. |
| 20 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 21 | `pred = model.predict(Xte)` | Predict from data the model has not fitted. |
| 22 | `print('Balanced accuracy:', balanced_accuracy_score(yte, pred))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Balanced accuracy: 0.6916666666666667
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Balanced accuracy averages recall across classes. A large majority class cannot dominate it as easily as ordinary accuracy.

## Try it yourself

Vary C on training folds and compare balanced accuracy with log loss.

## Important check

C is inverse regularization strength: smaller C means a stronger penalty, not a more flexible model.

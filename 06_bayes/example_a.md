# NAIVE BAYES — Example A

**Basic fit and held-out prediction**  
Book page: **28** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/06_bayes/example_a.py) · [All topics](../../README.md)

## What this example teaches

Uses Bayes’ theorem with a conditional-independence assumption: features are independent given the class. Gaussian, Multinomial and Bernoulli variants model different feature distributions.

## Python code

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
X, y = make_classification(n_samples=600,
    n_features=8, n_informative=4, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
model = make_pipeline(
    SimpleImputer(strategy="median"),
    GaussianNB(var_smoothing=1e-9))
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
| 6 | `from sklearn.naive_bayes import GaussianNB` | Import GaussianNB. |
| 7 | `(X, y) = (` | Generate inputs and class labels. |
| 8 | `    make_classification(n_samples=600, n_features=8,` | Set feature count and useful-signal dimensions. |
| 9 | `    n_informative=4, random_state=42)` | Set feature count and useful-signal dimensions. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 12 | `    train_test_split(X, y, test_size=0.2, stratify=y,` | Preserve class proportions in the split. |
| 13 | `    random_state=42)` | Fix the seed and other options for repeatability. |
| 14 | `)` | Close the preceding multi-line operation. |
| 15 | `model = (` | Chain preprocessing with the estimator. |
| 16 | `    make_pipeline(SimpleImputer(strategy='median'), GaussianNB(` | Replace missing inputs using training medians. |
| 17 | `    var_smoothing=1e-09))` | Supply arguments to the preceding operation. |
| 18 | `)` | Close the preceding multi-line operation. |
| 19 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 20 | `pred = model.predict(Xte)` | Predict from data the model has not fitted. |
| 21 | `print('Balanced accuracy:', balanced_accuracy_score(yte, pred))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Balanced accuracy: 0.7416666666666667
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

GaussianNB estimates class-specific feature distributions. Balanced accuracy measures classification, not probability calibration.

## Try it yourself

Change var_smoothing and inspect both class predictions and predict_proba.

## Important check

The independence assumption is conditional on the class; correlated features can distort probabilities.

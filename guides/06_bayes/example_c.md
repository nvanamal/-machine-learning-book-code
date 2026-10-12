# NAIVE BAYES — Example C

**Bernoulli NB for binary features**  
Book page: **30** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/06_bayes/example_c.py) · [All topics](../../README.md)

## What this example teaches

Uses Bayes’ theorem with a conditional-independence assumption: features are independent given the class. Gaussian, Multinomial and Bernoulli variants model different feature distributions.

## Python code

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import balanced_accuracy_score
X, y = make_classification(n_samples=450, n_features=8,
    n_informative=4, random_state=42)
X = (X > 0).astype(int)  # fixed threshold, binary inputs
Xtr, Xte, ytr, yte = train_test_split(X, y,
    stratify=y, random_state=42)
model = BernoulliNB(alpha=1).fit(Xtr, ytr)
print(balanced_accuracy_score(yte, model.predict(Xte)))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `from sklearn.datasets import make_classification` | Import a labeled synthetic-data generator. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.naive_bayes import BernoulliNB` | Import BernoulliNB. |
| 4 | `from sklearn.metrics import balanced_accuracy_score` | Import class-balanced accuracy. |
| 5 | `(X, y) = (` | Generate inputs and class labels. |
| 6 | `    make_classification(n_samples=450, n_features=8,` | Set feature count and useful-signal dimensions. |
| 7 | `    n_informative=4, random_state=42)` | Set feature count and useful-signal dimensions. |
| 8 | `)` | Close the preceding multi-line operation. |
| 9 | `X = (X > 0).astype(int)` | Convert thresholded features to binary integers. |
| 10 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 11 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 12 | `)` | Close the preceding multi-line operation. |
| 13 | `model = BernoulliNB(alpha=1).fit(Xtr, ytr)` | Fit model parameters using training inputs. |
| 14 | `print(balanced_accuracy_score(yte, model.predict(Xte)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
0.762374686716792
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Thresholding creates binary features. BernoulliNB models their presence or absence within each class.

## Try it yourself

Try a fixed threshold of 0.5 instead of 0 and compare training-only validation scores.

## Important check

If a threshold is estimated from data, estimate it inside training folds rather than on the complete dataset.

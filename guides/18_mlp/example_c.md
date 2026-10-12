# MULTILAYER PERCEPTRON (MLP) — Example C

**Fit a small neural classifier**  
Book page: **78** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/18_mlp/example_c.py) · [All topics](../../README.md)

## What this example teaches

A feed-forward neural network with fully connected hidden layers and nonlinear activations. Training adjusts weights by gradients propagated backward through the network.

## Python code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y,
    stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(12,), solver="lbfgs", max_iter=3000, random_state=42))
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
| 5 | `from sklearn.neural_network import MLPClassifier` | Import MLPClassifier. |
| 6 | `from sklearn.metrics import classification_report` | Import per-class evaluation metrics. |
| 7 | `X, y = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 8 | `(Xtr, Xte, ytr, yte) = (` | Reserve unseen data for testing. |
| 9 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `model = (` | Chain preprocessing with the estimator. |
| 12 | `    make_pipeline(StandardScaler(), MLPClassifier(` | Choose the estimator and its tuning options. |
| 13 | `    hidden_layer_sizes=(12,), solver='lbfgs', max_iter=3000,` | Supply arguments to the preceding operation. |
| 14 | `    random_state=42))` | Fix the seed and other options for repeatability. |
| 15 | `)` | Close the preceding multi-line operation. |
| 16 | `model.fit(Xtr, ytr)` | Learn parameters from the training data. |
| 17 | `(` | Print the result shown in the output box. |
| 18 | `    print(classification_report(yte, model.predict(Xte),` | Supply arguments to the preceding operation. |
| 19 | `    zero_division=0))` | Supply arguments to the preceding operation. |
| 20 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
precision    recall  f1-score   support
           0       1.00      0.92      0.96        12
           1       0.75      0.92      0.83        13
           2       0.91      0.77      0.83        13
    accuracy                           0.87        38
   macro avg       0.89      0.87      0.87        38
weighted avg       0.88      0.87      0.87        38
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The classifier learns a small hidden layer. The report assesses held-out flower classes after standardization.

## Try it yourself

Compare hidden widths 4, 12 and 24 using training-only validation.

## Important check

Convergence of optimization does not prove generalization; compare a simpler baseline.

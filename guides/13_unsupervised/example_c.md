# UNSUPERVISED LEARNING — Example C

**Unlabeled anomaly screening**  
Book page: **58** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/13_unsupervised/example_c.py) · [All topics](../../README.md)

## What this example teaches

Uses inputs without supplied task labels to learn patterns or representations. Includes clustering, dimensionality reduction and density estimation. Objectives differ across algorithms; there is no universal unsupervised loss.

## Python code

```python
import numpy as np
from sklearn.ensemble import IsolationForest
rng = np.random.default_rng(42)
Xtr = rng.normal(size=(400, 2))
Xquery = np.array([[0., 0.], [6., 6.]])
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(Xtr)
print("Predictions (+1 usual, -1 unusual):", model.predict(Xquery))
print("Decision scores:", model.decision_function(Xquery))
# Unusual is not the same as a confirmed fault.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.ensemble import IsolationForest` | Import IsolationForest. |
| 3 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 4 | `Xtr = rng.normal(size=(400, 2))` | Assign Xtr for the next step. |
| 5 | `Xquery = np.array([[0.0, 0.0], [6.0, 6.0]])` | Define unseen samples for anomaly screening. |
| 6 | `model = IsolationForest(contamination=0.05, random_state=42)` | Create or fit the selected estimator. |
| 7 | `model.fit(Xtr)` | Learn parameters from the training data. |
| 8 | `(` | Print the result shown in the output box. |
| 9 | `    print('Predictions (+1 usual, -1 unusual):', model.predict(` | Supply arguments to the preceding operation. |
| 10 | `    Xquery))` | Supply arguments to the preceding operation. |
| 11 | `)` | Close the preceding multi-line operation. |
| 12 | `print('Decision scores:', model.decision_function(Xquery))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Predictions (+1 usual, -1 unusual): [ 1 -1]
Decision scores: [ 0.18831976 -0.14987145]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

IsolationForest scores unusual inputs relative to the fitted distribution; +1 and -1 identify typical and anomalous cases.

## Try it yourself

Move the query gradually away from the training cloud and inspect decision scores.

## Important check

An anomalous score flags review candidates. It does not confirm failure, fraud or collision.

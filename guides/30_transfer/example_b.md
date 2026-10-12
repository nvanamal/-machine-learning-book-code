# TRANSFER LEARNING / FINE-TUNING — Example B

**Reuse source hidden features for a new label task**  
Book page: **125** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/30_transfer/example_b.py) · [All topics](../../README.md)

## What this example teaches

Reuse parameters learned on a source task for a target task. Feature extraction freezes a pretrained representation; fine-tuning updates some or all transferred parameters. A new task head often replaces the original output layer.

## Python code

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
A, B, ya, yb = train_test_split(X, y, stratify=y, random_state=42)
scale = StandardScaler().fit(A); A, B = scale.transform(A), scale.transform(B)
source = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=3000, random_state=42).fit(A, ya)
def features(X): return np.maximum(0, X @ source.coefs_[0] + source.intercepts_[0])
head = LogisticRegression(max_iter=1000).fit(features(A), (ya == 2).astype(int))
print("New binary-task test accuracy:", head.score(features(B), (yb == 2).astype(int)))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.datasets import load_iris` | Import load_iris. |
| 3 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 4 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 5 | `from sklearn.neural_network import MLPClassifier` | Import MLPClassifier. |
| 6 | `from sklearn.linear_model import LogisticRegression` | Import LogisticRegression. |
| 7 | `X, y = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 8 | `(A, B, ya, yb) = (` | Reserve unseen data for testing. |
| 9 | `    train_test_split(X, y, stratify=y, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `scale = StandardScaler().fit(A)` | Fit model parameters using training inputs. |
| 12 | `A, B = (scale.transform(A), scale.transform(B))` | Store transformed training or test features. |
| 13 | `source = (` | Fit model parameters using training inputs. |
| 14 | `    MLPClassifier(hidden_layer_sizes=(8,), solver='lbfgs',` | Choose the estimator and its tuning options. |
| 15 | `    max_iter=3000, random_state=42).fit(A, ya)` | Fix the seed and other options for repeatability. |
| 16 | `)` | Close the preceding multi-line operation. |
| 17 | `` |  |
| 18 | `def features(X):` | Define a reusable helper function. |
| 19 | `    return (` | Return the helper result to its caller. |
| 20 | `        np.maximum(0, X @ source.coefs_[0] + source.intercepts_[` | Supply arguments to the preceding operation. |
| 21 | `        0])` | Supply arguments to the preceding operation. |
| 22 | `    )` | Close the preceding multi-line operation. |
| 23 | `head = (` | Fit a task-specific output head. |
| 24 | `    LogisticRegression(max_iter=1000).fit(features(A), (ya == 2)` | Choose the estimator and its tuning options. |
| 25 | `    .astype(int))` | Supply arguments to the preceding operation. |
| 26 | `)` | Close the preceding multi-line operation. |
| 27 | `(` | Print the result shown in the output box. |
| 28 | `    print('New binary-task test accuracy:', head.score(features(` | Supply arguments to the preceding operation. |
| 29 | `    B), (yb == 2).astype(int)))` | Supply arguments to the preceding operation. |
| 30 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
New binary-task test accuracy: 0.8947368421052632
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

A binary head is trained on frozen source features and evaluated on independent held-out rows.

## Try it yourself

Compare the transferred representation with a binary model trained directly on raw scaled features.

## Important check

Reusing a representation is feature extraction; updating source parameters would be fine-tuning.

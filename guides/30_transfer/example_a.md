# TRANSFER LEARNING / FINE-TUNING — Example A

**Train and freeze a source representation**  
Book page: **124** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/30_transfer/example_a.py) · [All topics](../../README.md)

## What this example teaches

Reuse parameters learned on a source task for a target task. Feature extraction freezes a pretrained representation; fine-tuning updates some or all transferred parameters. A new task head often replaces the original output layer.

## Python code

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
X, y = load_iris(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
source = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=3000, random_state=42)
source.fit(Xs, y)
features = np.maximum(0, Xs @ source.coefs_[0] + source.intercepts_[0])
print("Frozen hidden-feature shape:", features.shape)
# This is a source-fit demonstration, not target-test performance.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.datasets import load_iris` | Import load_iris. |
| 3 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 4 | `from sklearn.neural_network import MLPClassifier` | Import MLPClassifier. |
| 5 | `X, y = load_iris(return_X_y=True)` | Load or generate the example dataset. |
| 6 | `Xs = StandardScaler().fit_transform(X)` | Learn the transformation and apply it. |
| 7 | `source = (` | Assign source for the next step. |
| 8 | `    MLPClassifier(hidden_layer_sizes=(8,), solver='lbfgs',` | Choose the estimator and its tuning options. |
| 9 | `    max_iter=3000, random_state=42)` | Fix the seed and other options for repeatability. |
| 10 | `)` | Close the preceding multi-line operation. |
| 11 | `source.fit(Xs, y)` | Learn parameters from the training data. |
| 12 | `features = (` | Extract frozen hidden-layer features. |
| 13 | `    np.maximum(0, Xs @ source.coefs_[0] + source.intercepts_[0])` | Supply arguments to the preceding operation. |
| 14 | `)` | Close the preceding multi-line operation. |
| 15 | `print('Frozen hidden-feature shape:', features.shape)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Frozen hidden-feature shape: (150, 8)
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

A fitted source MLP is used only to extract ReLU hidden features. The printed array shape shows the representation width.

## Try it yourself

Compare hidden widths and verify that the feature extraction uses the learned weights and biases.

## Important check

This demonstration fits the source model; it does not report independent target-task performance.

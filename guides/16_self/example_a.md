# SELF-SUPERVISED LEARNING — Example A

**Masked-feature reconstruction**  
Book page: **68** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/16_self/example_a.py) · [All topics](../../README.md)

## What this example teaches

Creates a prediction task from raw inputs, such as reconstructing masked content, predicting the next token, or matching augmented views. No manually supplied labels are needed for pretraining. It is often treated as a form of unsupervised learning.

## Python code

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
rng = np.random.default_rng(42)
z = rng.normal(size=(600, 1))
X = z * np.array([[1., 2., 3.]])
X += rng.normal(scale=0.1, size=X.shape)
Xtr, Xte = train_test_split(X, random_state=42)
# Hide the middle feature; targets come from raw X.
visible, hidden = [0, 2], 1
model = make_pipeline(StandardScaler(), Ridge(alpha=1))
model.fit(Xtr[:, visible], Xtr[:, hidden])
pred = model.predict(Xte[:, visible])
print("Hidden-feature MSE:",
      mean_squared_error(Xte[:, hidden], pred))
# No manually annotated target labels are used.
# This tests reconstruction, not downstream transfer.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.model_selection import train_test_split` | Import train/test splitting utilities. |
| 3 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 4 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 5 | `from sklearn.linear_model import Ridge` | Import Ridge. |
| 6 | `from sklearn.metrics import mean_squared_error` | Import squared-error evaluation. |
| 7 | `rng = np.random.default_rng(42)` | Create a reproducible random generator. |
| 8 | `z = rng.normal(size=(600, 1))` | Create a shared latent signal for the features. |
| 9 | `X = z * np.array([[1.0, 2.0, 3.0]])` | Build or transform the input feature matrix. |
| 10 | `X += rng.normal(scale=0.1, size=X.shape)` | Add noise to the synthetic feature matrix. |
| 11 | `Xtr, Xte = train_test_split(X, random_state=42)` | Reserve unseen data for testing. |
| 12 | `visible, hidden = ([0, 2], 1)` | Select visible inputs and a hidden target. |
| 13 | `model = make_pipeline(StandardScaler(), Ridge(alpha=1))` | Chain preprocessing with the estimator. |
| 14 | `model.fit(Xtr[:, visible], Xtr[:, hidden])` | Learn parameters from the training data. |
| 15 | `pred = model.predict(Xte[:, visible])` | Predict from data the model has not fitted. |
| 16 | `(` | Print the result shown in the output box. |
| 17 | `    print('Hidden-feature MSE:', mean_squared_error(Xte[:,` | Compute mean squared prediction or reconstruction error. |
| 18 | `    hidden], pred))` | Supply arguments to the preceding operation. |
| 19 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
Hidden-feature MSE: 0.01278692180861263
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The target is a feature hidden from the inputs. Reconstruction labels are generated directly from the raw data.

## Try it yourself

Hide another band and compare reconstruction MSE while keeping test rows fixed.

## Important check

Do not accidentally include the hidden feature in the model input.

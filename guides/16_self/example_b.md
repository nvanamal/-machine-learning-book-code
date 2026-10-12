# SELF-SUPERVISED LEARNING — Example B

**Denoising reconstruction without manual labels**  
Book page: **69** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/16_self/example_b.py) · [All topics](../../README.md)

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
z = rng.normal(size=(500, 1))
X = z * np.array([[1., 2., 3.]]) + rng.normal(0, 0.05, (500, 3))
Xtr, Xte = train_test_split(X, random_state=42)
noisy_tr = Xtr + rng.normal(0, 0.3, Xtr.shape)
noisy_te = Xte + rng.normal(0, 0.3, Xte.shape)
model = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(noisy_tr, Xtr)
print("Noisy MSE:", mean_squared_error(Xte, noisy_te))
print("Denoised MSE:", mean_squared_error(Xte, model.predict(noisy_te)))
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
| 8 | `z = rng.normal(size=(500, 1))` | Create a shared latent signal for the features. |
| 9 | `X = (` | Build or transform the input feature matrix. |
| 10 | `    z * np.array([[1.0, 2.0, 3.0]]) + rng.normal(0, 0.05, (500,` | Supply arguments to the preceding operation. |
| 11 | `    3))` | Supply arguments to the preceding operation. |
| 12 | `)` | Close the preceding multi-line operation. |
| 13 | `Xtr, Xte = train_test_split(X, random_state=42)` | Reserve unseen data for testing. |
| 14 | `noisy_tr = Xtr + rng.normal(0, 0.3, Xtr.shape)` | Corrupt inputs; clean data remain targets. |
| 15 | `noisy_te = Xte + rng.normal(0, 0.3, Xte.shape)` | Corrupt inputs; clean data remain targets. |
| 16 | `model = (` | Fit model parameters using training inputs. |
| 17 | `    make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(noisy_tr` | Standardize using statistics fitted on training data. |
| 18 | `    , Xtr)` | Supply arguments to the preceding operation. |
| 19 | `)` | Close the preceding multi-line operation. |
| 20 | `print('Noisy MSE:', mean_squared_error(Xte, noisy_te))` | Print the result shown in the output box. |
| 21 | `(` | Print the result shown in the output box. |
| 22 | `    print('Denoised MSE:', mean_squared_error(Xte, model.predict` | Compute mean squared prediction or reconstruction error. |
| 23 | `    (noisy_te)))` | Supply arguments to the preceding operation. |
| 24 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
Noisy MSE: 0.08537804431176137
Denoised MSE: 0.03755657542245105
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Noisy inputs predict clean targets. Compare denoised MSE with the error already present in the noisy inputs.

## Try it yourself

Increase noise standard deviation and examine when denoising ceases to help.

## Important check

This reconstruction test does not establish that learned features transfer to a new task.

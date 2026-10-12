# SELF-SUPERVISED LEARNING — Example C

**Next-sample prediction from a raw time signal**  
Book page: **70** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/16_self/example_c.py) · [All topics](../../README.md)

## What this example teaches

Creates a prediction task from raw inputs, such as reconstructing masked content, predicting the next token, or matching augmented views. No manually supplied labels are needed for pretraining. It is often treated as a form of unsupervised learning.

## Python code

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
signal = np.sin(np.arange(700) * 0.08)
window, cutoff = 8, 500
# Inputs end at t-1; the generated target is the next sample t.
train_t = np.arange(window, cutoff)
test_t = np.arange(cutoff, len(signal))
def windows(times): return np.array([signal[t-window:t] for t in times])
model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
model.fit(windows(train_t), signal[train_t])
print("One-step MSE:", mean_squared_error(signal[test_t],
      model.predict(windows(test_t))))
# Rolling one-step test uses observed past values, not future samples.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `from sklearn.pipeline import make_pipeline` | Import the preprocessing/model pipeline. |
| 3 | `from sklearn.preprocessing import StandardScaler` | Import feature standardization. |
| 4 | `from sklearn.linear_model import Ridge` | Import Ridge. |
| 5 | `from sklearn.metrics import mean_squared_error` | Import squared-error evaluation. |
| 6 | `signal = np.sin(np.arange(700) * 0.08)` | Generate a raw signal with no manual labels. |
| 7 | `window, cutoff = (8, 500)` | Define history length and time cutoff. |
| 8 | `train_t = np.arange(window, cutoff)` | Choose target times before the cutoff. |
| 9 | `test_t = np.arange(cutoff, len(signal))` | Choose target times at or after the cutoff. |
| 10 | `` |  |
| 11 | `def windows(times):` | Define a reusable helper function. |
| 12 | `    return np.array([signal[t - window:t] for t in times])` | Return the helper result to its caller. |
| 13 | `model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))` | Chain preprocessing with the estimator. |
| 14 | `model.fit(windows(train_t), signal[train_t])` | Learn parameters from the training data. |
| 15 | `(` | Print the result shown in the output box. |
| 16 | `    print('One-step MSE:', mean_squared_error(signal[test_t],` | Compute mean squared prediction or reconstruction error. |
| 17 | `    model.predict(windows(test_t))))` | Supply arguments to the preceding operation. |
| 18 | `)` | Close the preceding multi-line operation. |

## Recorded output from the book

```text
One-step MSE: 3.89353571924673e-08
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Past observed samples form a window for predicting the next sample. The train/test split respects time.

## Try it yourself

Increase the window length and compare with a predict-the-previous-value baseline.

## Important check

Rolling one-step evaluation uses observed past test values; it differs from multi-step free-running forecasting.

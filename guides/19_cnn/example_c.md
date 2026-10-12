# CONVOLUTIONAL NEURAL NETWORK (CNN) — Example C

**A 1-D convolutional spectral feature**  
Book page: **82** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/19_cnn/example_c.py) · [All topics](../../README.md)

## What this example teaches

A neural architecture with local filters whose weights are shared across spatial positions. Filters produce feature maps; pooling or stride changes resolution. A learned head maps features to a task output.

## Python code

```python
import numpy as np
signal = np.array([0., 0., 1., 3., 1., 0., 0.])
kernel = np.array([-1., 0., 1.])
feature = np.array([np.sum(signal[i:i+3] * kernel) for i in range(5)])
print("Local gradient feature:", feature)
print("Largest response:", np.max(np.abs(feature)))
# Shared fixed kernel; learning filters requires backpropagation.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `signal = np.array([0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0])` | Generate a raw signal with no manual labels. |
| 3 | `kernel = np.array([-1.0, 0.0, 1.0])` | Assign kernel for the next step. |
| 4 | `feature = (` | Assign feature for the next step. |
| 5 | `    np.array([np.sum(signal[i:i + 3] * kernel) for i in range(5)` | Supply arguments to the preceding operation. |
| 6 | `    ])` | Supply arguments to the preceding operation. |
| 7 | `)` | Close the preceding multi-line operation. |
| 8 | `print('Local gradient feature:', feature)` | Print the result shown in the output box. |
| 9 | `print('Largest response:', np.max(np.abs(feature)))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Local gradient feature: [ 1.  3.  0. -3. -1.]
Largest response: 3.0
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

A fixed three-value kernel measures local spectral differences. Large magnitudes mark steep local changes.

## Try it yourself

Smooth the signal first and compare responses when small noise is added.

## Important check

These are fixed features. Training a CNN requires learning filters with a loss and backpropagation.

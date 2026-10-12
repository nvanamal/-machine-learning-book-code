# GATED RECURRENT UNIT (GRU) — Example C

**A scalar GRU sequence**  
Book page: **98** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/23_gru/example_c.py) · [All topics](../../README.md)

## What this example teaches

A gated recurrent model using an update gate, a reset gate and one hidden state. The update gate balances old memory with a candidate; the reset gate controls prior-state influence when forming the candidate.

## Python code

```python
import numpy as np
def sigmoid(x): return 1 / (1 + np.exp(-x))
h = 0.0
for x in [1., 0., -1.]:
    z = sigmoid(x + h)
    r = sigmoid(x - h)
    candidate = np.tanh(x + r * h)
    h = z * h + (1 - z) * candidate
    print("Hidden:", round(h, 4))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `` |  |
| 3 | `def sigmoid(x):` | Define a reusable helper function. |
| 4 | `    return 1 / (1 + np.exp(-x))` | Return the helper result to its caller. |
| 5 | `h = 0.0` | Compute or initialize the hidden representation. |
| 6 | `for x in [1.0, 0.0, -1.0]:` | Repeat episodes or steps of interaction. |
| 7 | `    z = sigmoid(x + h)` | Create a shared latent signal for the features. |
| 8 | `    r = sigmoid(x - h)` | Compute the reset gate. |
| 9 | `    candidate = np.tanh(x + r * h)` | Compute new candidate hidden-state content. |
| 10 | `    h = z * h + (1 - z) * candidate` | Compute or initialize the hidden representation. |
| 11 | `    print('Hidden:', round(h, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Hidden: 0.2048
Hidden: 0.154
Hidden: -0.4755
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The GRU recomputes gates, forms a candidate and blends it with old memory at each step.

## Try it yourself

Reverse the input order and compare the hidden trajectory.

## Important check

This forward toy uses fixed coefficients; it is not a trained sequence predictor.

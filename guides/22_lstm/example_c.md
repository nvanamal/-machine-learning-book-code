# LONG SHORT-TERM MEMORY (LSTM) — Example C

**A scalar LSTM across a sequence**  
Book page: **94** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/22_lstm/example_c.py) · [All topics](../../README.md)

## What this example teaches

A gated recurrent architecture with a cell state and hidden state. Input, forget and output gates control writing, retention and exposure of memory. Gates help manage long dependencies but do not guarantee learning them.

## Python code

```python
import numpy as np
def sigmoid(z): return 1 / (1 + np.exp(-z))
c, h = 0.0, 0.0
for x in [1., 0., -1.]:
    i = sigmoid(x + 0.2 * h)
    f = sigmoid(1 + 0.1 * h)
    o = sigmoid(x)
    g = np.tanh(x + 0.3 * h)
    c = f * c + i * g
    h = o * np.tanh(c)
    print("Cell/hidden:", round(c, 4), round(h, 4))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `` |  |
| 3 | `def sigmoid(z):` | Define a reusable helper function. |
| 4 | `    return 1 / (1 + np.exp(-z))` | Return the helper result to its caller. |
| 5 | `c, h = (0.0, 0.0)` | Assign c, h for the next step. |
| 6 | `for x in [1.0, 0.0, -1.0]:` | Repeat episodes or steps of interaction. |
| 7 | `    i = sigmoid(x + 0.2 * h)` | Compute the input/write gate. |
| 8 | `    f = sigmoid(1 + 0.1 * h)` | Compute the forget/retention gate. |
| 9 | `    o = sigmoid(x)` | Compute the output gate. |
| 10 | `    g = np.tanh(x + 0.3 * h)` | Compute the candidate memory content. |
| 11 | `    c = f * c + i * g` | Update or initialize the memory cell. |
| 12 | `    h = o * np.tanh(c)` | Compute or initialize the hidden representation. |
| 13 | `    print('Cell/hidden:', round(c, 4), round(h, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Cell/hidden: 0.5568 0.3696
Cell/hidden: 0.4683 0.2184
Cell/hidden: 0.141 0.0377
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Input, forget and output gates are recomputed at each step, then the cell and hidden states are updated.

## Try it yourself

Reset both states and compare with a run that begins from nonzero states.

## Important check

The weights are fixed in this toy sequence; no training performance is claimed.

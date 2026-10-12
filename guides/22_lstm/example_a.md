# LONG SHORT-TERM MEMORY (LSTM) — Example A

**One gated memory update**  
Book page: **92** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/22_lstm/example_a.py) · [All topics](../../README.md)

## What this example teaches

A gated recurrent architecture with a cell state and hidden state. Input, forget and output gates control writing, retention and exposure of memory. Gates help manage long dependencies but do not guarantee learning them.

## Python code

```python
import numpy as np
old_c = np.array([1., -1.])
f = np.array([0.9, 0.5]); i = np.array([0.1, 0.8])
g = np.array([0.5, 0.2]); o = np.array([0.7, 0.9])
c = f * old_c + i * g
h = o * np.tanh(c)
print("Cell:", np.round(c, 4))
print("Hidden:", np.round(h, 4))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `old_c = np.array([1.0, -1.0])` | Assign old_c for the next step. |
| 3 | `f = np.array([0.9, 0.5])` | Compute the forget/retention gate. |
| 4 | `i = np.array([0.1, 0.8])` | Compute the input/write gate. |
| 5 | `g = np.array([0.5, 0.2])` | Compute the candidate memory content. |
| 6 | `o = np.array([0.7, 0.9])` | Compute the output gate. |
| 7 | `c = f * old_c + i * g` | Update or initialize the memory cell. |
| 8 | `h = o * np.tanh(c)` | Compute or initialize the hidden representation. |
| 9 | `print('Cell:', np.round(c, 4))` | Print the result shown in the output box. |
| 10 | `print('Hidden:', np.round(h, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Cell: [ 0.95 -0.34]
Hidden: [ 0.5178 -0.2947]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The cell combines retained old memory and a gated candidate; the hidden output exposes a gated tanh of the cell.

## Try it yourself

Set the input gate to zero and inspect the memory retained by the forget gate.

## Important check

Cell state and hidden state are different quantities.

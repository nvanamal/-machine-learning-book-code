# LONG SHORT-TERM MEMORY (LSTM) — Example B

**Forget gates change memory retention**  
Book page: **93** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/22_lstm/example_b.py) · [All topics](../../README.md)

## What this example teaches

A gated recurrent architecture with a cell state and hidden state. Input, forget and output gates control writing, retention and exposure of memory. Gates help manage long dependencies but do not guarantee learning them.

## Python code

```python
import numpy as np
for f in [0.5, 0.95]:
    c = 1.0
    for t in range(10):
        c = f * c  # no new input writes
    print("Forget gate", f, "retained memory:", round(c, 6))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `for f in [0.5, 0.95]:` | Repeat episodes or steps of interaction. |
| 3 | `    c = 1.0` | Update or initialize the memory cell. |
| 4 | `    for t in range(10):` | Repeat episodes or steps of interaction. |
| 5 | `        c = f * c` | Update or initialize the memory cell. |
| 6 | `    print('Forget gate', f, 'retained memory:', round(c, 6))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Forget gate 0.5 retained memory: 0.000977
Forget gate 0.95 retained memory: 0.598737
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Repeated retention multiplies memory by the forget factor. A factor near one preserves more over ten steps.

## Try it yourself

Add a small candidate write at every step and inspect the resulting memory trajectory.

## Important check

A fixed high forget gate does not by itself establish successful learning of long dependencies.

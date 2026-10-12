# GATED RECURRENT UNIT (GRU) — Example B

**Reset gate controls the candidate**  
Book page: **97** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/23_gru/example_b.py) · [All topics](../../README.md)

## What this example teaches

A gated recurrent model using an update gate, a reset gate and one hidden state. The update gate balances old memory with a candidate; the reset gate controls prior-state influence when forming the candidate.

## Python code

```python
import numpy as np
x, old = 0.4, 1.0
for r in [0.0, 1.0]:
    candidate = np.tanh(x + r * old)
    print("Reset gate", r, "candidate:", round(candidate, 4))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `x, old = (0.4, 1.0)` | Assign x, old for the next step. |
| 3 | `for r in [0.0, 1.0]:` | Repeat episodes or steps of interaction. |
| 4 | `    candidate = np.tanh(x + r * old)` | Compute new candidate hidden-state content. |
| 5 | `    print('Reset gate', r, 'candidate:', round(candidate, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Reset gate 0.0 candidate: 0.3799
Reset gate 1.0 candidate: 0.8854
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The reset gate changes how much old state enters the candidate computation.

## Try it yourself

Try a negative old state and inspect the candidate for several reset values.

## Important check

Reset and update gates serve different roles; they are not interchangeable.

# GATED RECURRENT UNIT (GRU) — Example A

**Update-gate interpolation**  
Book page: **96** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/23_gru/example_a.py) · [All topics](../../README.md)

## What this example teaches

A gated recurrent model using an update gate, a reset gate and one hidden state. The update gate balances old memory with a candidate; the reset gate controls prior-state influence when forming the candidate.

## Python code

```python
import numpy as np
old = np.array([1., -1.]); candidate = np.array([0.2, 0.5])
for z in [0.0, 0.5, 1.0]:
    h = z * old + (1 - z) * candidate
    print("Retention gate", z, ":", h)
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `old = np.array([1.0, -1.0])` | Assign old for the next step. |
| 3 | `candidate = np.array([0.2, 0.5])` | Compute new candidate hidden-state content. |
| 4 | `for z in [0.0, 0.5, 1.0]:` | Repeat episodes or steps of interaction. |
| 5 | `    h = z * old + (1 - z) * candidate` | Compute or initialize the hidden representation. |
| 6 | `    print('Retention gate', z, ':', h)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Retention gate 0.0 : [0.2 0.5]
Retention gate 0.5 : [ 0.6  -0.25]
Retention gate 1.0 : [ 1. -1.]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

With the retention-gate convention, z=1 keeps the old state and z=0 uses the candidate state.

## Try it yourself

Use a separate z value for each feature to see elementwise interpolation.

## Important check

Libraries may parameterize equivalent GRU equations differently; verify conventions before porting weights.

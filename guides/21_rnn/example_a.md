# RECURRENT NEURAL NETWORK (RNN) — Example A

**Scalar recurrent state**  
Book page: **88** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/21_rnn/example_a.py) · [All topics](../../README.md)

## What this example teaches

Processes a sequence by repeatedly updating a hidden state with shared parameters. The state summarizes prior inputs and supports time-dependent outputs. Training commonly uses backpropagation through time.

## Python code

```python
import numpy as np
seq = [1., 0., -1.]
h = 0.0
states = []
for x in seq:
    h = np.tanh(0.5 * x + 0.8 * h)
    states.append(h)
print("States:", np.round(states, 4))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `seq = [1.0, 0.0, -1.0]` | Assign seq for the next step. |
| 3 | `h = 0.0` | Compute or initialize the hidden representation. |
| 4 | `states = []` | Assign states for the next step. |
| 5 | `for x in seq:` | Repeat episodes or steps of interaction. |
| 6 | `    h = np.tanh(0.5 * x + 0.8 * h)` | Compute or initialize the hidden representation. |
| 7 | `    states.append(h)` | Record the episode result for summarization. |
| 8 | `print('States:', np.round(states, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
States: [ 0.4621  0.3537 -0.2137]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Each state combines the new input with the previous state using the same coefficients at every step.

## Try it yourself

Reverse the input sequence and compare the final state.

## Important check

A sequence model depends on order; treating rows as interchangeable loses temporal meaning.

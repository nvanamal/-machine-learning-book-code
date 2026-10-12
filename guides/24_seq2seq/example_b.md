# ENCODER-DECODER / SEQ2SEQ — Example B

**Decode conditionally on a context**  
Book page: **101** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/24_seq2seq/example_b.py) · [All topics](../../README.md)

## What this example teaches

An encoder represents an input sequence; a decoder produces an output sequence conditioned on that representation. Outputs may have a different length. Training often uses previous true tokens; inference uses generated tokens.

## Python code

```python
import numpy as np
context, previous = 0.4, 0.0
outputs = []
for step in range(3):
    previous = np.tanh(context + 0.3 * previous)
    outputs.append(previous)
print("Decoder outputs:", np.round(outputs, 4))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `context, previous = (0.4, 0.0)` | Assign context, previous for the next step. |
| 3 | `outputs = []` | Assign outputs for the next step. |
| 4 | `for step in range(3):` | Repeat episodes or steps of interaction. |
| 5 | `    previous = np.tanh(context + 0.3 * previous)` | Assign previous for the next step. |
| 6 | `    outputs.append(previous)` | Record the episode result for summarization. |
| 7 | `print('Decoder outputs:', np.round(outputs, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Decoder outputs: [0.3799 0.473  0.4944]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The decoder repeatedly conditions on a context and its previous output to generate a sequence.

## Try it yourself

Change context while keeping the initial decoder state fixed.

## Important check

This deterministic scalar demonstration is not a trained translation or text model.

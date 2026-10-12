# ENCODER-DECODER / SEQ2SEQ — Example A

**Encode a sequence into a context**  
Book page: **100** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/24_seq2seq/example_a.py) · [All topics](../../README.md)

## What this example teaches

An encoder represents an input sequence; a decoder produces an output sequence conditioned on that representation. Outputs may have a different length. Training often uses previous true tokens; inference uses generated tokens.

## Python code

```python
import numpy as np
h = 0.0
for x in [0.2, 0.5, -0.1]:
    h = np.tanh(x + 0.5 * h)
context = h
print("Encoder context:", round(context, 4))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `h = 0.0` | Compute or initialize the hidden representation. |
| 3 | `for x in [0.2, 0.5, -0.1]:` | Repeat episodes or steps of interaction. |
| 4 | `    h = np.tanh(x + 0.5 * h)` | Compute or initialize the hidden representation. |
| 5 | `context = h` | Store the encoded or attention-derived context. |
| 6 | `print('Encoder context:', round(context, 4))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Encoder context: 0.1665
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

A recurrent encoder compresses an input sequence into a final context value.

## Try it yourself

Append another input step and observe how the context changes.

## Important check

A fixed-size context can lose detail for long inputs; attention provides an alternative.

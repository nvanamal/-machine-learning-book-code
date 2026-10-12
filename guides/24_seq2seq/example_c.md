# ENCODER-DECODER / SEQ2SEQ — Example C

**Teacher forcing shifts targets by one position**  
Book page: **102** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/24_seq2seq/example_c.py) · [All topics](../../README.md)

## What this example teaches

An encoder represents an input sequence; a decoder produces an output sequence conditioned on that representation. Outputs may have a different length. Training often uses previous true tokens; inference uses generated tokens.

## Python code

```python
import numpy as np
BOS, EOS = 1, 2
target = np.array([7, 8, EOS])
decoder_input = np.r_[BOS, target[:-1]]
print("Decoder input:", decoder_input)
print("Expected next tokens:", target)
# During inference, previous predictions replace true prior tokens.
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `BOS, EOS = (1, 2)` | Assign BOS, EOS for the next step. |
| 3 | `target = np.array([7, 8, EOS])` | Assign target for the next step. |
| 4 | `decoder_input = np.r_[BOS, target[:-1]]` | Shift targets to form previous-token inputs. |
| 5 | `print('Decoder input:', decoder_input)` | Print the result shown in the output box. |
| 6 | `print('Expected next tokens:', target)` | Print the result shown in the output box. |

## Recorded output from the book

```text
Decoder input: [1 7 8]
Expected next tokens: [7 8 2]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

The decoder input starts with BOS and then uses preceding target tokens; expected outputs retain EOS.

## Try it yourself

Build the shifted input for another target sequence and verify alignment position by position.

## Important check

During inference, prior predictions replace true targets. Future target tokens must not be provided.

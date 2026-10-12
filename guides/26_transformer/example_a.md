# TRANSFORMER ARCHITECTURE — Example A

**Sinusoidal positional features**  
Book page: **108** · Author: **Dr. Nagendrababu Vanamala**

[Download / view the Python file](../../examples/26_transformer/example_a.py) · [All topics](../../README.md)

## What this example teaches

Combines attention, positionwise feed-forward layers, residual paths and normalization. Encoder, decoder and encoder-decoder variants use different masks and roles. The basic architecture does not require recurrence.

## Python code

```python
import numpy as np
length, width = 4, 6
pos = np.arange(length)[:, None]
scale = 10000 ** (np.arange(0, width, 2) / width)
PE = np.zeros((length, width))
PE[:, 0::2] = np.sin(pos / scale)
PE[:, 1::2] = np.cos(pos / scale)
print("Position features:", np.round(PE, 3))
```

## Line-by-line guide

The following layout matches the formatted code in the book; the downloadable script has equivalent Python operations.

| Line | Python code | Explanation |
| --- | --- | --- |
| 1 | `import numpy as np` | Import numerical array tools. |
| 2 | `length, width = (4, 6)` | Assign length, width for the next step. |
| 3 | `pos = np.arange(length)[:, None]` | Assign pos for the next step. |
| 4 | `scale = 10000 ** (np.arange(0, width, 2) / width)` | Assign scale for the next step. |
| 5 | `PE = np.zeros((length, width))` | Allocate the positional-encoding matrix. |
| 6 | `PE[:, 0::2] = np.sin(pos / scale)` | Assign PE[:, 0::2] for the next step. |
| 7 | `PE[:, 1::2] = np.cos(pos / scale)` | Assign PE[:, 1::2] for the next step. |
| 8 | `print('Position features:', np.round(PE, 3))` | Print the result shown in the output box. |

## Recorded output from the book

```text
Position features: [[ 0.     1.     0.     1.     0.     1.   ]
 [ 0.841  0.54   0.046  0.999  0.002  1.   ]
 [ 0.909 -0.416  0.093  0.996  0.004  1.   ]
 [ 0.141 -0.99   0.139  0.99   0.006  1.   ]]
```

Captured with NumPy 2.3.5 and scikit-learn 1.8.0. Outputs can vary with versions and numerical libraries.

## Reading the output

Even feature dimensions use sine and odd dimensions cosine. Positions share a range of frequency scales.

## Try it yourself

Compare the encodings of adjacent positions and verify the zero-position row.

## Important check

This is the original sinusoidal choice; modern models can use other position mechanisms.

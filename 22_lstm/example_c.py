# LONG SHORT-TERM MEMORY (LSTM) | EXAMPLE C
# A scalar LSTM across a sequence
# Dr. Nagendrababu Vanamala

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

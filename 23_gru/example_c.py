# GATED RECURRENT UNIT (GRU) | EXAMPLE C
# A scalar GRU sequence
# Dr. Nagendrababu Vanamala

import numpy as np
def sigmoid(x): return 1 / (1 + np.exp(-x))
h = 0.0
for x in [1., 0., -1.]:
    z = sigmoid(x + h)
    r = sigmoid(x - h)
    candidate = np.tanh(x + r * h)
    h = z * h + (1 - z) * candidate
    print("Hidden:", round(h, 4))

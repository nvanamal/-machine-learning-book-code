# LONG SHORT-TERM MEMORY (LSTM) | EXAMPLE A
# One gated memory update
# Dr. Nagendrababu Vanamala

import numpy as np
old_c = np.array([1., -1.])
f = np.array([0.9, 0.5]); i = np.array([0.1, 0.8])
g = np.array([0.5, 0.2]); o = np.array([0.7, 0.9])
c = f * old_c + i * g
h = o * np.tanh(c)
print("Cell:", np.round(c, 4))
print("Hidden:", np.round(h, 4))

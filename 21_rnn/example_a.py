# RECURRENT NEURAL NETWORK (RNN) | EXAMPLE A
# Scalar recurrent state
# Dr. Nagendrababu Vanamala

import numpy as np
seq = [1., 0., -1.]
h = 0.0
states = []
for x in seq:
    h = np.tanh(0.5 * x + 0.8 * h)
    states.append(h)
print("States:", np.round(states, 4))

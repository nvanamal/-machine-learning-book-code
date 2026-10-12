# ENCODER-DECODER / SEQ2SEQ | EXAMPLE A
# Encode a sequence into a context
# Dr. Nagendrababu Vanamala

import numpy as np
h = 0.0
for x in [0.2, 0.5, -0.1]:
    h = np.tanh(x + 0.5 * h)
context = h
print("Encoder context:", round(context, 4))

# ENCODER-DECODER / SEQ2SEQ | EXAMPLE B
# Decode conditionally on a context
# Dr. Nagendrababu Vanamala

import numpy as np
context, previous = 0.4, 0.0
outputs = []
for step in range(3):
    previous = np.tanh(context + 0.3 * previous)
    outputs.append(previous)
print("Decoder outputs:", np.round(outputs, 4))

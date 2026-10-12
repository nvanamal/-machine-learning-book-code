# RECURRENT NEURAL NETWORK (RNN) | EXAMPLE C
# Gradient propagation through time
# Dr. Nagendrababu Vanamala

import numpy as np
for scale in [0.5, 1.2]:
    gradient = 1.0
    for step in range(20):
        gradient *= scale
    print("Jacobian factor", scale, "after 20 steps:", round(gradient, 6))
# Simplified repeated Jacobian product, not a full RNN training loop.

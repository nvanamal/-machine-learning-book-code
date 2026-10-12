# GATED RECURRENT UNIT (GRU) | EXAMPLE B
# Reset gate controls the candidate
# Dr. Nagendrababu Vanamala

import numpy as np
x, old = 0.4, 1.0
for r in [0.0, 1.0]:
    candidate = np.tanh(x + r * old)
    print("Reset gate", r, "candidate:", round(candidate, 4))

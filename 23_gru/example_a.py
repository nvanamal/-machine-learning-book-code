# GATED RECURRENT UNIT (GRU) | EXAMPLE A
# Update-gate interpolation
# Dr. Nagendrababu Vanamala

import numpy as np
old = np.array([1., -1.]); candidate = np.array([0.2, 0.5])
for z in [0.0, 0.5, 1.0]:
    h = z * old + (1 - z) * candidate
    print("Retention gate", z, ":", h)

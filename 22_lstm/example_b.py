# LONG SHORT-TERM MEMORY (LSTM) | EXAMPLE B
# Forget gates change memory retention
# Dr. Nagendrababu Vanamala

import numpy as np
for f in [0.5, 0.95]:
    c = 1.0
    for t in range(10):
        c = f * c  # no new input writes
    print("Forget gate", f, "retained memory:", round(c, 6))

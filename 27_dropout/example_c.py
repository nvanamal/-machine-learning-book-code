# DROPOUT | EXAMPLE C
# Expected activation is approximately preserved
# Dr. Nagendrababu Vanamala

import numpy as np
rng = np.random.default_rng(42)
p = 0.4
samples = (rng.random(10000) >= p) / (1 - p)
print("Empirical mean:", round(samples.mean(), 3))
print("Theoretical mean:", 1.0)

# DROPOUT | EXAMPLE B
# Evaluation disables the training mask
# Dr. Nagendrababu Vanamala

import numpy as np
h = np.array([1., 2., 3.]); p = 0.5
training = False
rng = np.random.default_rng(42)
y = h * (rng.random(h.shape) >= p) / (1 - p) if training else h
print("Evaluation activations:", y)

# DROPOUT | EXAMPLE A
# Inverted dropout during training
# Dr. Nagendrababu Vanamala

import numpy as np
rng = np.random.default_rng(42)
h = np.ones(8); p = 0.25
mask = rng.random(h.shape) >= p
train_h = h * mask / (1 - p)
print("Mask:", mask.astype(int))
print("Training activations:", train_h)

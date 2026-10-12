# TRANSFORMER ARCHITECTURE | EXAMPLE A
# Sinusoidal positional features
# Dr. Nagendrababu Vanamala

import numpy as np
length, width = 4, 6
pos = np.arange(length)[:, None]
scale = 10000 ** (np.arange(0, width, 2) / width)
PE = np.zeros((length, width))
PE[:, 0::2] = np.sin(pos / scale)
PE[:, 1::2] = np.cos(pos / scale)
print("Position features:", np.round(PE, 3))

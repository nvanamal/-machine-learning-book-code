# ATTENTION MECHANISMS | EXAMPLE C
# Attention pooling retrieves similar keys
# Dr. Nagendrababu Vanamala

import numpy as np
keys = np.array([0., 1., 2.]); values = np.array([0., 1., 4.])
query = 1.2
scores = -0.5 * (keys - query) ** 2
weights = np.exp(scores - scores.max()); weights /= weights.sum()
print("Similarity weights:", np.round(weights, 3))
print("Weighted estimate:", round(weights @ values, 4))

# RECURRENT NEURAL NETWORK (RNN) | EXAMPLE B
# Batch sequence shapes
# Dr. Nagendrababu Vanamala

import numpy as np
rng = np.random.default_rng(42)
X = rng.normal(size=(2, 4, 3))
Wx = rng.normal(scale=0.2, size=(3, 5))
Wh = np.eye(5) * 0.5
h = np.zeros((2, 5))
for t in range(4):
    h = np.tanh(X[:, t] @ Wx + h @ Wh)
print("Batch/time/input shape:", X.shape)
print("Final hidden shape:", h.shape)

# SOFTMAX REGRESSION | EXAMPLE B
# Cross-entropy and score gradient
# Dr. Nagendrababu Vanamala

import numpy as np
z = np.array([1., 2., 0.])
p = np.exp(z - z.max()); p /= p.sum()
y = 1
loss = -np.log(p[y])
grad = p.copy(); grad[y] -= 1
print("Loss:", round(loss, 4))
print("Gradient:", np.round(grad, 4))

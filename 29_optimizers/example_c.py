# SGD, MOMENTUM & ADAM | EXAMPLE C
# Adam with bias correction
# Dr. Nagendrababu Vanamala

import numpy as np
theta, m, v = 5.0, 0.0, 0.0
b1, b2 = 0.9, 0.999
for t in range(1, 301):
    gradient = 2 * (theta - 2)
    m = b1 * m + (1 - b1) * gradient
    v = b2 * v + (1 - b2) * gradient ** 2
    mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
    theta -= 0.05 * mhat / (np.sqrt(vhat) + 1e-8)
print("Adam parameter:", round(theta, 6))
